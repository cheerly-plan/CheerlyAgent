#!/usr/bin/env python
"""多轮压缩评测：历史被压到预算内之后，追问还接不接得住。

「摘要压缩」最容易出的问题不是摘要写得好不好看，而是**压缩完之后追问
里指向的那些事实没了**。「我刚才说的那张卡，推荐的量化格式是哪个？」
必须还能从重建的上下文里找到 AWQ。

用例集 ``cases/multiturn_compress_cases.json`` 每条带：

* ``setup``    前几轮对话（会被压缩进摘要的那部分）
* ``followup`` 压缩后仍要回答的追问
* ``must``     回答必须能溯源到的关键事实（子串，来自 setup）

评测方式（纯确定性，不调模型、不连库）：

1. 用 ``short_term.compress`` 把 setup 压到预算内
2. 用 ``short_term.assemble`` 重建成下一轮真正拿到的上下文
3. 检查每条 ``must`` 是否还在上下文里（保留窗口 + 摘要）

这样测的是「压缩策略有没有丢事实」，与摘要模型写得好不好解耦——
``--summarize none`` 用确定性 crude_summary，任何环境都能跑。

用法：
    python scripts/eval_multiturn_compress.py
    python scripts/eval_multiturn_compress.py --keep-last 2
    python scripts/eval_multiturn_compress.py --keep-last 2 --summarize llm
    python scripts/eval_multiturn_compress.py --sweep 1,2,3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CASES_PATH = ROOT / "scripts" / "cases" / "multiturn_compress_cases.json"


def _norm(text: str) -> str:
    """去掉空白再比：中文分词差异不该导致「事实丢了」的误判。"""
    return "".join(str(text or "").split())


def _auto_budget(messages: list[dict], keep_last: int) -> int:
    """按内容自动定预算：只留最后 keep_last 轮，前面全部挤进摘要。

    用例本身只有几十 token，写死预算要么压不动、要么压过头。固定一个
    「保留最后 N 轮」的语义，再反推预算，每条用例的压缩强度才一致。
    """
    from ai_agent.memory.short_term import messages_tokens

    total = messages_tokens(messages)
    keep = messages[-keep_last * 2 :] if len(messages) > keep_last * 2 else messages
    return max(1, total - messages_tokens(keep) - 1)


def run_case(
    case: dict, *, budget: int, summarize=None, keep_last: int = 0
) -> dict:
    from ai_agent.memory.short_term import assemble, compress

    messages: list[dict] = []
    seq = 0
    for msg in case.get("setup") or []:
        seq += 1
        messages.append({**msg, "seq": seq})
    before = len(messages)
    if keep_last > 0:
        budget = _auto_budget(messages, keep_last)

    recent, summary, _cursor = compress(
        messages, max_tokens=budget, summarize=summarize
    )
    context = assemble("", summary, recent)
    haystack = _norm(
        " ".join(str(m.get("content") or "") for m in context)
    )

    missing = [token for token in case.get("must") or [] if _norm(token) not in haystack]
    return {
        "id": case.get("id"),
        "topic": case.get("topic", ""),
        "kind": case.get("kind", ""),
        "ok": not missing,
        "missing": missing,
        "must": list(case.get("must") or []),
        "setup_messages": before,
        "kept_messages": len(recent),
        "summarized": before - len(recent),
        "summary_chars": len(summary),
        "followup": case.get("followup", ""),
    }


def run(cases: list[dict], *, budget: int, summarize=None, keep_last: int = 0) -> dict:
    results = [
        run_case(case, budget=budget, summarize=summarize, keep_last=keep_last)
        for case in cases
    ]
    passed = sum(1 for r in results if r["ok"])
    by_kind: dict[str, list[int]] = {}
    for result in results:
        slot = by_kind.setdefault(result["kind"] or "-", [0, 0])
        slot[1] += 1
        if result["ok"]:
            slot[0] += 1
    return {
        "cases": len(results),
        "passed": passed,
        "rate": passed / len(results) if results else 0.0,
        "budget": budget,
        "keep_last": keep_last,
        "by_kind": {k: {"pass": v[0], "total": v[1]} for k, v in by_kind.items()},
        "failures": [r for r in results if not r["ok"]],
        "results": results,
    }


def format_report(report: dict, *, verbose: bool = False) -> str:
    lines = ["# 多轮压缩评测（追问事实保留率）", ""]
    scope = (
        f"保留最后 {report['keep_last']} 轮" if report.get("keep_last")
        else f"预算 {report['budget']} tokens"
    )
    lines.append(
        f"{scope}：{report['passed']}/{report['cases']} 条追问仍可溯源"
        f"（保留率 {report['rate']:.1%}）"
    )
    if report["by_kind"]:
        lines.append("")
        for kind, counts in sorted(report["by_kind"].items()):
            lines.append(f"- {kind}: {counts['pass']}/{counts['total']}")

    if report["failures"]:
        lines += ["", "## 丢事实"]
        for item in report["failures"]:
            lines.append(f"- #{item['id']} {item['topic']}")
            lines.append(f"    丢失: {', '.join(item['missing'])}")
            lines.append(
                f"    消息 {item['setup_messages']} → 保留 {item['kept_messages']}"
                f"，摘要 {item['summary_chars']} 字"
            )

    if verbose:
        lines += ["", "## 逐条"]
        for item in report["results"]:
            mark = "PASS" if item["ok"] else "FAIL"
            lines.append(
                f"[{mark}] #{item['id']} {item['topic']} "
                f"({item['kept_messages']}/{item['setup_messages']} 轮保留)"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="多轮压缩评测")
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--budget", type=int, default=400, help="压缩后允许的 token 预算")
    parser.add_argument("--sweep", default="", help="逗号分隔的多个预算，如 200,400,800")
    parser.add_argument(
        "--keep-last",
        type=int,
        default=2,
        help="保留最后 N 轮、其余压进摘要（按内容自动算预算，默认 2）。0 = 用 --budget",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--summarize",
        choices=["none", "llm"],
        default="none",
        help="none=确定性 crude_summary（不调模型）；llm=真实摘要链路",
    )
    parser.add_argument("--min-rate", type=float, default=0.85)
    args = parser.parse_args()

    path = Path(args.cases)
    if not path.is_absolute() and not path.exists():
        candidate = ROOT / "scripts" / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        print(f"用例集不存在: {path}", file=sys.stderr)
        return 2
    cases = json.loads(path.read_text(encoding="utf-8"))

    summarize = None
    if args.summarize == "llm":
        from ai_agent.memory.summarize import summarize_session

        summarize = summarize_session

    keep_lasts = (
        [int(x) for x in args.sweep.split(",") if x.strip()]
        if args.sweep
        else [args.keep_last]
    )
    budgets = [args.budget] * len(keep_lasts) if keep_lasts != [0] else [args.budget]
    reports = []
    worst = 1.0
    for budget, keep_last in zip(budgets, keep_lasts, strict=True):
        report = run(
            cases, budget=budget, summarize=summarize, keep_last=keep_last
        )
        reports.append(report)
        worst = min(worst, report["rate"])
        if not args.json:
            print(format_report(report, verbose=args.verbose))
            print()

    if args.json:
        print(json.dumps(
            [{k: v for k, v in r.items() if k != "results"} for r in reports],
            ensure_ascii=False, indent=2,
        ))
    else:
        ok = worst >= args.min_rate
        print(f"最差保留率 {worst:.1%}，通过线 {args.min_rate:.0%}")
        print("verdict:", "PASS" if ok else "FAIL")
    return 0 if worst >= args.min_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
