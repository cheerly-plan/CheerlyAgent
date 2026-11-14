#!/usr/bin/env python
"""记忆规则层离线评测：不调模型、不连库，可在 CI 里跑。

为什么单独一层：记忆质量原来只有「喂真模型看最终记没记」一条链，CI 连不到
模型就只能降级跳过，闭环是断的。规则层是确定性的，用同一批标注的
``extracted`` 直接喂 ``_keep_extracted``，就能量化两件事：

* **误杀**（remember 用例被拦掉）：用户明确说过的长期事实没入库
* **漏放**（forget 用例被放行）：任务动作、一次性查询被当成长期记忆

用例集 ``cases/memory_rule_cases.json`` 只收「规则该管」的条目——开放式技术
提问（「这个报错怎么修」）需要语义判断，归模型层评测，放进这里只会制造假失败。

用法：
    python scripts/eval_memory_rules.py
    python scripts/eval_memory_rules.py --verbose     # 逐条打印归因
    python scripts/eval_memory_rules.py --json        # 机器可读输出
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CASES_PATH = ROOT / "scripts" / "cases" / "memory_rule_cases.json"


def _rules_hit(text: str) -> list[str]:
    """这条文本被哪些确定性规则命中。

    逐条复算而不是只问 ``_keep_extracted`` 的布尔结果：归因要说清「是哪条
    规则拦的、还是根本没规则覆盖」，否则补规则只能靠猜。
    """
    from ai_agent.memory import long_term as lt

    hits: list[str] = []
    body = (text or "").strip()
    if not body:
        return ["empty"]
    if lt._ONGOING.search(body):
        hits.append("_ONGOING")
    if lt._JOB_STATE.search(body):
        hits.append("_JOB_STATE")
    if lt._MARKET_CODE.search(body) and lt._domain_noise_enabled():
        hits.append("_MARKET_CODE")
    if lt._RECAP_HEAD.match(body):
        hits.append("_RECAP_HEAD")
    if lt._TEMPORAL.match(body):
        hits.append("_TEMPORAL")
    if lt._OBJECT_SUBJECT.match(body):
        hits.append("_OBJECT_SUBJECT")
    for mark in lt._TOOL_MARK:
        if mark in body:
            hits.append(f"_TOOL_MARK({mark})")
            break
    if lt._ASSISTANT_RECAP.search(body):
        hits.append("_ASSISTANT_RECAP")
    if lt._domain_noise_enabled():
        for mark in lt._DOC_QA:
            if mark in body:
                hits.append(f"_DOC_QA({mark})")
                break
    return hits


def _judge(case: dict) -> dict:
    from ai_agent.memory.long_term import ExtractedMemory, _keep_extracted

    items = [
        ExtractedMemory(
            kind=entry.get("kind", "episode"),
            facet=entry.get("facet", ""),
            content=entry["content"],
            op="add",
        )
        for entry in case.get("extracted") or []
    ]
    kept = [item for item in items if _keep_extracted(item)]
    dropped = [item for item in items if item not in kept]
    expect = case.get("expect")

    if expect == "remember":
        # 全部保留才算对：只要有一条被拦就是误杀
        ok = bool(items) and len(kept) == len(items)
    else:
        ok = not kept

    return {
        "id": case.get("id"),
        "expect": expect,
        "ok": ok,
        "user": case.get("user", ""),
        "note": case.get("note", ""),
        "kept": [item.content for item in kept],
        "dropped": [
            {"content": item.content, "rules": _rules_hit(item.content)}
            for item in dropped
        ],
    }


def run(cases: list[dict]) -> dict:
    results = [_judge(case) for case in cases]
    stat: dict[str, Counter] = {"remember": Counter(), "forget": Counter()}
    rule_hits: Counter = Counter()
    uncovered: list[dict] = []

    for result in results:
        bucket = stat[result["expect"]]
        bucket["total"] += 1
        if result["ok"]:
            bucket["pass"] += 1
        elif result["expect"] == "remember":
            bucket["false_drop"] += 1
            for entry in result["dropped"]:
                for rule in entry["rules"] or ["<no-rule>"]:
                    rule_hits[f"误杀:{rule}"] += 1
        else:
            bucket["false_keep"] += 1
            uncovered.append(result)

    def rate(bucket: Counter) -> float:
        return bucket["pass"] / bucket["total"] if bucket["total"] else 0.0

    return {
        "cases": len(results),
        "remember": dict(stat["remember"]),
        "forget": dict(stat["forget"]),
        "remember_rate": rate(stat["remember"]),
        "forget_rate": rate(stat["forget"]),
        "rule_attribution": dict(rule_hits.most_common()),
        "uncovered": uncovered,
        "results": results,
    }


def format_report(report: dict, *, verbose: bool = False) -> str:
    lines = ["# 记忆规则层离线评测", ""]
    rm = report["remember"]
    fg = report["forget"]
    lines.append(
        f"remember {rm.get('pass', 0)}/{rm.get('total', 0)}"
        f"（误杀 {rm.get('false_drop', 0)}）  "
        f"forget {fg.get('pass', 0)}/{fg.get('total', 0)}"
        f"（漏放 {fg.get('false_keep', 0)}）"
    )
    lines.append("")
    lines.append(f"该记未记率 {1 - report['remember_rate']:.1%}   "
                 f"不该记记了率 {1 - report['forget_rate']:.1%}")

    if report["rule_attribution"]:
        lines += ["", "## 误杀归因"]
        for rule, count in report["rule_attribution"].items():
            lines.append(f"- {rule}: {count}")

    if report["uncovered"]:
        lines += ["", "## 漏放（没有规则覆盖）"]
        for item in report["uncovered"]:
            lines.append(f"- #{item['id']} {item['user'][:50]}")

    if verbose:
        lines += ["", "## 逐条"]
        for result in report["results"]:
            mark = "PASS" if result["ok"] else "FAIL"
            lines.append(f"[{mark}] #{result['id']} {result['expect']} {result['user'][:40]}")
            for entry in result["dropped"]:
                lines.append(f"      拦: {entry['content'][:40]} ← {entry['rules']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="记忆规则层离线评测")
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-remember", type=float, default=0.95)
    parser.add_argument("--min-forget", type=float, default=0.80)
    args = parser.parse_args()

    path = Path(args.cases)
    if not path.exists():
        print(f"用例集不存在: {path}", file=sys.stderr)
        return 2
    report = run(json.loads(path.read_text(encoding="utf-8")))

    if args.json:
        print(json.dumps(
            {k: v for k, v in report.items() if k != "results"},
            ensure_ascii=False, indent=2,
        ))
    else:
        print(format_report(report, verbose=args.verbose))

    ok = (
        report["remember_rate"] >= args.min_remember
        and report["forget_rate"] >= args.min_forget
    )
    if not args.json:
        print()
        print("verdict:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
