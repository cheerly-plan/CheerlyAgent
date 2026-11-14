#!/usr/bin/env python
"""记忆抽取端到端评测：喂真实模型，量化「该记没记」与「不该记记了」。

与 ``eval_memory_rules`` 的分工：

* ``eval_memory_rules``  只测规则层，不调模型，进 CI
* 本脚本                测「模型 + 规则」的整体行为，需要模型

用例集 ``cases/memory_golden_cases.json`` 每条带 ``owner``：

* ``rule``   确定性规则应当处理（本脚本也会统计，用于确认规则没有误伤）
* ``model``  需要语义判断，规则不该管（如「这个报错怎么修」）
* ``either``  两种解读都合理，只记录不计分

用法：
    python scripts/eval_memory_golden.py
    python scripts/eval_memory_golden.py --owner model
    python scripts/eval_memory_golden.py --verbose --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CASES_PATH = ROOT / "scripts" / "cases" / "memory_golden_cases.json"


def _turn(user: str) -> list[dict]:
    return [
        {"role": "user", "content": user},
        # 助手回复只作上下文：抽取判的是「用户说了什么」，不依赖助手措辞
        {"role": "assistant", "content": "好的。"},
    ]


def run_case(case: dict) -> dict:
    from ai_agent.memory.long_term import extract_turn_memories

    started = time.monotonic()
    error = ""
    try:
        items = extract_turn_memories(_turn(case.get("user", "")))
    except Exception as exc:  # 单条失败不中断整轮
        items = []
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started

    expect = case.get("expect")
    if expect == "remember":
        # 只要抽出一条内容与用户自述相关的记忆就算命中；抽空即「该记没记」
        ok = bool(items)
    elif expect == "forget":
        ok = not items
    else:
        ok = True  # either 不计分

    return {
        "id": case.get("id"),
        "expect": expect,
        "owner": case.get("owner", ""),
        "ok": ok,
        "user": case.get("user", ""),
        "note": case.get("note", ""),
        "got": [
            {"kind": item.kind, "facet": item.facet, "content": item.content}
            for item in items
        ],
        "error": error,
        "seconds": round(elapsed, 2),
    }


def run(cases: list[dict]) -> dict:
    results = [run_case(case) for case in cases]
    stat: dict[str, Counter] = {"remember": Counter(), "forget": Counter()}
    by_owner: dict[str, Counter] = {}
    errors = [r for r in results if r["error"]]

    for result in results:
        owner = result["owner"] or "-"
        by_owner.setdefault(owner, Counter())
        by_owner[owner]["total"] += 1
        if result["expect"] not in stat:
            continue
        bucket = stat[result["expect"]]
        bucket["total"] += 1
        if result["ok"]:
            bucket["pass"] += 1
            by_owner[owner]["pass"] += 1
        elif result["expect"] == "remember":
            bucket["false_drop"] += 1
        else:
            bucket["false_keep"] += 1

    def rate(bucket: Counter) -> float:
        return bucket["pass"] / bucket["total"] if bucket["total"] else 0.0

    return {
        "cases": len(results),
        "remember": dict(stat["remember"]),
        "forget": dict(stat["forget"]),
        "remember_rate": rate(stat["remember"]),
        "forget_rate": rate(stat["forget"]),
        "by_owner": {k: dict(v) for k, v in by_owner.items()},
        "errors": errors,
        "failures": [r for r in results if not r["ok"]],
        "results": results,
        "mean_seconds": round(
            sum(r["seconds"] for r in results) / len(results), 2
        ) if results else 0.0,
    }


def format_report(report: dict, *, verbose: bool = False) -> str:
    rm, fg = report["remember"], report["forget"]
    lines = ["# 记忆抽取端到端评测（真实模型）", ""]
    lines.append(
        f"remember {rm.get('pass', 0)}/{rm.get('total', 0)}"
        f"（该记没记 {rm.get('false_drop', 0)}）  "
        f"forget {fg.get('pass', 0)}/{fg.get('total', 0)}"
        f"（不该记记了 {fg.get('false_keep', 0)}）"
    )
    lines.append("")
    lines.append(f"该记未记率 {1 - report['remember_rate']:.1%}   "
                 f"不该记记了率 {1 - report['forget_rate']:.1%}")
    lines.append(f"平均耗时 {report['mean_seconds']}s/条")

    if report["by_owner"]:
        lines += ["", "## 按归属"]
        for owner, counts in sorted(report["by_owner"].items()):
            total = counts.get("total", 0)
            ok = counts.get("pass", 0)
            lines.append(f"- {owner}: {ok}/{total}")

    if report["errors"]:
        lines += ["", "## 调用失败"]
        for item in report["errors"][:10]:
            lines.append(f"- #{item['id']} {item['error'][:80]}")

    if report["failures"]:
        lines += ["", "## 失败明细"]
        for item in report["failures"]:
            tag = "该记没记" if item["expect"] == "remember" else "不该记记了"
            lines.append(f"- [{tag}] #{item['id']} {item['user'][:44]}")
            for got in item["got"][:2]:
                lines.append(f"    抽出: [{got['facet'] or got['kind']}] {got['content'][:44]}")

    if verbose:
        lines += ["", "## 逐条"]
        for item in report["results"]:
            mark = "PASS" if item["ok"] else "FAIL"
            lines.append(
                f"[{mark}] #{item['id']} {item['expect']}/{item['owner']} "
                f"{item['user'][:38]}  → {len(item['got'])} 条"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="记忆抽取端到端评测")
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--owner", default="", help="只跑某类归属：rule / model / either")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-remember", type=float, default=0.85)
    parser.add_argument("--min-forget", type=float, default=0.80)
    args = parser.parse_args()

    path = Path(args.cases)
    if not path.exists():
        print(f"用例集不存在: {path}", file=sys.stderr)
        return 2
    cases = json.loads(path.read_text(encoding="utf-8"))
    if args.owner:
        cases = [c for c in cases if c.get("owner") == args.owner]
    # either 不计分，默认不浪费时间调模型
    cases = [c for c in cases if c.get("expect") in {"remember", "forget"}]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("没有可跑的用例", file=sys.stderr)
        return 2

    report = run(cases)
    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != "results"},
                         ensure_ascii=False, indent=2))
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
