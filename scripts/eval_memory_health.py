"""记忆健康度报告：把抽取、噪声、召回合成一份可对比的指标。

与 ``eval_memory.py`` 的分工：
- ``eval_memory.py`` 跑单条黄金对话，看「该记的记了没」；
- 本脚本补三件它不直接回答的事：

1. **噪声注入率**：一批「看着像输入、其实不该记」的句子（一次性查询、
   任务状态、文档事实、短期行程……）送进抽取，统计被错误写入的比例；
2. **召回质量**：给定情景库与问句，算 Recall@k / Precision@k / MRR；
3. **排序健康度**：统计返回结果里「一次性内容」的占比，衡量召回被噪声
   污染的程度（降权后这个数应该下降）。

三项都走纯本地逻辑 + 可选模型调用，不写真实用户数据。

用法：
    uv run python scripts/eval_memory_health.py                 # 全部
    uv run python scripts/eval_memory_health.py --noise-only    # 只跑噪声注入
    uv run python scripts/eval_memory_health.py --json out.json # 机器可读
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CASES_DIR = Path(__file__).resolve().parent / "cases"
NOISE_CASES_PATH = CASES_DIR / "memory_noise_cases.json"
RECALL_CASES_PATH = CASES_DIR / "memory_recall_cases.json"


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ——— 1. 噪声注入 ———


def run_noise(cases: list[dict]) -> dict:
    """把「不该记」的句子当作一轮对话送进抽取，统计被写入的比例。"""
    from ai_agent.memory.long_term import is_transient_text

    rows: list[dict] = []
    for case in cases:
        text = case.get("text") or ""
        turn = [
            {"role": "user", "content": case.get("user") or text},
            {
                "role": "assistant",
                "content": case.get("assistant") or "好的。",
            },
        ]
        # 规则层是否已经拦住（零成本，先看这一层）
        blocked_by_rule = is_transient_text(text)
        rows.append(
            {
                "id": case.get("id") or "",
                "kind": case.get("kind") or "",
                # owner=model 的用例只含行业术语、没有通用结构特征，
                # 规则层本来就不该拦（交给抽取模型判断），不计入拦截率
                "owner": case.get("owner") or "rule",
                "blocked_by_rule": blocked_by_rule,
                "text": text[:60],
                "turn": turn,
            }
        )
    rule_rows = [row for row in rows if row["owner"] == "rule"]
    total = len(rule_rows)
    blocked = sum(1 for row in rule_rows if row["blocked_by_rule"])
    return {
        "total": total,
        "blocked_by_rule": blocked,
        "rule_rate": (blocked / total) if total else 0.0,
        "model_owned": [row["id"] for row in rows if row["owner"] == "model"],
        # 模型层的漏网率由调用方按需跑（成本高），这里只给出规则层结论
        "leaks": [row["id"] for row in rule_rows if not row["blocked_by_rule"]],
    }


# ——— 2. 召回质量 ———


def run_recall(data: dict, *, top_k: int) -> dict:
    """用确定性相关性给「库里的条目」排序，算 Recall@k / Precision@k / MRR。

    不连库、不调模型：把用例里的 items 当作检索结果全集，按
    ``progressive.relevance`` + ``rank_score`` 排序，看期望条目排在哪。
    """
    from ai_agent.memory.progressive import rank_score, relevance
    from ai_agent.storage.db import MemoryItem

    items = [
        MemoryItem(
            id=str(raw.get("id")),
            user_id="eval",
            text=str(raw.get("text") or ""),
            created_at=str(raw.get("created_at") or "2026-11-14T00:00:00+00:00"),
            facet=str(raw.get("facet") or "other"),
        )
        for raw in data.get("items") or []
    ]
    rows: list[dict] = []
    for query in data.get("queries") or []:
        text = str(query.get("query") or "")
        expect = {str(item) for item in query.get("expect") or []}
        demote = {str(item) for item in query.get("demote") or []}
        scored = [(item, rank_score(item, relevance(text, item))) for item in items]
        ranked = [item for item, _ in sorted(scored, key=lambda pair: -pair[1])]
        top = ranked[:top_k]
        hit = [item for item in top if item.id in expect]
        rank = next(
            (index + 1 for index, item in enumerate(ranked) if item.id in expect),
            None,
        )
        # demote：断言这些条目被挤出 top_k（一次性内容应被降权，而不是被命中）
        still_top = [item.id for item in top if item.id in demote]
        rows.append(
            {
                "id": query.get("id") or "",
                "query": text,
                "expect": sorted(expect),
                "demote": sorted(demote),
                "top": [item.id for item in top],
                "recall": (len(hit) / len(expect)) if expect else 1.0,
                "precision": (len(hit) / len(top)) if top else 0.0,
                "mrr": (1.0 / rank) if rank else 0.0,
                "demoted_ok": not still_top,
                "still_top": still_top,
                "passed": (bool(hit) if expect else True) and not still_top,
            }
        )
    total = len(rows)
    return {
        "total": total,
        "recall": (sum(r["recall"] for r in rows) / total) if total else 0.0,
        "precision": (sum(r["precision"] for r in rows) / total) if total else 0.0,
        "mrr": (sum(r["mrr"] for r in rows) / total) if total else 0.0,
        "pass_rate": (sum(1 for r in rows if r["passed"]) / total) if total else 0.0,
        "rows": rows,
    }


# ——— 3. 排序健康度 ———


def run_ranking_health(data: dict, *, top_k: int) -> dict:
    """召回结果里一次性内容占比：衡量噪声污染召回的程度。"""
    from ai_agent.memory.long_term import is_transient_text
    from ai_agent.memory.progressive import rank_score, relevance
    from ai_agent.storage.db import MemoryItem

    items = [
        MemoryItem(
            id=str(raw.get("id")),
            user_id="eval",
            text=str(raw.get("text") or ""),
            created_at=str(raw.get("created_at") or "2026-11-14T00:00:00+00:00"),
            facet=str(raw.get("facet") or "other"),
        )
        for raw in data.get("items") or []
    ]
    transient_total = sum(1 for item in items if is_transient_text(item.text))
    returned = 0
    returned_transient = 0
    for query in data.get("queries") or []:
        text = str(query.get("query") or "")
        scored = [(item, rank_score(item, relevance(text, item))) for item in items]
        top = [item for item, _ in sorted(scored, key=lambda pair: -pair[1])][:top_k]
        returned += len(top)
        returned_transient += sum(1 for item in top if is_transient_text(item.text))
    return {
        "library_size": len(items),
        "library_transient": transient_total,
        "returned": returned,
        "returned_transient": returned_transient,
        "transient_rate": (returned_transient / returned) if returned else 0.0,
    }


# ——— 报告 ———


def render(report: dict) -> str:
    lines = ["# 记忆健康度", ""]
    noise = report.get("noise")
    if noise:
        lines.append("## 噪声拦截")
        lines.append(
            f"用例 {noise['total']} 条，规则层拦截 {noise['blocked_by_rule']} 条"
            f"（{noise['rule_rate']:.0%}）"
        )
        if noise["leaks"]:
            lines.append(f"规则未覆盖：{', '.join(noise['leaks'])}")
        lines.append("")
    recall = report.get("recall")
    if recall:
        lines.append("## 召回质量")
        lines.append(
            f"查询 {recall['total']} 条  Recall {recall['recall']:.0%}"
            f"  Precision {recall['precision']:.0%}  MRR {recall['mrr']:.2f}"
        )
        for row in recall["rows"]:
            if row["passed"]:
                continue
            if row.get("still_top"):
                lines.append(
                    f"- {row['id']}  一次性内容未被降权  query={row['query']}"
                    f"  仍在 top={row['still_top']}"
                )
            else:
                lines.append(
                    f"- {row['id']}  未命中  query={row['query']}"
                    f"  期望={row['expect']}  实际 top={row['top']}"
                )
        lines.append("")
    health = report.get("ranking")
    if health:
        lines.append("## 排序健康度")
        lines.append(
            f"库内 {health['library_size']} 条（一次性 {health['library_transient']} 条），"
            f"召回 {health['returned']} 条中一次性 {health['returned_transient']} 条"
            f"（{health['transient_rate']:.0%}）"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="记忆健康度报告")
    parser.add_argument("--noise-cases", help="噪声用例集路径")
    parser.add_argument("--recall-cases", help="召回用例集路径")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--noise-only", action="store_true")
    parser.add_argument("--json", help="把报告写成 JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="噪声拦截率与召回通过率的下限，默认 0.8",
    )
    args = parser.parse_args(argv)

    report: dict = {}
    noise_path = Path(args.noise_cases) if args.noise_cases else NOISE_CASES_PATH
    if noise_path.is_file():
        try:
            report["noise"] = run_noise(load_json(noise_path))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"噪声用例读取失败：{exc}", file=sys.stderr)
            return 2

    if not args.noise_only:
        recall_path = (
            Path(args.recall_cases) if args.recall_cases else RECALL_CASES_PATH
        )
        if recall_path.is_file():
            try:
                data = load_json(recall_path)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"召回用例读取失败：{exc}", file=sys.stderr)
                return 2
            if isinstance(data, dict):
                report["recall"] = run_recall(data, top_k=args.top_k)
                report["ranking"] = run_ranking_health(data, top_k=args.top_k)

    print(render(report))
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    # 退出码与 eval_run 的 exit_map 对齐：0 通过 / 1 低于阈值 / 2 用例不可用
    failures: list[str] = []
    noise = report.get("noise")
    if noise and noise["rule_rate"] < args.threshold:
        failures.append(f"噪声拦截率 {noise['rule_rate']:.0%} < {args.threshold:.0%}")
    recall = report.get("recall")
    if recall and recall["pass_rate"] < args.threshold:
        failures.append(
            f"召回通过率 {recall['pass_rate']:.0%} < {args.threshold:.0%}"
        )
    if failures:
        print("未达标：" + "；".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
