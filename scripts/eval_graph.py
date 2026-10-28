"""图谱质量评测：金标实体/别名/关系 → 召回与连通率。

为什么要有这个脚本：图谱是"慢工出细活"的东西，改一次 prompt、换一次
切块策略、加一次实体对齐，效果都是"看图凭感觉"。这个脚本把感觉换成数字，
让每次改动都有前后对照。

指标口径（详见 kb/graph_metrics.py）：
- entity_recall   金标实体在图上的命中率（含别名命中）
- alias_recall    释义表里的简称-全称对，最终是否落成同一个点（P2 的效果）
- relation_recall 金标关联是否在图上 2 跳内连通（不苛求边名完全一致）
- isolated_ratio  孤立实体占比
- largest_component_ratio 最大连通分量占比
- relations_per_entity    关系密度

用法：
    uv run python scripts/eval_graph.py                        # 按 gold 文件里的文件名找库内文件
    uv run python scripts/eval_graph.py --file-id <uuid>
    uv run python scripts/eval_graph.py --gold data/eval/gold_guodian.json --out data/eval/report.json
    uv run python scripts/eval_graph.py --before data/eval/report_before.json   # 与上一次对比

退出码：0 正常；1 找不到文件或图谱为空。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uuid  # noqa: E402

from ai_agent.kb.graph_metrics import evaluate  # noqa: E402
from ai_agent.memory.store import store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / "data" / "eval" / "gold_guodian.json"


def find_file(filename: str) -> dict | None:
    rows = store.list_kb_files()
    for row in rows:
        if row["filename"] == filename:
            return row
    for row in rows:
        if filename and filename in row["filename"]:
            return row
    return None


def load_graph(file_id: uuid.UUID) -> tuple[list[dict], list[dict], list[dict]]:
    """库层返回的是 source_id/target_id，指标层统一用 source/target。"""
    graph = store.get_file_graph(file_id, limit=100000)
    nodes = list(graph["nodes"])
    edges = [
        {**dict(e), "source": str(e["source_id"]), "target": str(e["target_id"])}
        for e in graph["edges"]
    ]
    aliases = store.list_aliases(file_id)
    return nodes, edges, aliases


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(report: dict, gold: dict, filename: str) -> str:
    lines = [
        f"# 图谱质量评测 · {filename}",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 实体数 | {report['entities']} |",
        f"| 关系数 | {report['relations']} |",
        f"| 关系密度（边/点） | {fmt(report['relations_per_entity'], 2)} |",
        f"| 孤立实体占比 | {fmt(report['isolated_ratio'])} |",
        f"| 最大连通分量占比 | {fmt(report['largest_component_ratio'])} |",
        f"| 金标实体召回（{len(gold.get('entities') or [])}） | {fmt(report.get('entity_recall'))} |",
        f"| 别名合并率（{len(gold.get('aliases') or [])}） | {fmt(report.get('alias_recall'))} |",
        f"| 金标关系连通率（{len(gold.get('relations') or [])}） | {fmt(report.get('relation_recall'))} |",
    ]
    for key, title in (
        ("entity_misses", "金标实体未命中"),
        ("alias_unmerged", "别名未合并"),
        ("alias_missing", "别名端点缺失"),
        ("relation_unlinked", "关系未连通"),
        ("relation_absent", "关系端点缺失"),
    ):
        items = report.get(key) or []
        if items:
            lines += ["", f"## {title}（{len(items)}）", ""]
            lines += [f"- {' ↔ '.join(i) if isinstance(i, list) else i}" for i in items]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="图谱质量评测")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD), help="金标文件路径")
    parser.add_argument("--file-id", default="", help="直接指定文件 id")
    parser.add_argument("--out", default="", help="报告输出路径（.json）")
    parser.add_argument("--before", default="", help="上一次的报告 json，用于对比")
    parser.add_argument("--print", action="store_true", help="只打印，不落盘")
    args = parser.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))

    if args.file_id:
        file_id = uuid.UUID(args.file_id)
        record = store.get_kb_file(file_id) or {"filename": args.file_id}
    else:
        record = find_file(gold.get("file", ""))
        if record is None:
            print(f"库里找不到文件：{gold.get('file')}")
            return 1
        file_id = record.get("id") or record["file_id"]

    nodes, edges, aliases = load_graph(file_id)
    if not nodes:
        print(f"《{record['filename']}》还没有图谱，先建图再评测")
        return 1

    report = evaluate(nodes, edges, gold, aliases)
    text = render(report, gold, record["filename"])
    print(text)

    if args.before:
        prev = json.loads(Path(args.before).read_text(encoding="utf-8"))
        keys = [
            "entities",
            "relations",
            "isolated_ratio",
            "largest_component_ratio",
            "entity_recall",
            "alias_recall",
            "relation_recall",
        ]
        print("\n## 与上次对比\n")
        print("| 指标 | 上次 | 本次 | 变化 |")
        print("|---|---|---|---|")
        for key in keys:
            old, new = prev.get(key), report.get(key)
            if old is None and new is None:
                continue
            delta = ""
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                delta = f"{new - old:+.3f}" if isinstance(new, float) else f"{new - old:+d}"
            print(f"| {key} | {fmt(old)} | {fmt(new)} | {delta} |")

    if not args.print:
        out = Path(args.out) if args.out else ROOT / "data" / "eval" / f"report_{file_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
