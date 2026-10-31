"""图谱对齐质量检查：找出历史对齐留下的错误合并。

背景：早期 align 允许 metric（财务科目）走「向量近邻 → LLM 裁决」这条路，
而「经营活动现金流入小计」和「流出小计」的向量相似度高达 0.99、「应付职工
薪酬」和「长期应付职工薪酬」同理，被成批判成同一实体。修好规则只能防住
将来，已经合进去的错并要单独清理——

错并的后果比漏并严重：`search_entities` 的第三路召回是别名 LIKE，
「流出」会命中「流入」那个实体，模型据此给出的财务数据方向是反的。

清理只做一件事：删掉「按新规则本不该存在」的别名记录。
被合并掉的实体在 `apply_alignment` 里已经删除，删别名不会把它变回来——
要恢复得重建图谱（POST /v1/kg/graph/build）。

用法：
    uv run python scripts/check_kg_alignment.py           # 只报告，不改数据
    uv run python scripts/check_kg_alignment.py --apply   # 删除判定为错的别名
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_agent.kb import align  # noqa: E402
from ai_agent.memory.store import store  # noqa: E402


def load_aliases() -> list[dict]:
    with store._get_pool().connection() as conn:
        return conn.execute(
            """
            SELECT a.id, a.alias_name, a.method, a.score,
                   e.id AS entity_id, e.name AS entity_name, e.type, e.mention_count,
                   f.filename
            FROM agent_kg_alias a
            JOIN agent_kg_entity e ON e.id = a.entity_id
            JOIN agent_kb_file f ON f.id = a.file_id
            ORDER BY f.filename, e.name
            """
        ).fetchall()


def judge(row: dict) -> tuple[bool, str]:
    """返回（是否该删, 理由）。判据与 align.candidate_pairs 保持一致。"""
    entity = {
        "name": row["entity_name"],
        "type": row["type"],
    }
    if align.is_rigid(entity):
        if row["type"] in align._RIGID_TYPES:
            return True, "metric 类不该参与对齐"
        return True, "泛指词/数值名不该参与对齐"
    # 限定词 + 短名：长期↔短期、期初↔期末、流动↔非流动，语义已经变了
    if align.has_qualifier_prefix(row["alias_name"], row["entity_name"]):
        return True, "限定词改变了语义"
    if align.has_qualifier_prefix(row["entity_name"], row["alias_name"]):
        return True, "限定词改变了语义"
    # 数字对不上：4.0↔5.0、第1期↔第2期，是两代/两期不同的东西
    if align.number_conflict(row["alias_name"], row["entity_name"]):
        return True, "数字对不上（不同版本/期数）"
    return False, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="检查图谱对齐的错误合并")
    parser.add_argument(
        "--apply", action="store_true", help="删除判定为错的别名（默认只报告）"
    )
    args = parser.parse_args()

    rows = load_aliases()
    if not rows:
        print("库里没有别名记录")
        return 0

    wrong: list[tuple[dict, str]] = []
    for row in rows:
        drop, why = judge(row)
        if drop:
            wrong.append((row, why))

    print(f"别名总数 {len(rows)}｜判定为错误合并 {len(wrong)}")
    by_type = Counter(row["type"] for row, _ in wrong)
    if by_type:
        print("错误项按类型：" + "、".join(f"{k} {v}" for k, v in by_type.most_common()))
    print()

    current_file = None
    for row, why in wrong:
        if row["filename"] != current_file:
            current_file = row["filename"]
            print(f"《{current_file}》")
        print(f"  {row['alias_name']} → {row['entity_name']}   （{why}）")

    if not wrong:
        print("没有发现错误合并")
        return 0

    print()
    print(
        "注意：别名删掉不会让被合并的实体回来——它在 apply_alignment 里已经删除，"
        "关系也已改指到代表实体。要让图谱真正干净，需要重建该文件的图谱。"
    )

    if not args.apply:
        print("\n这是 dry-run。确认后加 --apply 执行删除。")
        return 0

    with store._get_pool().connection() as conn:
        for row, _ in wrong:
            conn.execute("DELETE FROM agent_kg_alias WHERE id = %s", (row["id"],))
        conn.commit()
    print(f"\n已删除 {len(wrong)} 条错误别名")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
