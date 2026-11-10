"""切片体检：能抓出静默错数与垃圾片，又不会把正常表误判成有问题。"""

from ai_agent.kb.ingest.selfcheck import (
    check_balance_identity,
    check_chunks,
)


def _table(*rows: str) -> str:
    return "\n".join(
        ["| 项目 | 2026年9月30日 | 2025年12月31日 |", "|---|---|---|", *rows]
    )


def test_split_number_in_cell_is_flagged():
    """真值 403122922 被拆成「40312292 2」：金额差 10 倍，必须报出来。"""
    issues = check_chunks([_table("| 磐石润企 | 40312292 2 | 0 |")])
    assert issues.split_numbers == ["40312292 2"]
    assert not issues.ok


def test_normal_amounts_are_not_flagged():
    issues = check_chunks([_table("| 货币资金 | 5,060,456,799.16 | 5,574,955,555.50 |")])
    assert issues.split_numbers == []
    assert issues.ok


def test_page_mark_chunk_and_placeholder_header_are_flagged():
    issues = check_chunks(
        [
            "6 / 12",
            "| 项目 | 列2 | 列3 |\n|---|---|---|\n| 甲公司 | 1 | 2 |",
        ]
    )
    assert issues.page_marks == 1
    assert issues.placeholder_tables == 1


def test_balance_identity_across_split_chunks():
    """资产负债表被切成两片，两行总额各在一处，仍要能对上。"""
    assets = _table("| 资产总计 | 62,249,241,511.39 | 55,516,503,601.44 |")
    liabilities = _table("| 负债和所有者权益总计 | 62,249,241,511.39 | 55,516,503,601.44 |")
    assert check_balance_identity([assets, liabilities]) == []


def test_balance_identity_mismatch_is_reported():
    assets = _table("| 资产总计 | 62,249,241,511.39 | 55,516,503,601.44 |")
    liabilities = _table("| 负债和所有者权益总计 | 6,224,924,151.39 | 55,516,503,601.44 |")
    problems = check_balance_identity([assets, liabilities])
    assert len(problems) == 1
    assert "资产总计" in problems[0]


def test_balance_identity_skips_when_ambiguous():
    """合并表与母公司表都有总计行 → 凑不齐对应关系，宁可不判。"""
    assets = _table("| 资产总计 | 100 | 90 |")
    liabilities = _table("| 负债和所有者权益总计 | 100 | 90 |")
    parent_assets = _table("| 资产总计 | 50 | 45 |")
    assert check_balance_identity([assets, liabilities, parent_assets]) == []


def test_balance_identity_ignores_empty_cells():
    assets = _table("| 资产总计 | 100 | - |")
    liabilities = _table("| 负债和所有者权益总计 | 100 | - |")
    assert check_balance_identity([assets, liabilities]) == []
