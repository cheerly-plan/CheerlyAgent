"""表格页 OCR 兜底的判定逻辑（不真的调模型）。"""

from ai_agent.kb.ingest.parsing import TABLE, TEXT, Segment
from ai_agent.kb.ingest.table_rescue import (
    _candidates,
    _column_count,
    numbers_verified,
)


def test_numbers_verified_accepts_fragmented_text_layer():
    """文字层把 819,871,170.19 切成 819,8 / 71,17 / 0.19，拼起来应当对得上。"""
    # 文字层原样是「819,8 / 71,17 / 0.19」，按阅读顺序拼起来就是这串
    stream = "81987117019"
    assert numbers_verified("<td>819,871,170.19</td>", stream)


def test_numbers_verified_rejects_hallucinated_amount():
    """OCR 读出一个原文没有的金额 → 整页弃用。"""
    assert not numbers_verified("<td>123,456,789.00</td>", "81987117019")


def test_numbers_verified_ignores_short_numbers():
    """0、5.00 这种短数字遍地都是，没有区分度，不参与校验。"""
    assert numbers_verified("<td>0</td><td>5.00</td>", "")


def test_candidates_only_picks_broken_tables():
    segs = [
        Segment("| 项目 | 列2 | 列3 |", 3, TABLE),
        Segment("| 项目 | 本期 | 上期 |", 4, TABLE),
        Segment("正文", 5, TEXT),
    ]
    assert _candidates(segs, set()) == {3: [0]}


def test_candidates_includes_crashed_pages():
    segs = [Segment("| 项目 | 本期 | 上期 |", 9, TABLE)]
    assert _candidates(segs, {9}) == {9: [0]}


def test_column_count_reads_html_and_markdown():
    assert _column_count("<table><tr><td>a</td><td>b</td><td>c</td></tr></table>") == 3
    assert _column_count("| a | b |\n|---|---|\n| 1 | 2 |") == 2
