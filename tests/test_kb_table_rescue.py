"""表格页 OCR 兜底的判定逻辑（不真的调模型）。"""

from pathlib import Path

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


def test_table_blocks_keeps_only_html_tables():
    """整页 OCR 会把正文也吐出来，只保留 <table> 块，否则正文重复一遍。"""
    from ai_agent.kb.ingest.table_rescue import _table_blocks

    text = (
        "## 第二节 公司简介\n\n"
        "<table><tr><td>股票简称</td><td>汇川技术</td></tr></table>\n\n"
        "1、公司联系方式\n公司注册地址无变化。\n\n"
        "<table><tr><td>姓名</td><td>宋君恩</td></tr></table>"
    )
    blocks = _table_blocks(text)
    assert len(blocks) == 2
    assert all(block.startswith("<table") for block in blocks)
    assert "公司注册地址" not in "".join(blocks)


def test_rescue_drops_every_table_on_the_page(monkeypatch):
    """OCR 覆盖整页：页内「表头正常」的旧表也要让位，否则同一张表出现两次。"""
    from ai_agent.kb.ingest import ocr
    from ai_agent.kb.ingest import table_rescue as rescue

    segments = [
        Segment("| 项目 | 列2 |\n|---|---|\n| a | 1 |", 7, TABLE),
        # 这张表头是好的，但它和上面那张在 OCR 里会被一起读出来
        Segment("| 姓名 | 宋君恩 |", 7, TABLE),
        Segment("正文段落", 8, TEXT),
    ]
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(
        ocr,
        "ocr_pdf",
        lambda *_a, **_k: [
            Segment(
                "正文段落\n<table><tr><td>股票简称</td><td>汇川技术</td></tr></table>",
                7,
                TEXT,
            )
        ],
    )
    monkeypatch.setattr(rescue, "_page_digit_stream", lambda *_a, **_k: "")
    monkeypatch.setattr(rescue, "numbers_verified", lambda *_a, **_k: True)

    out, stats = rescue.rescue_table_pages(Path("x.pdf"), segments)
    page7 = [s for s in out if s.page == 7]
    assert stats["OCR补回"] == 1
    assert len(page7) == 1                     # 旧的两张表都被摘掉
    assert "股票简称" in page7[0].text
    assert "正文段落" not in page7[0].text       # 正文不重复
    assert any(s.page == 8 for s in out)        # 别的页不受影响


def test_collapsed_cells_detects_crammed_numbers():
    """中国银行利润表实录：8 个数字挤在一格 → 列被压扁。"""
    from ai_agent.kb.ingest.table_rescue import collapsed_cells

    text = (
        "| 项目 | 本期 |\n|---|---|\n"
        "| 一、营业收入 | 162,201 491,204 133,776 391,889 169,462 526,365 |"
    )
    assert collapsed_cells(text) == 1


def test_collapsed_cells_ignores_dates():
    """2026年9月30日 里也有三个数字，但不能当成压扁的表。"""
    from ai_agent.kb.ingest.table_rescue import collapsed_cells

    text = "| 项目 | 日期 |\n|---|---|\n| 报表日 | 2026年9月30日 |"
    assert collapsed_cells(text) == 0


def test_collapsed_cells_works_on_html_tables():
    from ai_agent.kb.ingest.table_rescue import collapsed_cells

    html = "<table><tr><td>营业收入</td><td>1 2 3 4</td></tr></table>"
    assert collapsed_cells(html) == 1


def test_repeated_header_detects_split_year_column():
    """`年年年年年年年年2025` 是 find_tables 按字切表头的产物。"""
    from ai_agent.kb.ingest.table_rescue import has_repeated_header

    assert has_repeated_header("| 年年年年年年年年2025 | 2026 |\n|---|---|\n| a | 1 |")
    assert not has_repeated_header("| 项目 | 2026年 |\n|---|---|\n| a | 1 |")


def test_column_limit_trusts_ocr_on_collapsed_tables():
    """列被压扁时文字层列数不可信：2 列塞了 8 个数，OCR 读成 9 列是对的。"""
    from ai_agent.kb.ingest.table_rescue import _COLUMN_CAP, _column_limit

    collapsed = "| 项目 | 本期 |\n|---|---|\n| 营业收入 | 162,201 491,204 133,776 391,889 |"
    assert _column_limit(collapsed) == _COLUMN_CAP


def test_column_limit_allows_multi_period_reports():
    """所有者权益变动表：文字层 6 列、OCR 13 列（12 个期间列），要放行。"""
    from ai_agent.kb.ingest.table_rescue import _column_limit

    old = (
        "| 项目 | 本期 | 上期 | 年初 | 增减 | 备注 |\n"
        "|---|---|---|---|---|---|\n"
        "| a | 1 | 2 | 3 | 4 | 5 |\n| b | 6 | 7 | 8 | 9 | 10 |"
    )
    assert _column_limit(old) >= 13


def test_column_limit_still_catches_shredded_output():
    from ai_agent.kb.ingest.table_rescue import _COLUMN_CAP, _column_limit

    normal = "| 项目 | 本期 |\n|---|---|\n| a | 1 |"
    assert _column_limit(normal) <= _COLUMN_CAP
