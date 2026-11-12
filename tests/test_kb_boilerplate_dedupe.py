"""页眉页脚剔除与切片去重。"""

from ai_agent.kb.ingest.chunking import (
    Chunk,
    _drop_duplicate_chunks,
    _drop_page_boilerplate,
)
from ai_agent.kb.ingest.parsing import TABLE, TEXT, Segment

FOOTER = "沈阳新松机器人自动化股份有限公司2026 年半年度报告全文"


def _pages(n: int, body: str = "正文内容") -> list[Segment]:
    return [Segment(f"{body}\n{FOOTER}", page=i, kind=TEXT, section="S") for i in range(1, n + 1)]


def test_repeated_footer_is_removed():
    out = _drop_page_boilerplate(_pages(5))
    assert all(FOOTER not in (s.text or "") for s in out)
    assert all("正文内容" in (s.text or "") for s in out)


def test_footer_only_segment_disappears():
    segs = [Segment(FOOTER, page=i, kind=TEXT, section="S") for i in range(1, 5)]
    assert _drop_page_boilerplate(segs) == []


def test_signature_lines_removed():
    segs = [
        Segment(f"公司负责人：郑力\n主管会计工作负责人：梁征\n正文{i}", page=i, kind=TEXT, section="S")
        for i in range(1, 5)
    ]
    out = _drop_page_boilerplate(segs)
    assert all("负责人" not in (s.text or "") for s in out)


def test_content_line_repeated_on_a_few_pages_is_kept():
    """「确定该组合依据的说明：」在 6 页出现，但它是正文，不能删。"""
    line = "确定该组合依据的说明："
    segs = [Segment(f"{line}\n正文{i}", page=i, kind=TEXT, section="S") for i in range(1, 5)]
    out = _drop_page_boilerplate(segs)
    assert all(line in (s.text or "") for s in out)


def test_footer_merged_with_content_is_kept():
    """页眉和正文拼成一行时不能整行删掉。"""
    segs = [
        Segment(f"{FOOTER}：☑（已勾选）适用", page=i, kind=TEXT, section="S")
        for i in range(1, 5)
    ]
    out = _drop_page_boilerplate(segs)
    assert len(out) == 4


def test_duplicate_chunks_keep_first_on_same_page():
    """同一页被切两遍（文字层 + OCR）才去重。"""
    chunks = [
        Chunk(text="甲", page=1, kind=TABLE, section="S"),
        Chunk(text=" 甲 ", page=1, kind=TABLE, section="S"),
        Chunk(text="乙", page=1, kind=TABLE, section="S"),
    ]
    out = _drop_duplicate_chunks(chunks)
    assert [c.text for c in out] == ["甲", "乙"]


def test_same_table_on_different_pages_is_kept():
    """正文与附录各印一次同样的表：两处出处都要留。"""
    chunks = [
        Chunk(text="甲", page=1, kind=TABLE, section="S"),
        Chunk(text="甲", page=2, kind=TABLE, section="S"),
    ]
    assert len(_drop_duplicate_chunks(chunks)) == 2
