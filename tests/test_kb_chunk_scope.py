"""切片补 scope：表题写了母公司/合并，就把它标成对应报表口径。"""

from ai_agent.kb.ingest.chunking import _scope_from_caption, chunk_segments
from ai_agent.kb.ingest.parsing import TABLE, Segment

BODY = "| 项目 | 2026年9月30日 | 2025年12月31日 |\n|---|---|---|\n| 货币资金 | 2,696,401 | 770,790 |"


def test_scope_from_caption_reads_first_line_only():
    assert _scope_from_caption("母公司资产负债表\n| 项目 | 1 |") == "parent"
    assert _scope_from_caption("合并利润表\n| 项目 | 1 |") == "consolidated"


def test_subject_row_does_not_make_it_parent():
    """「归属于母公司所有者的净利润」是科目名，不能把合并报表标成母公司。"""
    text = "合并利润表\n| 归属于母公司所有者的净利润 | 63,888 |"
    assert _scope_from_caption(text) == "consolidated"


def test_table_without_caption_has_no_scope():
    assert _scope_from_caption(BODY) is None


def test_chunk_gets_scope_from_its_caption():
    segs = [
        Segment("母公司资产负债表\n" + BODY, page=14, kind=TABLE, section="四、季度财务报表"),
    ]
    chunks = chunk_segments(segs, max_chars=500, table_group_mode="section")
    assert chunks and all(c.scope == "parent" for c in chunks)


def test_scope_set_by_parser_is_kept():
    segs = [
        Segment("合并资产负债表\n" + BODY, page=7, kind=TABLE, section="四、季度财务报表", scope="consolidated"),
    ]
    chunks = chunk_segments(segs, max_chars=500, table_group_mode="section")
    assert all(c.scope == "consolidated" for c in chunks)
