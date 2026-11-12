"""跨页表格补表题：表题留在上一页正文段里时，数据片也要带上它。"""

from ai_agent.kb.ingest.chunking import _carry_md_table_captions
from ai_agent.kb.ingest.parsing import TABLE, TEXT, Segment

TABLE_BODY = "| 项目 | 2026年9月30日 | 2025年12月31日 |\n|---|---|---|\n| 货币资金 | 2,696,401 | 770,790 |"


def test_caption_from_previous_text_segment_is_carried():
    segs = [
        Segment("公司负责人：倪真\n母公司资产负债表\n2026年9月30日", page=13, kind=TEXT, section="四、季度财务报表"),
        Segment(TABLE_BODY, page=14, kind=TABLE, section="四、季度财务报表"),
    ]
    out = _carry_md_table_captions(segs)
    assert out[1].text.startswith("母公司资产负债表\n")
    assert "货币资金" in out[1].text


def test_caption_reset_when_section_changes():
    segs = [
        Segment("母公司资产负债表", page=13, kind=TEXT, section="A"),
        Segment(TABLE_BODY, page=14, kind=TABLE, section="B"),
    ]
    out = _carry_md_table_captions(segs)
    assert not out[1].text.startswith("母公司资产负债表")


def test_own_caption_not_duplicated():
    segs = [
        Segment("合并现金流量表\n" + TABLE_BODY, page=14, kind=TABLE, section="A"),
    ]
    out = _carry_md_table_captions(segs)
    assert out[0].text.count("合并现金流量表") == 1


def test_unit_line_follows_caption():
    segs = [
        Segment("母公司资产负债表\n单位：千元", page=13, kind=TEXT, section="A"),
        Segment(TABLE_BODY, page=14, kind=TABLE, section="A"),
    ]
    out = _carry_md_table_captions(segs)
    assert out[1].text.startswith("母公司资产负债表\n单位：千元\n")
