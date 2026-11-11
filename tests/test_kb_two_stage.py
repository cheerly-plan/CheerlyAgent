"""两段式检索（影子模式）：清单不打码不漏数、kb_read 能取回全文。"""

from ai_agent.core.config import settings
from ai_agent.kb.retrieval.retrieval import Hit
from ai_agent.tools.kb import kb_search as ks


def _hit(seq: int, page: int, text: str) -> Hit:
    return Hit(
        file_id="f1",
        filename="某公司2026年第三季度报告",
        seq=seq,
        page=page,
        kind="table",
        text=text,
        distance=0.2,
        section="四、季度财务报表",
        period="2026Q3",
    )


def test_preview_masks_numbers():
    """清单预览不能漏出金额，否则模型会拿预览直接作答。"""
    text = "| 货币资金 | 5,060,456,799.16 | 5,574,955,555.50 |"
    preview = ks._preview(text)
    assert "5,060,456,799" not in preview
    assert "5,574,955,555" not in preview
    assert "…" in preview


def test_list_format_has_index_and_location():
    hits = [_hit(39, 10, "| 五、净利润 | 1,589,422,902.20 |"), _hit(40, 11, "合并利润表")]
    ks.set_last_hits(hits)
    lines = ks._format_hit_list(hits)
    assert lines.startswith("[1] ")
    assert "第10页" in lines
    assert "1,589,422,902.20" not in lines   # 金额必须打码


def test_read_hits_returns_full_text():
    hits = [_hit(39, 10, "| 五、净利润 | 1,589,422,902.20 |"), _hit(40, 11, "合并利润表")]
    ks.set_last_hits(hits)
    detail = ks._read_hits([1])
    assert "1,589,422,902.20" in detail


def test_read_hits_without_search_hints():
    ks.set_last_hits([])
    assert "先调用 kb_search" in ks._read_hits([1])


def test_read_hits_out_of_range():
    ks.set_last_hits([_hit(1, 1, "a")])
    assert "有效范围 1~1" in ks._read_hits([9])


def test_list_only_flag_off_by_default():
    assert settings.kb_search_list_only is False
