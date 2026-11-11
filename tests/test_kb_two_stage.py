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


def test_list_only_flag_declared_off_by_default():
    """声明默认关（.env 打开后不影响这条断言）。"""
    from ai_agent.core.config import Settings

    assert Settings.model_fields["kb_search_list_only"].default is False


def test_cache_survives_thread_boundary():
    """工具跑在线程池里：kb_search 与 kb_read 是两次独立调用。

    早先用 ContextVar 存命中，工作线程之间不共享，kb_read 永远读不到——
    线上表现为模型反复「搜索→读取失败」，直到步数预算耗尽。
    """
    import threading

    hits = [_hit(1, 1, "| 负债合计 | 100 |")]
    ks.set_last_hits(hits, search_id="abc123")
    got: list[int] = []

    def _other_thread() -> None:
        got.append(len(ks._read_hits([1], "abc123")))

    worker = threading.Thread(target=_other_thread)
    worker.start()
    worker.join()
    assert got == [1] or got[0] > 0, "跨线程读不到缓存"
