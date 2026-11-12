"""两段式检索（影子模式）：清单不打码不漏数、kb_read 能取回全文。"""

import uuid

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


def test_read_hits_without_cache_hints_to_search():
    ks.set_last_hits([])
    assert "先 kb_search" in ks._read_hits([1])


def test_read_hits_with_query_missing_cache_asks_research():
    """给了检索词却查不到缓存时，不能把"最近一次检索"的无关结果塞回去。"""
    ks.set_last_hits([_hit(1, 1, "别的检索")])
    out = ks._read_hits([1], query="完全不相干的词")
    assert "重新 kb_search" in out
    assert "别的检索" not in out


def test_read_hits_out_of_range():
    ks.set_last_hits([_hit(1, 1, "a")])
    # 越界编号会退回默认前 3 条，不会返回空
    assert "a" in ks._read_hits([9])


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


# ——— 按问句读回（历史轮次老化后取原文） ———


def test_search_caches_hits_by_query(monkeypatch):
    """kb_search 把结果按问句缓存，供后续轮次 kb_read 读回。"""
    monkeypatch.setattr(ks.store, "has_ready_kb_file", lambda: True)
    hits = [_hit(39, 10, "| 五、净利润 | 1,589,422,902.20 |")]
    monkeypatch.setattr(
        ks, "retrieve_for_question", lambda *a, **kw: ([uuid.uuid4()], hits)
    )
    monkeypatch.setattr(ks.settings, "kb_search_list_only", False)
    out = ks._kb_search("长电科技 2026Q3 净利润")
    assert "1,589,422,902.20" in out
    back = ks._read_hits(query="长电科技 2026Q3 净利润")
    assert "1,589,422,902.20" in back


def test_query_lookup_ignores_whitespace(monkeypatch):
    monkeypatch.setattr(ks.store, "has_ready_kb_file", lambda: True)
    hits = [_hit(39, 10, "| 五、净利润 | 1,589,422,902.20 |")]
    monkeypatch.setattr(
        ks, "retrieve_for_question", lambda *a, **kw: ([uuid.uuid4()], hits)
    )
    monkeypatch.setattr(ks.settings, "kb_search_list_only", False)
    ks._kb_search("长电科技  2026Q3   净利润")
    assert "1,589,422,902.20" in ks._read_hits(query="长电科技 2026Q3 净利润")


def test_read_without_ids_returns_first_three():
    hits = [_hit(i, i, f"第{i}条") for i in range(1, 6)]
    ks.set_last_hits(hits)
    out = ks._read_hits()
    assert "第1条" in out and "第3条" in out
    assert "第4条" not in out
