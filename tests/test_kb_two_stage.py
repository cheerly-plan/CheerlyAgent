"""两段式检索：清单带引用、kb_read 按引用回库里读原文。

缓存正文的旧实现已经去掉——切片引用（``file 前 8 位:seq``）本身就是稳定标识，
``kb_read`` 拿它现读库，所以进程重启、多实例部署、知识库重跑都不会读到旧内容。
"""

import uuid
from unittest.mock import patch

from ai_agent.core.config import settings
from ai_agent.kb.retrieval.retrieval import Hit
from ai_agent.tools.kb import kb_search as ks

FILE_ID = "3823ff6e-1111-2222-3333-444455556666"


def _hit(seq: int, page: int, text: str, file_id: str = FILE_ID) -> Hit:
    return Hit(
        file_id=file_id,
        filename="某公司2026年第三季度报告",
        seq=seq,
        page=page,
        kind="table",
        text=text,
        distance=0.2,
        section="四、季度财务报表",
        period="2026Q3",
    )


def _row(seq: int, text: str) -> dict:
    return {
        "file_id": FILE_ID,
        "seq": seq,
        "page": 10,
        "page_end": 10,
        "kind": "table",
        "section": "四、季度财务报表",
        "scope": None,
        "period": "2026Q3",
        "text": text,
        "filename": "某公司2026年第三季度报告",
    }


def test_preview_masks_numbers():
    """清单预览不能漏出金额，否则模型会拿预览直接作答。"""
    preview = ks._preview("| 货币资金 | 5,060,456,799.16 | 5,574,955,555.50 |")
    assert "5,060,456,799" not in preview
    assert "5,574,955,555" not in preview
    assert "…" in preview


def test_list_format_has_index_location_and_ref():
    hits = [_hit(39, 10, "| 五、净利润 | 1,589,422,902.20 |")]
    lines = ks._format_hit_list(hits)
    assert lines.startswith("[1] ")
    assert "第10页" in lines
    assert "3823ff6e:39" in lines          # 抬头里带引用，kb_read 靠它读回
    assert "1,589,422,902.20" not in lines  # 金额仍然打码


def test_full_mode_header_has_no_ref():
    """正文里不能出现 ref —— 模型会把它抄进回答（实测泄漏）。"""
    hit = _hit(39, 10, "| 五、净利润 | 1,589,422,902.20 |")
    out = ks._format_hit_with_context(hit)
    assert "ref=" not in out


def test_refs_are_passed_out_of_band(monkeypatch):
    """引用走旁路（ContextVar），由 loop 记到工具消息的 refs 字段。"""
    monkeypatch.setattr(ks.store, "has_ready_kb_file", lambda: True)
    hits = [_hit(39, 10, "| 五、净利润 | 1,589,422,902.20 |")]
    monkeypatch.setattr(
        ks, "retrieve_for_question", lambda *a, **kw: ([uuid.uuid4()], hits)
    )
    monkeypatch.setattr(ks.settings, "kb_search_list_only", False)
    out = ks._kb_search("长电科技 2026Q3 净利润")
    assert "ref=" not in out
    assert ks.take_refs() == ["3823ff6e:39"]


def test_extract_refs_from_tool_output():
    text = "〔ref=3823ff6e:35〕…〔ref=61a849b4:179〕…〔ref=3823ff6e:35〕"
    assert ks.extract_refs(text) == ["3823ff6e:35", "61a849b4:179"]


def test_read_hits_reads_current_text_from_db():
    with patch.object(
        ks.store, "fetch_kb_chunks_by_prefix",
        return_value={("3823ff6e", 39): _row(39, "| 五、净利润 | 1,589,422,902.20 |")},
    ):
        out = ks._read_hits(refs=["3823ff6e:39"])
    assert "1,589,422,902.20" in out


def test_read_hits_without_refs_asks_for_ref():
    assert "没有可读取的引用" in ks._read_hits(refs=[])


def test_read_hits_unknown_ref_asks_research():
    with patch.object(ks.store, "fetch_kb_chunks_by_prefix", return_value={}):
        out = ks._read_hits(refs=["deadbeef:1"])
    assert "重新 kb_search" in out


def test_read_hits_ignores_malformed_refs():
    with patch.object(ks.store, "fetch_kb_chunks_by_prefix", return_value={}) as spy:
        out = ks._read_hits(refs=["不是引用", "3823ff6e"])
    assert "没有可读取的引用" in out
    spy.assert_not_called()


def test_read_hits_is_stateless_across_processes():
    """没有进程内缓存：同样的引用两次都回库里读，结果一致。"""
    payload = {("3823ff6e", 39): _row(39, "| 净利润 | 100 |")}
    with patch.object(ks.store, "fetch_kb_chunks_by_prefix", return_value=payload) as spy:
        first = ks._read_hits(refs=["3823ff6e:39"])
        second = ks._read_hits(refs=["3823ff6e:39"])
    assert first == second
    assert spy.call_count == 2


def test_list_only_flag_declared_off_by_default():
    from ai_agent.core.config import Settings

    assert Settings.model_fields["kb_search_list_only"].default is False


def test_ref_round_trip_through_search(monkeypatch):
    """kb_search 输出里的引用能直接喂回 kb_read。"""
    monkeypatch.setattr(ks.store, "has_ready_kb_file", lambda: True)
    hits = [_hit(39, 10, "| 五、净利润 | 1,589,422,902.20 |")]
    monkeypatch.setattr(
        ks, "retrieve_for_question", lambda *a, **kw: ([uuid.uuid4()], hits)
    )
    monkeypatch.setattr(ks.settings, "kb_search_list_only", False)
    out = ks._kb_search("长电科技 2026Q3 净利润")
    refs = ks.take_refs()
    assert refs == ["3823ff6e:39"]
    with patch.object(
        ks.store, "fetch_kb_chunks_by_prefix",
        return_value={("3823ff6e", 39): _row(39, "| 五、净利润 | 1,589,422,902.20 |")},
    ):
        assert "1,589,422,902.20" in ks._kb_read(refs=refs)
