"""检索质量四件套：语境前缀、去冗余、重排、邻接扩展。"""

import uuid

import pytest

from ai_agent.core.config import settings
from ai_agent.kb import retrieval
from ai_agent.kb.chunking import context_prefix, embedding_input
from ai_agent.tools.kb_search import _format_hit_with_context

FILE_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


def _row(seq: int, text: str, *, distance: float = 0.3) -> dict:
    return {
        "id": uuid.uuid4(),
        "file_id": FILE_ID,
        "filename": "半导体报告.pdf",
        "seq": seq,
        "page": seq + 1,
        "kind": "text",
        "text": text,
        "distance": distance,
    }


def test_context_prefix_carries_document_and_position():
    assert context_prefix("半导体报告.pdf", page=3, kind="text") == "《半导体报告》第3页"
    assert context_prefix("财务表.xlsx", page=2, kind="table") == "《财务表》表2"
    assert context_prefix("年报.pdf", page=5, kind="figure") == "《年报》第5页图表"
    # 无页概念的文档（Word）只带文档名，不硬编段号
    assert context_prefix("考勤制度.docx", page=None, kind="text") == "《考勤制度》"


def test_embedding_input_prefixes_context_but_keeps_body():
    text = "HBM 供需缺口扩大"
    assert embedding_input(text, filename="a.pdf", page=2, kind="text") == (
        f"《a》第2页\n{text}"
    )


def test_dedupe_drops_near_duplicate_chunks():
    """切片 overlap 会让相邻片高度重合，只保留信息量最大的那条。"""
    body = "2026 年 HBM 市场规模约 546 亿美元，同比增速 58%，产能供不应求。" * 3
    rows = [
        _row(0, body),
        _row(1, body + "略有改动"),  # 与上一条三字组重合度极高
        _row(2, "公司考勤制度规定，迟到超过 30 分钟按旷工处理。"),
    ]
    kept = retrieval._dedupe(rows)
    assert [row["seq"] for row in kept] == [0, 2]


def test_rerank_follows_model_order_and_keeps_unlisted(monkeypatch):
    rows = [_row(0, "甲"), _row(1, "乙"), _row(2, "丙")]
    monkeypatch.setattr(retrieval, "complete_chat_once", lambda prompt: "[3,1]")
    ordered = retrieval._rerank("问题", rows, top_k=2)
    assert [row["seq"] for row in ordered] == [2, 0, 1]


def test_rerank_falls_back_on_garbage_or_error(monkeypatch):
    rows = [_row(0, "甲"), _row(1, "乙")]
    monkeypatch.setattr(retrieval, "complete_chat_once", lambda prompt: "我觉得第2条更好")
    assert retrieval._rerank("问题", rows, top_k=2) == rows

    def boom(prompt):
        raise RuntimeError("模型超时")

    monkeypatch.setattr(retrieval, "complete_chat_once", boom)
    assert retrieval._rerank("问题", rows, top_k=2) == rows


def test_parse_order_ignores_out_of_range_and_duplicates():
    assert retrieval._parse_order("[2,2,9,0,-1,1]", 3) == [2, 1]
    assert retrieval._parse_order("no json here", 3) == []


def test_search_knowledge_applies_threshold_and_neighbors(monkeypatch):
    calls: dict = {}

    def fake_search(embedding, *, file_ids=None, top_k=3, max_distance=None):
        calls["top_k"] = top_k
        calls["max_distance"] = max_distance
        return [_row(4, "命中片段")]

    def fake_neighbors(pairs, *, radius=1):
        calls["radius"] = radius
        return {
            (str(FILE_ID), 3): [{"seq": 3, "page": 3, "kind": "text", "text": "上文"}],
            (str(FILE_ID), 5): [{"seq": 5, "page": 5, "kind": "text", "text": "下文"}],
        }

    monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(retrieval.store, "search_kb_chunks", fake_search)
    monkeypatch.setattr(retrieval.store, "kb_chunk_neighbors", fake_neighbors)
    monkeypatch.setattr(settings, "kb_rerank_enabled", False)
    monkeypatch.setattr(settings, "kb_hybrid_enabled", False)
    monkeypatch.setattr(settings, "kb_search_candidates", 16)
    monkeypatch.setattr(settings, "kb_search_max_distance", 0.72)
    monkeypatch.setattr(settings, "kb_neighbor_expand", 1)

    hits = retrieval.search_knowledge("HBM 缺口", top_k=3)

    assert len(hits) == 1
    assert hits[0].before == ["上文"] and hits[0].after == ["下文"]
    assert calls["top_k"] == 16  # 召回放大，再重排/去重后取 top_k
    assert calls["max_distance"] == pytest.approx(0.72)
    assert calls["radius"] == 1


def test_search_knowledge_survives_neighbor_failure(monkeypatch):
    monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1])
    monkeypatch.setattr(
        retrieval.store, "search_kb_chunks", lambda *a, **k: [_row(0, "片段")]
    )

    def boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(retrieval.store, "kb_chunk_neighbors", boom)
    monkeypatch.setattr(settings, "kb_rerank_enabled", False)
    monkeypatch.setattr(settings, "kb_hybrid_enabled", False)
    monkeypatch.setattr(settings, "kb_neighbor_expand", 1)

    hits = retrieval.search_knowledge("问题")
    assert hits[0].text == "片段"
    assert hits[0].before == [] and hits[0].after == []


def test_format_hit_with_context_marks_neighbors_and_position():
    hit = retrieval.Hit(
        file_id=str(FILE_ID),
        filename="半导体报告.pdf",
        seq=4,
        page=5,
        kind="text",
        text="命中正文",
        distance=0.2,
        before=["上一段结尾"],
        after=["下一段开头"],
    )
    out = _format_hit_with_context(hit)
    assert out.startswith("[半导体报告.pdf · 第5页]")
    assert "〔上文〕上一段结尾" in out
    assert "命中正文" in out
    assert "〔下文〕下一段开头" in out
    # 顺序必须是 上文 → 命中 → 下文，模型才能顺着读
    assert out.index("〔上文〕") < out.index("命中正文") < out.index("〔下文〕")


def test_format_hit_with_context_labels_figure_evidence():
    hit = retrieval.Hit(
        file_id=str(FILE_ID),
        filename="年报.pdf",
        seq=0,
        page=2,
        kind="figure",
        text="图中显示 2026 年营收",
        distance=0.4,
    )
    out = _format_hit_with_context(hit)
    assert "第2页图表（图像识别，可能有误差）" in out


def test_kb_search_follows_session_scope_when_model_omits_file_ids(monkeypatch):
    """输入框里选了「只查这几份文件」，工具默认就只查这些，模型不传也一样。"""
    from ai_agent.kb.scope import reset_kb_scope, set_kb_scope
    from ai_agent.tools import kb_search as kb_tool

    seen: dict = {}

    def fake_search(query, *, file_ids=None, top_k=3):
        seen["file_ids"] = file_ids
        return []

    monkeypatch.setattr(kb_tool.store, "has_ready_kb_file", lambda: True)
    monkeypatch.setattr(kb_tool, "search_knowledge", fake_search)

    token = set_kb_scope([str(FILE_ID)])
    try:
        kb_tool._kb_search("问题")
    finally:
        reset_kb_scope(token)

    assert seen["file_ids"] == [FILE_ID]

    # 模型显式给了 file_ids 时以模型为准（它可能只查其中一份）
    other = uuid.UUID("99999999-9999-9999-9999-999999999999")
    token = set_kb_scope([str(FILE_ID)])
    try:
        kb_tool._kb_search("问题", file_ids=[str(other)])
    finally:
        reset_kb_scope(token)
    assert seen["file_ids"] == [other]

    # 无范围 = 全库（None，不加过滤）
    kb_tool._kb_search("问题")
    assert seen["file_ids"] is None


def test_kg_search_follows_session_scope_when_model_omits_file_ids(monkeypatch):
    """图谱检索也要跟输入框的文件范围，否则圈定文件后仍会搜到库里别的实体。"""
    from ai_agent.kb.scope import reset_kb_scope, set_kb_scope
    from ai_agent.tools import kg_search as kg_tool

    seen: dict = {}

    monkeypatch.setattr(kg_tool.store, "graph_stats", lambda **k: {"entities": 3})
    monkeypatch.setattr(kg_tool, "embed_query", lambda q: [0.1])

    def fake_entities(embedding, *, query="", top_k=5, file_ids=None, min_score=0.0):
        seen["file_ids"] = file_ids
        return []

    monkeypatch.setattr(kg_tool.store, "search_entities", fake_entities)

    token = set_kb_scope([str(FILE_ID)])
    try:
        kg_tool._kg_search("问题和谁有关")
    finally:
        reset_kb_scope(token)

    assert seen["file_ids"] == [FILE_ID]

    kg_tool._kg_search("问题和谁有关")
    assert seen["file_ids"] is None


def test_kb_scope_is_isolated_and_resettable():
    from ai_agent.kb.scope import kb_scope, reset_kb_scope, set_kb_scope

    assert kb_scope() == ()
    token = set_kb_scope([" a ", "", "b"])
    try:
        assert kb_scope() == ("a", "b")
    finally:
        reset_kb_scope(token)
    assert kb_scope() == ()


def test_run_loop_applies_kb_scope_around_tool_execution(monkeypatch):
    """范围必须落到「工具执行那一刻」：run_stream 的生成器会被不同线程
    轮流 next()，外层设置的 contextvar 传不进去，只能在执行点同栈设置。"""
    from ai_agent.agents import loop as loop_mod
    from ai_agent.kb.scope import kb_scope

    seen: list[tuple[str, ...]] = []

    def fake_stream(messages, tools, **kwargs):
        if not any(m.get("role") == "tool" for m in messages):
            yield "text", ""
            yield "result", {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "kb_search", "arguments": {"query": "q"}}
                ],
            }
        else:
            yield "text", "好的"
            yield "result", {"content": "好的", "tool_calls": []}

    def fake_execute(name, arguments):
        seen.append(kb_scope())
        return "检索结果"

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)
    monkeypatch.setattr(loop_mod, "execute_tool", fake_execute)

    events = list(
        loop_mod.run_loop(
            [{"role": "user", "content": "问个问题"}],
            user_id="u1",
            kb_file_ids=(str(FILE_ID),),
        )
    )

    assert seen == [(str(FILE_ID),)]
    assert any(e.get("type") == "text" for e in events)
    # 执行完必须复位，不能泄漏到同一线程里的下一次调用
    assert kb_scope() == ()
