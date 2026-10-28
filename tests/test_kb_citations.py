"""引用（sources）：检索命中的切片要能传到前端、并随回答一起落库。

工具返回值只能是字符串，引用走 contextvar 旁路，所以这里重点验证
「旁路能收集到 / 没有收集器时不炸 / 跨多次工具调用去重 / 最终挂到回答消息上」。
"""

import uuid

from ai_agent.kb.citations import (
    MAX_SOURCES,
    Citation,
    add_citations,
    collect_citations,
    merge_sources,
)
from ai_agent.kb.retrieval import Hit

FILE_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


def _hit(seq: int = 2, *, page: int | None = 3, kind: str = "text") -> Hit:
    return Hit(
        file_id=str(FILE_ID),
        filename="考勤制度.docx",
        seq=seq,
        page=page,
        kind=kind,
        text="一个月内迟到三次视为旷工，按事假处理。",
        distance=0.2,
    )


def _citation(seq: int = 0) -> Citation:
    return Citation(
        file_id=str(FILE_ID),
        filename="考勤制度.docx",
        seq=seq,
        page=None,
        kind="text",
        where=f"第{seq + 1}段",
        text="正文",
    )


def test_kb_search_records_citations_with_position(monkeypatch):
    from ai_agent.tools import kb_search as kb_tool

    monkeypatch.setattr(kb_tool.store, "has_ready_kb_file", lambda: True)
    monkeypatch.setattr(kb_tool, "search_knowledge", lambda *a, **kw: [_hit()])

    with collect_citations() as cited:
        text = kb_tool._kb_search("迟到怎么处理")

    # 给模型看的文本不变，引用结构另走旁路
    assert "[考勤制度.docx · 第3页]" in text
    assert len(cited) == 1
    payload = cited[0].payload()
    assert payload["file_id"] == str(FILE_ID)
    assert payload["where"] == "第3页"
    assert payload["kind"] == "text"
    assert payload["key"] == f"{FILE_ID}#2"
    assert "迟到三次" in payload["excerpt"]


def test_citation_excerpt_is_clipped_for_very_long_chunks():
    long = Citation(
        file_id=str(FILE_ID),
        filename="长文.pdf",
        seq=0,
        page=1,
        kind="text",
        where="第1页",
        text="甲" * 900,
    )
    excerpt = long.payload()["excerpt"]
    assert len(excerpt) == 601 and excerpt.endswith("…")


def test_citations_without_collector_are_silently_dropped(monkeypatch):
    """单测/脚本直接调工具时没有收集器，不能因为没人接就抛异常。"""
    from ai_agent.tools import kb_search as kb_tool

    monkeypatch.setattr(kb_tool.store, "has_ready_kb_file", lambda: True)
    monkeypatch.setattr(kb_tool, "search_knowledge", lambda *a, **kw: [_hit()])

    text = kb_tool._kb_search("迟到怎么处理")
    assert "考勤制度" in text


def test_merge_sources_dedupes_by_chunk_and_caps_size():
    merged = merge_sources([], [_citation(0), _citation(1), _citation(0)])
    assert [item["seq"] for item in merged] == [0, 1]

    # 再次命中同一片不重复追加
    merged = merge_sources(merged, [_citation(1), _citation(2)])
    assert [item["seq"] for item in merged] == [0, 1, 2]

    # 工具循环失控时不能无限膨胀
    flooded = merge_sources([], [_citation(i) for i in range(MAX_SOURCES + 5)])
    assert len(flooded) == MAX_SOURCES


def test_run_loop_streams_sources_and_attaches_them_to_answer(monkeypatch):
    from ai_agent.agents import loop as loop_mod

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
            yield "text", "迟到三次算旷工。"
            yield "result", {"content": "迟到三次算旷工。", "tool_calls": []}

    def fake_execute(name, arguments):
        # 工具把引用塞进执行点的收集器
        add_citations([_citation(4)])
        return "检索结果"

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)
    monkeypatch.setattr(loop_mod, "execute_tool", fake_execute)

    messages: list = [{"role": "user", "content": "问个问题"}]
    events = list(loop_mod.run_loop(messages, user_id="u1"))

    tool_end = next(e for e in events if e.get("type") == "tool_end")
    assert [s["seq"] for s in tool_end["sources"]] == [4]
    done = next(e for e in events if e.get("type") == "done")
    assert [s["seq"] for s in done["sources"]] == [4]
    # 引用挂在回答消息上（而不是带 tool_calls 的中间消息），历史回放才读得到
    answer = messages[-1]
    assert answer["role"] == "assistant"
    assert answer["meta"]["sources"][0]["key"] == f"{FILE_ID}#4"


def test_run_loop_keeps_sources_from_every_tool_round(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    rounds = {"n": 0}

    def fake_stream(messages, tools, **kwargs):
        rounds["n"] += 1
        if rounds["n"] <= 2:
            yield "text", ""
            yield "result", {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"c{rounds['n']}",
                        "name": "kb_search",
                        # 参数必须不同：完全相同的调用会被去重短路
                        "arguments": {"query": f"问题{rounds['n']}"},
                    }
                ],
            }
        else:
            yield "text", "答"
            yield "result", {"content": "答", "tool_calls": []}

    seqs = iter([1, 2])

    def fake_execute(name, arguments):
        add_citations([_citation(next(seqs))])
        return "结果"

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)
    monkeypatch.setattr(loop_mod, "execute_tool", fake_execute)

    messages: list = [{"role": "user", "content": "问"}]
    events = list(loop_mod.run_loop(messages, user_id="u1"))

    done = next(e for e in events if e.get("type") == "done")
    assert [s["seq"] for s in done["sources"]] == [1, 2]
    assert len(messages[-1]["meta"]["sources"]) == 2


def test_run_loop_omits_sources_when_nothing_was_cited(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    def fake_stream(messages, tools, **kwargs):
        yield "text", "你好"
        yield "result", {"content": "你好", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)

    messages: list = [{"role": "user", "content": "hi"}]
    events = list(loop_mod.run_loop(messages, user_id="u1"))

    done = next(e for e in events if e.get("type") == "done")
    # 没引用时不下发空数组：前端不必区分「空」和「没有」
    assert "sources" not in done
    assert "meta" not in messages[-1]
