"""知识库预检索（always-retrieve）：kb 开关打开时服务端先查一次库再注入。

为什么做这个：靠模型自己决定「这问题要不要查」不稳定，用户把库里的
原文粘进对话框都可能不触发（实测）。这里验证注入、距离过滤、引用上报、
失败静默和续跑不重复注入。
"""

import uuid

from ai_agent.kb.retrieval import Hit

FILE_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


def _hit(distance: float = 0.2, seq: int = 2) -> Hit:
    return Hit(
        file_id=str(FILE_ID),
        filename="招股意向书.PDF",
        seq=seq,
        page=149,
        kind="text",
        text="公司核心技术来源于自主研发与原厂资产注入。",
        distance=distance,
    )


def _enable(monkeypatch):
    from ai_agent.core.config import settings

    monkeypatch.setattr(settings, "kb_prefetch_enabled", True)


def test_prefetch_injects_hits_as_system_message(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    monkeypatch.setattr(loop_mod, "search_knowledge", lambda *a, **kw: [_hit()])

    message, cited = loop_mod._kb_prefetch("核心技术的来源", "u1", ())

    assert message is not None
    assert message["role"] == "system"
    assert message["meta"]["kb_prefetch"] is True
    assert "招股意向书.PDF · 第149页" in message["content"]
    assert "核心技术来源于" in message["content"]
    # 命中同时作为引用上报，前端展示与工具检索一致
    assert cited[0].payload()["key"] == f"{FILE_ID}#2"


def test_prefetch_drops_weak_matches_by_distance(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    # 0.9 远超预检索门槛（0.45）：闲聊时检索器仍会返回「最不坏」的片段，
    # 直接注入会把通用闲聊带偏
    monkeypatch.setattr(loop_mod, "search_knowledge", lambda *a, **kw: [_hit(0.9)])

    message, cited = loop_mod._kb_prefetch("今天天气不错", "u1", ())

    assert message is None and cited == []


def test_prefetch_survives_retrieval_failure(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("embedding 挂了")

    monkeypatch.setattr(loop_mod, "search_knowledge", boom)

    message, cited = loop_mod._kb_prefetch("问题", "u1", ())

    assert message is None and cited == []


def _stub_empty_graph(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    monkeypatch.setattr(loop_mod.store, "graph_stats", lambda **k: {"entities": 0})


def _stub_graph_neighborhood(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    e1 = {"id": uuid.uuid4(), "name": "长鑫存储", "type": "org"}
    e2 = {"id": uuid.uuid4(), "name": "DRAM", "type": "product"}
    monkeypatch.setattr(loop_mod.store, "graph_stats", lambda **k: {"entities": 10})
    monkeypatch.setattr(loop_mod, "embed_query", lambda q: [0.1])
    monkeypatch.setattr(loop_mod.store, "search_entities", lambda *a, **k: [e1])
    monkeypatch.setattr(
        loop_mod.store,
        "expand_subgraph",
        lambda *a, **k: {
            "nodes": [e1, e2],
            "edges": [
                {
                    "source_id": e1["id"],
                    "target_id": e2["id"],
                    "relation": "生产",
                    "description": "",
                }
            ],
        },
    )
    return e1, e2


def test_run_loop_injects_prefetch_before_first_llm_call(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    _stub_empty_graph(monkeypatch)
    monkeypatch.setattr(loop_mod, "search_knowledge", lambda *a, **kw: [_hit()])

    seen: dict = {}

    def fake_stream(messages, tools, **kwargs):
        seen["messages"] = list(messages)
        yield "text", "答"
        yield "result", {"content": "答", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)

    messages: list = [{"role": "user", "content": "核心技术的来源"}]
    events = list(loop_mod.run_loop(messages, user_id="u1"))

    # 注入块排在 user 消息之后、送进模型
    assert seen["messages"][-1]["role"] == "system"
    assert seen["messages"][-1]["meta"]["kb_prefetch"] is True
    done = next(e for e in events if e.get("type") == "done")
    assert [s["seq"] for s in done["sources"]] == [2]


def test_run_loop_skips_prefetch_when_kb_disabled(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    calls: list = []
    monkeypatch.setattr(
        loop_mod, "search_knowledge", lambda *a, **kw: calls.append(1) or []
    )

    def fake_stream(messages, tools, **kwargs):
        yield "text", "你好"
        yield "result", {"content": "你好", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)

    messages: list = [{"role": "user", "content": "hi"}]
    list(loop_mod.run_loop(messages, user_id="u1", use_kb=False))

    assert calls == []  # kb 关闭：剔除工具，也不预检索


def test_run_loop_does_not_reinject_after_tool_round(monkeypatch):
    """断点续跑/工具轮场景：最后一条不是 user，注入块已在快照里，不重复。"""
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    _stub_empty_graph(monkeypatch)
    monkeypatch.setattr(loop_mod, "search_knowledge", lambda *a, **kw: [_hit()])

    seen: dict = {}

    def fake_stream(messages, tools, **kwargs):
        seen.setdefault("calls", 0)
        seen["calls"] += 1
        if seen["calls"] == 1:
            yield "text", ""
            yield "result", {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "kb_search", "arguments": {"query": "q"}}
                ],
            }
        else:
            yield "text", "答"
            yield "result", {"content": "答", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)
    monkeypatch.setattr(loop_mod, "execute_tool", lambda *a, **kw: "结果")

    messages: list = [
        {"role": "user", "content": "问"},
        # 模拟已在中断前跑过一轮：注入块 + assistant(tool_calls) + tool 结果
        {"role": "system", "content": "旧注入", "meta": {"kb_prefetch": True}},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c0", "name": "kb_search", "arguments": {}}],
        },
        {"role": "tool", "tool_call_id": "c0", "name": "kb_search", "content": "旧结果"},
    ]
    list(loop_mod.run_loop(messages, user_id="u1"))

    # 没有追加新的注入块（仍只有那一条旧的）
    prefetch_blocks = [
        m for m in messages if m.get("meta", {}).get("kb_prefetch")
    ]
    assert len(prefetch_blocks) == 1


def test_kg_prefetch_injects_neighborhood_without_chunks(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    _stub_graph_neighborhood(monkeypatch)

    message = loop_mod._kg_prefetch("长鑫存储和谁有关", "u1", ())

    assert message is not None
    assert message["role"] == "system"
    assert message["meta"]["kg_prefetch"] is True
    assert "预检索知识图谱得到的实体邻域" in message["content"]
    assert "长鑫存储(org)" in message["content"]
    assert "—生产→" in message["content"]
    assert "DRAM" in message["content"]
    assert "原文依据" not in message["content"]


def test_kg_prefetch_skips_when_no_seeds(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    monkeypatch.setattr(loop_mod.store, "graph_stats", lambda **k: {"entities": 10})
    monkeypatch.setattr(loop_mod, "embed_query", lambda q: [0.1])
    monkeypatch.setattr(loop_mod.store, "search_entities", lambda *a, **k: [])

    assert loop_mod._kg_prefetch("今天天气不错", "u1", ()) is None


def test_kg_prefetch_survives_store_failure(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)

    def boom(**k):
        raise RuntimeError("db down")

    monkeypatch.setattr(loop_mod.store, "graph_stats", boom)

    assert loop_mod._kg_prefetch("长鑫存储", "u1", ()) is None


def test_prefetch_knowledge_merges_chunk_and_neighborhood(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    _stub_graph_neighborhood(monkeypatch)
    monkeypatch.setattr(loop_mod, "search_knowledge", lambda *a, **kw: [_hit()])

    message, cited = loop_mod._prefetch_knowledge("长鑫存储的产业链", "u1", ())

    assert message is not None
    assert message["meta"]["kb_prefetch"] is True
    assert message["meta"]["kg_prefetch"] is True
    assert "预检索知识库得到的原文片段" in message["content"]
    assert "预检索知识图谱得到的实体邻域" in message["content"]
    assert cited[0].payload()["key"] == f"{FILE_ID}#2"


def test_run_loop_injects_graph_neighborhood_when_chunks_miss(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    _stub_graph_neighborhood(monkeypatch)
    monkeypatch.setattr(loop_mod, "search_knowledge", lambda *a, **kw: [])

    seen: dict = {}

    def fake_stream(messages, tools, **kwargs):
        seen["messages"] = list(messages)
        yield "text", "答"
        yield "result", {"content": "答", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)

    messages: list = [{"role": "user", "content": "长鑫存储和谁有关"}]
    list(loop_mod.run_loop(messages, user_id="u1"))

    assert seen["messages"][-1]["meta"]["kg_prefetch"] is True
    assert "长鑫存储" in seen["messages"][-1]["content"]
