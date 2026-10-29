"""知识库预检索（always-retrieve）：kb 开关打开时服务端先查一次库再注入。

为什么做这个：靠模型自己决定「这问题要不要查」不稳定，用户把库里的
原文粘进对话框都可能不触发（实测）。这里验证注入、距离过滤、引用上报、
失败静默和续跑不重复注入。
"""

import uuid

from ai_agent.kb.retrieval import Hit

FILE_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


def _hit(distance: float | None = 0.2, seq: int = 2) -> Hit:
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


def test_prefetch_keeps_keyword_only_hits(monkeypatch):
    """编号/专名常走关键词路，没有向量距离，不能当成弱匹配丢掉。"""
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    monkeypatch.setattr(
        loop_mod, "search_knowledge", lambda *a, **kw: [_hit(None)]
    )

    message, cited = loop_mod._kb_prefetch("BMS-001告警", "u1", ())

    assert message is not None and cited
    assert "关键词命中" in message["content"]


def test_prefetch_does_not_let_keyword_hits_bypass_weak_vector_gate(monkeypatch):
    """弱向量 + 关键词独有命中：关键词不能写成距离 0 把闲聊门控打穿。"""
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    monkeypatch.setattr(
        loop_mod,
        "search_knowledge",
        lambda *a, **kw: [_hit(0.9, seq=1), _hit(None, seq=2)],
    )

    message, cited = loop_mod._kb_prefetch("今天天气不错", "u1", ())

    assert message is None and cited == []


def test_prefetch_drops_weak_matches_by_distance(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    # 0.9 远超预检索门槛（0.45）：闲聊时检索器仍会返回「最不坏」的片段，
    # 直接注入会把通用闲聊带偏
    monkeypatch.setattr(loop_mod, "search_knowledge", lambda *a, **kw: [_hit(0.9)])

    message, cited = loop_mod._kb_prefetch("今天天气不错", "u1", ())

    assert message is None and cited == []


def test_prefetch_keeps_reranked_hits_when_query_is_close(monkeypatch):
    """有一条足够近，就保留整组 top_k：重排挑中的相关原文距离可能略高于门槛。"""
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    monkeypatch.setattr(
        loop_mod,
        "search_knowledge",
        lambda *a, **kw: [_hit(0.20, seq=1), _hit(0.52, seq=2)],
    )

    message, cited = loop_mod._kb_prefetch("核心技术的来源", "u1", ())

    assert message is not None
    assert [c.seq for c in cited] == [1, 2]
    assert "命中 1" in message["content"] and "命中 2" in message["content"]


def test_prefetch_forwards_selected_file_ids(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    seen: dict = {}

    def fake_search(query, *, file_ids=None, top_k=3):
        seen["file_ids"] = file_ids
        seen["top_k"] = top_k
        return [_hit()]

    monkeypatch.setattr(loop_mod, "search_knowledge", fake_search)

    loop_mod._kb_prefetch("核心技术的来源", "u1", (str(FILE_ID),))

    assert seen["file_ids"] == [FILE_ID]


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


# ——— 对话改写（补全 + 扩写）——


def _enable_rewrite(monkeypatch):
    from ai_agent.core.config import settings

    monkeypatch.setattr(settings, "kb_rewrite_enabled", True)


def test_rewrite_skips_without_previous_turn(monkeypatch):
    """第一问无上文：不调模型、原样返回。"""
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)
    calls: list = []
    monkeypatch.setattr(loop_mod, "complete_chat", lambda *a, **kw: calls.append(1) or "")

    assert loop_mod._rewrite_query("股东持股情况", []) == "股东持股情况"
    assert loop_mod._rewrite_query("中国能源建设股份有限公司", ["   "]) == "中国能源建设股份有限公司"
    assert calls == []  # 无上文根本不该发模型调用


def test_rewrite_skips_when_disabled(monkeypatch):
    from ai_agent.agents import loop as loop_mod
    from ai_agent.core.config import settings

    monkeypatch.setattr(settings, "kb_rewrite_enabled", False)
    calls: list = []
    monkeypatch.setattr(loop_mod, "complete_chat", lambda *a, **kw: calls.append(1) or "")

    assert loop_mod._rewrite_query("股东持股情况", ["中国能源建设股份有限公司"]) == "股东持股情况"
    assert calls == []


def test_rewrite_fills_in_elided_subject(monkeypatch):
    """有上文时把省略的主体补回 query。"""
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)
    monkeypatch.setattr(
        loop_mod,
        "complete_chat",
        lambda messages, **kw: "中国能源建设股份有限公司 股东持股情况",
    )

    out = loop_mod._rewrite_query("股东持股情况", ["中国能源建设股份有限公司的股东"])
    assert out == "中国能源建设股份有限公司 股东持股情况"


def test_rewrite_sees_subject_from_previous_turn(monkeypatch):
    """主体藏在上一轮助手的回答里：改写应从上一轮 user+assistant 里拿到主体。

    对应线上 bug：seq1 问「安克创新…」，seq3 问「归母净利润」（无主体），
    助手回答里带出「安克创新」；seq9 问「本季度毛利率」（无主体）——只喂上一句
    用户原话时改写模型看不到「安克创新」，把「本季度」补成了别家公司。
    这里验证：喂上一轮 user+assistant 时，模型能拿到主体。
    """
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)
    seen: dict = {}

    def fake(messages, **kw):
        seen["user_content"] = messages[-1]["content"]
        return "安克创新 2026年第三季度 毛利率变动原因"

    monkeypatch.setattr(loop_mod, "complete_chat", fake)

    # 上一轮：用户问「归母净利润」（无主体），助手答（含主体「安克创新」）
    prev_turn = [
        "归母净利润、扣非净利润分别是多少",
        "安克创新科技股份有限公司 2026 年第三季度归母净利润为 5.59 亿元…",
    ]
    out = loop_mod._rewrite_query("本季度毛利率变动的原因", prev_turn)

    assert out == "安克创新 2026年第三季度 毛利率变动原因"
    # 喂给模型的上下文里必须包含助手回答里的主体「安克创新」
    assert "安克创新" in seen["user_content"]


def test_rewrite_allows_expansion(monkeypatch):
    """允许扩写：模型补充同义词/相关要点也算合法改写。"""
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)
    monkeypatch.setattr(
        loop_mod,
        "complete_chat",
        lambda messages, **kw: "中国能源建设股份有限公司 前十大股东 持股比例 股本结构",
    )

    out = loop_mod._rewrite_query("股东持股情况", ["中国能源建设股份有限公司"])
    assert out == "中国能源建设股份有限公司 前十大股东 持股比例 股本结构"


def test_rewrite_returns_original_when_identical(monkeypatch):
    """模型返回和原句一样（含引导符）：剥掉后一致，退回原话。"""
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)
    monkeypatch.setattr(loop_mod, "complete_chat", lambda *a, **kw: '"股东持股情况"')

    assert loop_mod._rewrite_query("股东持股情况", ["上一轮问别的"]) == "股东持股情况"


def test_rewrite_falls_back_on_llm_error(monkeypatch):
    """小模型失败：退回原话，不抛异常、不挡回答。"""
    from ai_agent.agents import loop as loop_mod
    from ai_agent.core.exceptions import LLMError

    _enable_rewrite(monkeypatch)

    def boom(*a, **kw):
        raise LLMError("timeout")

    monkeypatch.setattr(loop_mod, "complete_chat", boom)

    assert loop_mod._rewrite_query("股东持股情况", ["中国能源建设股份有限公司"]) == "股东持股情况"


def test_rewrite_falls_back_on_unexpected_error(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(loop_mod, "complete_chat", boom)

    assert loop_mod._rewrite_query("股东持股情况", ["中国能源建设股份有限公司"]) == "股东持股情况"


def test_run_loop_passes_previous_turn_to_rewrite(monkeypatch):
    """回归：调用点提取「上一轮 user+assistant」必须跳过当前 user。

    对应线上 bug：seq1 问「安克创新…」、seq2 助手答（含主体）、seq3 问
    「单季营收」（无主体）。旧实现从 messages[-1] 开始找上一轮 user，
    把「单季营收」这句当前问题当成了 prev_turn，改写上下文里没有「安克创新」，
    追问漂到中国能源建设。这里验证：喂给 _rewrite_query 的 prev_turn 含主体。
    """
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    _enable_rewrite(monkeypatch)
    _stub_empty_graph(monkeypatch)
    monkeypatch.setattr(loop_mod, "search_knowledge", lambda *a, **kw: [])

    seen: dict = {}

    def fake_rewrite(user_input, prev_turn):
        seen["prev_turn"] = prev_turn
        return user_input  # 改写结果不影响断言，关键是 prev_turn 内容

    monkeypatch.setattr(loop_mod, "_rewrite_query", fake_rewrite)

    def fake_stream(messages, tools, **kwargs):
        yield "text", "答"
        yield "result", {"content": "答", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)

    messages: list = [
        {"role": "user", "content": "安克创新2026第三季度 归母净利润、扣非净利润分别是多少"},
        {"role": "assistant", "content": "根据安克创新科技股份有限公司2026年第三季度报告：归母净利润 558,794,649.36 元…"},
        {"role": "user", "content": "单季营收、同比增速是多少"},
    ]
    list(loop_mod.run_loop(messages, user_id="u1"))

    # 上一轮 user 问句 + assistant 回答都喂进去了，且都含主体「安克创新」
    assert len(seen["prev_turn"]) == 2
    assert any("安克创新" in s for s in seen["prev_turn"])
    # 当前这句「单季营收」（无主体）不能混进 prev_turn
    assert not any("单季营收" in s for s in seen["prev_turn"])


def test_run_loop_rewrite_skips_tool_call_intermediate(monkeypatch):
    """上一轮带工具调用时，取「最终回答」而非空 content 的 tool_calls 中间态。"""
    from ai_agent.agents import loop as loop_mod

    _enable(monkeypatch)
    _enable_rewrite(monkeypatch)
    _stub_empty_graph(monkeypatch)
    monkeypatch.setattr(loop_mod, "search_knowledge", lambda *a, **kw: [])

    seen: dict = {}

    def fake_rewrite(user_input, prev_turn):
        seen["prev_turn"] = prev_turn
        return user_input

    monkeypatch.setattr(loop_mod, "_rewrite_query", fake_rewrite)

    def fake_stream(messages, tools, **kwargs):
        yield "text", "答"
        yield "result", {"content": "答", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)

    # 上一轮带工具：user -> assistant(tool_calls, content 空) -> tool -> assistant(最终回答)
    messages: list = [
        {"role": "user", "content": "安克创新归母净利润多少"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "kb_search", "arguments": {"query": "安克创新"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "kb_search", "content": "检索结果…"},
        {"role": "assistant", "content": "安克创新科技股份有限公司归母净利润 5.59 亿元。"},
        {"role": "user", "content": "单季营收呢"},
    ]
    list(loop_mod.run_loop(messages, user_id="u1"))

    # 最终回答（含主体）被取进来，空 content 的中间态不出现
    assert len(seen["prev_turn"]) == 2
    assert any("安克创新" in s for s in seen["prev_turn"])
    assert not any(s == "" for s in seen["prev_turn"])


def test_run_loop_rewrite_collects_two_turns(monkeypatch):
    """多轮：主体在最前一轮、中间隔一轮无主体追问，2 轮上下文能拿到主体。"""
    from ai_agent.agents import loop as loop_mod
    from ai_agent.core.config import settings

    _enable(monkeypatch)
    _enable_rewrite(monkeypatch)
    _stub_empty_graph(monkeypatch)
    monkeypatch.setattr(loop_mod, "search_knowledge", lambda *a, **kw: [])
    monkeypatch.setattr(settings, "kb_rewrite_history_turns", 2)

    seen: dict = {}

    def fake_rewrite(user_input, prev_turn):
        seen["prev_turn"] = prev_turn
        return user_input

    monkeypatch.setattr(loop_mod, "_rewrite_query", fake_rewrite)

    def fake_stream(messages, tools, **kwargs):
        yield "text", "答"
        yield "result", {"content": "答", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)

    # seq1 有主体，seq3 无主体（回答里也没带主体），seq5 无主体
    messages: list = [
        {"role": "user", "content": "安克创新2026第三季度营收情况"},
        {"role": "assistant", "content": "安克创新科技股份有限公司单季营收 95.35 亿元，同比 +16.96%。"},
        {"role": "user", "content": "归母净利润多少"},
        {"role": "assistant", "content": "归母净利润 5.59 亿元，扣非 8.40 亿元。"},
        {"role": "user", "content": "单季营收同比增速呢"},
    ]
    list(loop_mod.run_loop(messages, user_id="u1"))

    # 两轮上下文里，最前一轮的「安克创新」被保留
    assert any("安克创新" in s for s in seen["prev_turn"])
    # 当前这句不能混进 prev_turn
    assert not any("单季营收同比增速" in s for s in seen["prev_turn"])

