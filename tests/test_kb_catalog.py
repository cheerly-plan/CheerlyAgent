"""知识库文件目录注入 + 对话改写：模型自己决定要不要查、查哪些文件。"""

import uuid

from ai_agent.core.config import settings

FILE_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


def _ready_file(**overrides) -> dict:
    row = {
        "file_id": FILE_ID,
        "filename": "中国能源建设股份有限公司2026年第三季度报告.pdf",
        "status": "ready",
        "tags": ["季报"],
        "doc_type": "quarterly_report",
    }
    row.update(overrides)
    return row


def _enable_catalog(monkeypatch, files=None):
    from ai_agent.agents import loop as loop_mod

    monkeypatch.setattr(settings, "kb_catalog_enabled", True)
    monkeypatch.setattr(
        loop_mod.store,
        "list_kb_files",
        lambda **k: files if files is not None else [_ready_file()],
    )


def test_format_catalog_is_markdown_list_without_ids():
    from ai_agent.agents import loop as loop_mod

    text = loop_mod._format_kb_catalog(
        [_ready_file()],
        original="股东持股情况",
        rewritten="中国能建 股东持股情况",
    )
    assert "文件目录：" in text
    assert "中国能源建设股份有限公司2026年第三季度报告.pdf" in text
    assert "标签：季报" in text
    assert "类型：上市公司季报" in text
    assert "建议检索词：中国能建 股东持股情况" in text
    assert str(FILE_ID) not in text
    assert "不要填文件 id" in text


def test_format_catalog_empty_ready_files():
    from ai_agent.agents import loop as loop_mod

    text = loop_mod._format_kb_catalog([], original="hi", rewritten="hi")
    assert "没有已处理完成的文件" in text
    assert "不要调用 kb_search" in text


def test_run_loop_injects_catalog_before_first_llm_call(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable_catalog(monkeypatch)
    seen: dict = {}

    def fake_stream(messages, tools, **kwargs):
        seen["messages"] = list(messages)
        yield "text", "答"
        yield "result", {"content": "答", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)
    monkeypatch.setattr(loop_mod, "_rewrite_query", lambda q, prev: q)

    messages: list = [{"role": "user", "content": "核心技术的来源"}]
    list(loop_mod.run_loop(messages, user_id="u1"))

    assert seen["messages"][-1]["role"] == "system"
    assert seen["messages"][-1]["meta"]["kb_catalog"] is True
    assert "中国能源建设股份有限公司2026年第三季度报告.pdf" in seen["messages"][-1]["content"]
    assert str(FILE_ID) not in seen["messages"][-1]["content"]


def test_run_loop_skips_catalog_when_kb_disabled(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable_catalog(monkeypatch)
    calls: list = []
    monkeypatch.setattr(
        loop_mod.store, "list_kb_files", lambda **k: calls.append(1) or []
    )

    def fake_stream(messages, tools, **kwargs):
        yield "text", "你好"
        yield "result", {"content": "你好", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)

    messages: list = [{"role": "user", "content": "hi"}]
    list(loop_mod.run_loop(messages, user_id="u1", use_kb=False))

    assert calls == []


def test_run_loop_does_not_reinject_after_tool_round(monkeypatch):
    """断点续跑/工具轮：最后一条不是 user，目录已在快照里，不重复。"""
    from ai_agent.agents import loop as loop_mod

    _enable_catalog(monkeypatch)
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
        {"role": "system", "content": "旧目录", "meta": {"kb_catalog": True}},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c0", "name": "kb_search", "arguments": {}}],
        },
        {"role": "tool", "tool_call_id": "c0", "name": "kb_search", "content": "旧结果"},
    ]
    list(loop_mod.run_loop(messages, user_id="u1"))

    catalog_blocks = [
        m for m in messages if isinstance(m.get("meta"), dict) and m["meta"].get("kb_catalog")
    ]
    assert len(catalog_blocks) == 1
    assert catalog_blocks[0]["content"] == "旧目录"


def test_run_loop_replaces_stale_catalog_on_new_turn(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable_catalog(monkeypatch)
    monkeypatch.setattr(loop_mod, "_rewrite_query", lambda q, prev: q)

    def fake_stream(messages, tools, **kwargs):
        yield "text", "答"
        yield "result", {"content": "答", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)

    messages: list = [
        {"role": "user", "content": "上一问"},
        {"role": "system", "content": "旧目录", "meta": {"kb_catalog": True}},
        {"role": "assistant", "content": "旧答"},
        {"role": "user", "content": "这一问"},
    ]
    list(loop_mod.run_loop(messages, user_id="u1"))

    catalog_blocks = [
        m for m in messages if isinstance(m.get("meta"), dict) and m["meta"].get("kb_catalog")
    ]
    assert len(catalog_blocks) == 1
    assert "旧目录" not in catalog_blocks[0]["content"]
    assert "中国能源建设股份有限公司2026年第三季度报告.pdf" in catalog_blocks[0]["content"]


def test_run_loop_forces_kb_search_when_model_skips(monkeypatch):
    """知识库开着时模型直接作答：丢掉这段正文，服务端代调 kb_search。"""
    from ai_agent.agents import loop as loop_mod

    _enable_catalog(monkeypatch)
    monkeypatch.setattr(settings, "kb_force_search", True)
    monkeypatch.setattr(loop_mod, "_rewrite_query", lambda q, prev: q)
    calls: list[int] = []
    executed: list[tuple[str, dict]] = []

    def fake_stream(messages, tools, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            yield "text", "按我的了解，这款充电器去年开售"
            yield "result", {
                "content": "按我的了解，这款充电器去年开售",
                "tool_calls": [],
            }
            return
        yield "text", "文档写的是 2026 年 3 月"
        yield "result", {"content": "文档写的是 2026 年 3 月", "tool_calls": []}

    def fake_execute(name, arguments):
        executed.append((name, arguments))
        return "[季报.pdf · 第12页]\n2026 年 3 月开售"

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)
    monkeypatch.setattr(loop_mod, "execute_tool", fake_execute)

    events = list(
        loop_mod.run_loop(
            [{"role": "user", "content": "安心充什么时候开售"}],
            user_id="u1",
        )
    )
    texts = [e.get("content") for e in events if e.get("type") == "text"]
    assert "按我的了解，这款充电器去年开售" not in texts
    assert "文档写的是 2026 年 3 月" in texts
    assert any(e.get("type") == "tool_start" and e.get("name") == "kb_search" for e in events)
    assert executed and executed[0][0] == "kb_search"
    assert executed[0][1]["query"] == "安心充什么时候开售"


def test_run_loop_does_not_force_kb_search_for_weather(monkeypatch):
    """出行/天气不能强行检索：关键词会把「深圳」撞上「深圳市顺丰投资」。"""
    from ai_agent.agents import loop as loop_mod

    _enable_catalog(monkeypatch)
    monkeypatch.setattr(settings, "kb_force_search", True)
    monkeypatch.setattr(loop_mod, "_rewrite_query", lambda q, prev: q)
    monkeypatch.setattr(loop_mod, "query_looks_on_topic", lambda q, **k: False)
    executed: list = []

    def fake_stream(messages, tools, **kwargs):
        yield "text", "建议看明天的天气预报"
        yield "result", {"content": "建议看明天的天气预报", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)
    monkeypatch.setattr(loop_mod, "execute_tool", lambda *a, **k: executed.append(a) or "")

    events = list(
        loop_mod.run_loop(
            [{"role": "user", "content": "深圳宝安明天出门需要带伞吗"}],
            user_id="u1",
        )
    )
    texts = [e.get("content") for e in events if e.get("type") == "text"]
    assert "建议看明天的天气预报" in texts
    assert executed == []
    assert not any(e.get("name") == "kb_search" for e in events)


def test_skip_forced_search_weather_and_chitchat(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    monkeypatch.setattr(loop_mod, "query_looks_on_topic", lambda q, **k: False)
    assert loop_mod._skip_forced_kb_search("深圳宝安明天出门需要带伞吗") is True
    assert loop_mod._skip_forced_kb_search("你好") is True
    assert loop_mod._skip_forced_kb_search("安心充什么时候开售") is False

    monkeypatch.setattr(loop_mod, "query_looks_on_topic", lambda q, **k: True)
    assert loop_mod._skip_forced_kb_search("顺丰在深圳遇上台风对物流有何影响") is False


def test_catalog_survives_list_failure(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable_catalog(monkeypatch)

    def boom(**k):
        raise RuntimeError("db down")

    monkeypatch.setattr(loop_mod.store, "list_kb_files", boom)
    seen: dict = {}

    def fake_stream(messages, tools, **kwargs):
        seen["messages"] = list(messages)
        yield "text", "答"
        yield "result", {"content": "答", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)
    monkeypatch.setattr(loop_mod, "_rewrite_query", lambda q, prev: q)

    messages: list = [{"role": "user", "content": "问题"}]
    list(loop_mod.run_loop(messages, user_id="u1"))

    assert not any(
        isinstance(m.get("meta"), dict) and m["meta"].get("kb_catalog")
        for m in seen["messages"]
    )


def test_run_loop_passes_rewrite_hint_into_tool(monkeypatch):
    from ai_agent.agents import loop as loop_mod
    from ai_agent.kb.scope import effective_kb_query

    _enable_catalog(monkeypatch)
    monkeypatch.setattr(
        loop_mod, "_rewrite_query", lambda q, prev: "中国能建 股东持股情况"
    )
    seen: dict = {}

    def fake_stream(messages, tools, **kwargs):
        if not any(m.get("role") == "tool" for m in messages):
            yield "text", ""
            yield "result", {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "kb_search",
                        "arguments": {"query": "股东持股情况"},
                    }
                ],
            }
        else:
            yield "text", "答"
            yield "result", {"content": "答", "tool_calls": []}

    def fake_execute(name, arguments):
        seen["effective"] = effective_kb_query(arguments.get("query") or "")
        return "结果"

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)
    monkeypatch.setattr(loop_mod, "execute_tool", fake_execute)

    list(
        loop_mod.run_loop(
            [{"role": "user", "content": "股东持股情况"}],
            user_id="u1",
        )
    )
    assert seen["effective"] == "中国能建 股东持股情况"


# ——— 对话改写（补全 + 扩写）——


def _enable_rewrite(monkeypatch):
    monkeypatch.setattr(settings, "kb_rewrite_enabled", True)


def test_rewrite_skips_without_previous_turn(monkeypatch):
    """第一问无上文：不调模型、原样返回。"""
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)
    calls: list = []
    monkeypatch.setattr(loop_mod, "complete_chat", lambda *a, **kw: calls.append(1) or "")

    assert loop_mod._rewrite_query("股东持股情况", []) == "股东持股情况"
    assert loop_mod._rewrite_query("中国能源建设股份有限公司", ["   "]) == "中国能源建设股份有限公司"
    assert calls == []


def test_rewrite_skips_when_disabled(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    monkeypatch.setattr(settings, "kb_rewrite_enabled", False)
    calls: list = []
    monkeypatch.setattr(loop_mod, "complete_chat", lambda *a, **kw: calls.append(1) or "")

    assert loop_mod._rewrite_query("股东持股情况", ["中国能源建设股份有限公司"]) == "股东持股情况"
    assert calls == []


def test_rewrite_fills_in_elided_subject(monkeypatch):
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
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)
    seen: dict = {}

    def fake(messages, **kw):
        seen["user_content"] = messages[-1]["content"]
        return "安克创新 2026年第三季度 毛利率变动原因"

    monkeypatch.setattr(loop_mod, "complete_chat", fake)

    prev_turn = [
        "归母净利润、扣非净利润分别是多少",
        "安克创新科技股份有限公司 2026 年第三季度归母净利润为 5.59 亿元…",
    ]
    out = loop_mod._rewrite_query("本季度毛利率变动的原因", prev_turn)

    assert out == "安克创新 2026年第三季度 毛利率变动原因"
    assert "安克创新" in seen["user_content"]


def test_rewrite_allows_expansion(monkeypatch):
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
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)
    monkeypatch.setattr(loop_mod, "complete_chat", lambda *a, **kw: '"股东持股情况"')

    assert loop_mod._rewrite_query("股东持股情况", ["上一轮问别的"]) == "股东持股情况"


def test_rewrite_rejects_answer_like_output(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)
    monkeypatch.setattr(
        loop_mod,
        "complete_chat",
        lambda *a, **kw: (
            "根据知识库中的信息，开售时间未明确提及，建议访问安克创新官方网站。"
        ),
    )

    assert loop_mod._rewrite_query("开售时间", ["安克创新"]) == "开售时间"


def test_rewrite_falls_back_on_llm_error(monkeypatch):
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
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)
    seen: dict = {}

    def fake_rewrite(user_input, prev_turn):
        seen["prev_turn"] = prev_turn
        return user_input

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

    assert len(seen["prev_turn"]) == 2
    assert any("安克创新" in s for s in seen["prev_turn"])
    assert not any("单季营收" in s for s in seen["prev_turn"])


def test_run_loop_rewrite_skips_tool_call_intermediate(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)
    seen: dict = {}

    def fake_rewrite(user_input, prev_turn):
        seen["prev_turn"] = prev_turn
        return user_input

    monkeypatch.setattr(loop_mod, "_rewrite_query", fake_rewrite)

    def fake_stream(messages, tools, **kwargs):
        yield "text", "答"
        yield "result", {"content": "答", "tool_calls": []}

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream)

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

    assert len(seen["prev_turn"]) == 2
    assert any("安克创新" in s for s in seen["prev_turn"])
    assert not any(s == "" for s in seen["prev_turn"])


def test_run_loop_rewrite_collects_two_turns(monkeypatch):
    from ai_agent.agents import loop as loop_mod

    _enable_rewrite(monkeypatch)
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

    messages: list = [
        {"role": "user", "content": "安克创新2026第三季度营收情况"},
        {"role": "assistant", "content": "安克创新科技股份有限公司单季营收 95.35 亿元，同比 +16.96%。"},
        {"role": "user", "content": "归母净利润多少"},
        {"role": "assistant", "content": "归母净利润 5.59 亿元，扣非 8.40 亿元。"},
        {"role": "user", "content": "单季营收同比增速呢"},
    ]
    list(loop_mod.run_loop(messages, user_id="u1"))

    assert any("安克创新" in s for s in seen["prev_turn"])
    assert not any("单季营收同比增速" in s for s in seen["prev_turn"])
