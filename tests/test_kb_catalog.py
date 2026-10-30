"""检索改写：多轮对话里的追问改写成自包含的检索词。"""

from ai_agent.core.config import settings


def _enable_rewrite(monkeypatch):
    monkeypatch.setattr(settings, "kb_rewrite_enabled", True)


def test_rewrite_skips_without_previous_turn(monkeypatch):
    """第一问无上文：不调模型、原样返回。"""
    from ai_agent.agents import loop as loop_mod, rewrite as rewrite_mod

    _enable_rewrite(monkeypatch)
    calls: list = []
    monkeypatch.setattr(rewrite_mod, "complete_chat", lambda *a, **kw: calls.append(1) or "")

    assert loop_mod._rewrite_query("股东持股情况", []) == "股东持股情况"
    assert loop_mod._rewrite_query("中国能源建设股份有限公司", ["   "]) == "中国能源建设股份有限公司"
    assert calls == []


def test_rewrite_skips_when_disabled(monkeypatch):
    from ai_agent.agents import loop as loop_mod, rewrite as rewrite_mod

    monkeypatch.setattr(settings, "kb_rewrite_enabled", False)
    calls: list = []
    monkeypatch.setattr(rewrite_mod, "complete_chat", lambda *a, **kw: calls.append(1) or "")

    assert loop_mod._rewrite_query("股东持股情况", ["中国能源建设股份有限公司"]) == "股东持股情况"
    assert calls == []


def test_rewrite_fills_in_elided_subject(monkeypatch):
    from ai_agent.agents import loop as loop_mod, rewrite as rewrite_mod

    _enable_rewrite(monkeypatch)
    monkeypatch.setattr(
        rewrite_mod,
        "complete_chat",
        lambda messages, **kw: "中国能源建设股份有限公司 股东持股情况",
    )

    out = loop_mod._rewrite_query("股东持股情况", ["中国能源建设股份有限公司的股东"])
    assert out == "中国能源建设股份有限公司 股东持股情况"


def test_rewrite_sees_subject_from_previous_turn(monkeypatch):
    from ai_agent.agents import loop as loop_mod, rewrite as rewrite_mod

    _enable_rewrite(monkeypatch)
    seen: dict = {}

    def fake(messages, **kw):
        seen["user_content"] = messages[-1]["content"]
        return "安克创新 2026年第三季度 毛利率变动原因"

    monkeypatch.setattr(rewrite_mod, "complete_chat", fake)

    prev_turn = [
        "归母净利润、扣非净利润分别是多少",
        "安克创新科技股份有限公司 2026 年第三季度归母净利润为 5.59 亿元…",
    ]
    out = loop_mod._rewrite_query("本季度毛利率变动的原因", prev_turn)

    assert out == "安克创新 2026年第三季度 毛利率变动原因"
    assert "安克创新" in seen["user_content"]


def test_rewrite_allows_expansion(monkeypatch):
    from ai_agent.agents import loop as loop_mod, rewrite as rewrite_mod

    _enable_rewrite(monkeypatch)
    monkeypatch.setattr(
        rewrite_mod,
        "complete_chat",
        lambda messages, **kw: "中国能源建设股份有限公司 前十大股东 持股比例 股本结构",
    )

    out = loop_mod._rewrite_query("股东持股情况", ["中国能源建设股份有限公司"])
    assert out == "中国能源建设股份有限公司 前十大股东 持股比例 股本结构"


def test_rewrite_returns_original_when_identical(monkeypatch):
    from ai_agent.agents import loop as loop_mod, rewrite as rewrite_mod

    _enable_rewrite(monkeypatch)
    monkeypatch.setattr(rewrite_mod, "complete_chat", lambda *a, **kw: '"股东持股情况"')

    assert loop_mod._rewrite_query("股东持股情况", ["上一轮问别的"]) == "股东持股情况"


def test_rewrite_rejects_answer_like_output(monkeypatch):
    from ai_agent.agents import loop as loop_mod, rewrite as rewrite_mod

    _enable_rewrite(monkeypatch)
    monkeypatch.setattr(
        rewrite_mod,
        "complete_chat",
        lambda *a, **kw: (
            "根据知识库中的信息，开售时间未明确提及，建议访问安克创新官方网站。"
        ),
    )

    assert loop_mod._rewrite_query("开售时间", ["安克创新"]) == "开售时间"


def test_rewrite_falls_back_on_llm_error(monkeypatch):
    from ai_agent.agents import loop as loop_mod, rewrite as rewrite_mod
    from ai_agent.core.exceptions import LLMError

    _enable_rewrite(monkeypatch)

    def boom(*a, **kw):
        raise LLMError("timeout")

    monkeypatch.setattr(rewrite_mod, "complete_chat", boom)

    assert loop_mod._rewrite_query("股东持股情况", ["中国能源建设股份有限公司"]) == "股东持股情况"


def test_rewrite_falls_back_on_unexpected_error(monkeypatch):
    from ai_agent.agents import loop as loop_mod, rewrite as rewrite_mod

    _enable_rewrite(monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(rewrite_mod, "complete_chat", boom)

    assert loop_mod._rewrite_query("股东持股情况", ["中国能源建设股份有限公司"]) == "股东持股情况"


def test_run_loop_passes_previous_turn_to_rewrite(monkeypatch):
    from ai_agent.agents import loop as loop_mod, rewrite as rewrite_mod

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
    from ai_agent.agents import loop as loop_mod, rewrite as rewrite_mod

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
    from ai_agent.agents import loop as loop_mod, rewrite as rewrite_mod

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
