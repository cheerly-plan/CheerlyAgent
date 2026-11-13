"""P0 回归：上下文预算与短期记忆的一致性。

覆盖四件事：
1. 被 token 预算挤掉的轮次必须进会话摘要，不能静默蒸发；
2. system 动态块与单轮工具输出超预算时有兜底，绝不发送超限请求；
3. 删除/清空会话后热缓存同步失效；
4. 工具输出裁剪保头保尾并留显式省略标记。
"""

from unittest.mock import MagicMock

from ai_agent.agents.loop import _fit_context, _system_prompt
from ai_agent.core.text import clip_text
from ai_agent.memory.short_term import (
    ShortTermMemory,
    compress,
    fit_turns,
    messages_tokens,
)


def test_clip_text_keeps_head_and_tail_within_limit():
    text = "开头关键结论\n" + "中间内容" * 500 + "\n结尾下载链接 /v1/report/files/a.pdf"
    out = clip_text(text, 600)
    assert len(out) <= 600
    assert "已省略" in out
    assert "开头关键结论" in out
    assert "/v1/report/files/a.pdf" in out
    assert clip_text("短文本", 600) == "短文本"
    assert clip_text(text, 0) is text


def test_fit_turns_reports_dropped_messages():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "旧回答"},
        {"role": "user", "content": "当前问题"},
        {"role": "assistant", "content": "当前回答"},
    ]
    kept, dropped = fit_turns(messages, 10)
    assert [m["content"] for m in dropped] == ["旧问题", "旧回答"]
    assert [m["content"] for turn in kept for m in turn] == ["当前问题", "当前回答"]
    assert messages[1]["content"] == "旧问题"  # 原列表不动


def test_compress_folds_budget_overflow_into_summary():
    """核心回归：轮数窗口没溢出、但token 预算挤掉的轮次也必须进摘要。

    修复前这些轮次只在 _fit_context 的「送模型副本」里被丢，既不进窗口
    也不进摘要/长期记忆，用户视角就是模型突然忘了前面说过什么。
    """
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "关键事实：代号 ALPHA"},
        {"role": "assistant", "content": "收到"},
        {"role": "user", "content": "B" * 600},
        {"role": "assistant", "content": "C" * 600},
        {"role": "user", "content": "现在呢"},
        {"role": "assistant", "content": "回答"},
    ]
    seen: list[list[str]] = []

    def fake_summarize(previous: str, overflow):
        seen.append([str(m.get("content") or "") for m in overflow])
        return "含 ALPHA 的摘要"

    recent, summary = compress(
        messages, old_summary="", summarize=fake_summarize, max_tokens=500
    )
    assert seen and "关键事实：代号 ALPHA" in seen[0]
    assert summary == "含 ALPHA 的摘要"
    assert recent[0]["content"] == "现在呢"


def test_compress_budget_does_not_change_short_conversations():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": "a2"},
    ]
    recent, summary = compress(
        messages,
        summarize=lambda *_: "不该调用",
        max_tokens=40000,
    )
    assert summary == ""
    assert [m["content"] for m in recent] == ["1", "a1", "2", "a2"]


def test_fit_context_caps_oversized_system():
    """system 动态块原先完全不受预算约束，裁完 turns 仍超会把超限请求发出去。"""
    big_system = "静态纪律\n" + "动态内容" * 5000
    messages = [
        {"role": "system", "content": big_system},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
    ]
    fitted = _fit_context(messages, 1000)
    assert messages_tokens(fitted) <= 1000
    assert fitted[0]["content"].startswith("静态纪律")
    assert "省略" in fitted[0]["content"]
    assert messages[0]["content"] == big_system  # 原列表不动


def test_fit_context_shrinks_oversized_tool_output_without_orphans():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "查"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "kb_search", "arguments": {}}],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "kb_search",
            "content": "开头要点\n" + "中间内容" * 1000 + "\n结尾链接 /v1/report/files/x.pdf",
        },
    ]
    fitted = _fit_context(messages, 900)
    assert messages_tokens(fitted) <= 900
    tool_msgs = [m for m in fitted if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "c1"
    assert "已省略" in tool_msgs[0]["content"]
    calls = {c["id"] for m in fitted for c in (m.get("tool_calls") or [])}
    assert all(m["tool_call_id"] in calls for m in tool_msgs)


def test_system_prompt_caps_dynamic_tail(monkeypatch):
    from ai_agent.agents import loop
    from ai_agent.core.config import settings

    monkeypatch.setattr(loop, "_profile_prompt", lambda _uid: "")
    monkeypatch.setattr(loop, "_kb_capability_block", lambda _ids=(): "X" * 20000)
    monkeypatch.setattr(settings, "system_dynamic_max_chars", 1000)
    out = _system_prompt("u1", use_kb=True)
    assert "你是 Cheerly" in out  # 静态纪律完整保留
    assert "省略" in out
    assert len(out) < 4000  # 动态 tail 被裁到上限，而不是 2 万字符


def test_short_term_invalidate_by_session_and_user():
    memory = ShortTermMemory(max_tokens=40000)
    memory.save("s1", [{"role": "user", "content": "1"}], user_id="a")
    memory.save("s2", [{"role": "user", "content": "2"}], user_id="a")
    memory.save("s1", [{"role": "user", "content": "3"}], user_id="b")

    memory.invalidate("a", "s1")
    assert memory.load("s1", user_id="a") is None
    assert memory.load("s2", user_id="a") is not None
    assert memory.load("s1", user_id="b") is not None  # 其他用户不受影响

    memory.invalidate("a")
    assert memory.load("s2", user_id="a") is None
    assert memory.load("s1", user_id="b") is not None


def test_short_term_save_respects_token_budget():
    memory = ShortTermMemory(max_tokens=200)
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "旧" * 300},
        {"role": "assistant", "content": "答"},
        {"role": "user", "content": "现在呢"},
        {"role": "assistant", "content": "回答"},
    ]
    memory.save("s1", messages, user_id="a")
    loaded = memory.load("s1", user_id="a")
    assert loaded is not None
    assert "旧" * 300 not in [m["content"] for m in loaded.messages]
    assert loaded.messages[-1]["content"] == "回答"


def _fake_session_store(monkeypatch, invalidate):
    from ai_agent.memory import session_store as mod

    store = mod.SessionStore()
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"id": "s1"}
    conn_cm = MagicMock()
    conn_cm.__enter__.return_value = conn
    pool = MagicMock()
    pool.connection.return_value = conn_cm
    monkeypatch.setattr(store, "_get_pool", lambda: pool)
    monkeypatch.setattr(mod.short_term_memory, "invalidate", invalidate)
    return store


def test_clear_chat_invalidates_hot_cache(monkeypatch):
    calls: list[tuple] = []
    store = _fake_session_store(monkeypatch, lambda uid, sid=None: calls.append((uid, sid)))
    store.clear_chat("a", "s1")
    assert calls == [("a", "s1")]


def test_clear_all_chat_invalidates_whole_user(monkeypatch):
    calls: list[tuple] = []
    store = _fake_session_store(monkeypatch, lambda uid, sid=None: calls.append((uid, sid)))
    store.clear_chat("a")
    assert calls == [("a", None)]


def test_delete_session_invalidates_hot_cache(monkeypatch):
    calls: list[tuple] = []
    store = _fake_session_store(monkeypatch, lambda uid, sid=None: calls.append((uid, sid)))
    assert store.delete_session("a", "s1") is True
    assert calls == [("a", "s1")]


def test_execute_tool_clips_output_with_marker(monkeypatch):
    from ai_agent.tools.registry import TOOLS_MAP, execute_tool

    tool = TOOLS_MAP["get_weather"]
    long_text = (
        "开头关键结论\n"
        + "中间内容" * 1000
        + "\n结尾下载链接 /v1/report/files/a.pdf"
    )
    monkeypatch.setattr(tool, "run", lambda **_: long_text)
    out = execute_tool("get_weather", {})
    assert len(out) <= tool.max_output_chars
    assert "已省略" in out
    assert "开头关键结论" in out
    assert "/v1/report/files/a.pdf" in out


def _tool_turn(n: int) -> list[dict]:
    return [
        {"role": "user", "content": f"问{n}"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": f"call{n}", "name": "kb_search", "arguments": {"query": "q"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call{n}",
            "name": "kb_search",
            "content": f"[文件{n} · 第{n}页]\n" + "数" * 300,
        },
    ]


def test_context_limits_subtract_reserve_and_apply_soft_ratio(monkeypatch):
    from ai_agent.core.config import settings

    monkeypatch.setattr(settings, "llm_context_token_budget", 10000)
    monkeypatch.setattr(settings, "context_reserve_tokens", 1000)
    monkeypatch.setattr(settings, "context_soft_limit_ratio", 0.8)
    assert settings.context_token_limits() == (9000, 7200)


def test_soft_limit_ages_tool_results_more_aggressively(monkeypatch):
    from ai_agent.core.config import settings

    monkeypatch.setattr(settings, "tool_result_full_turns", 2)
    messages = _tool_turn(1) + _tool_turn(2) + _tool_turn(3)
    fitted = _fit_context(messages, 100000, soft_limit=1)
    assert "原文已省略" in fitted[2]["content"]  # 第 1 轮被提前老化
    assert "原文已省略" in fitted[5]["content"]  # 软水位下只保留当前轮
    assert fitted[8]["content"].endswith("数" * 300)  # 当前轮完整
