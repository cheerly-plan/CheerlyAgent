import contextvars
from unittest.mock import patch

import pytest

from ai_agent.agents.loop import run, run_stream
from ai_agent.memory.short_term import short_term_memory
from ai_agent.memory.store import MemoryItem


@pytest.fixture(autouse=True)
def _isolate_loop_memory():
    short_term_memory.clear()
    with (
        patch("ai_agent.agents.loop.store.list_profile", return_value=[]),
        patch("ai_agent.agents.loop.store.load_chat", return_value=[]),
        patch("ai_agent.agents.loop.store.append_chat"),
    ):
        yield
    short_term_memory.clear()


def _ok_stream(messages, tools):
    yield ("text", "ok")
    yield ("result", {"content": "ok", "tool_calls": []})


def test_loop_returns_model_text_when_no_tool():
    def fake_stream(messages, tools):
        yield ("text", "你好")
        yield ("result", {"content": "你好", "tool_calls": []})

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        assert run("你好") == "你好"


def test_loop_runs_weather_tool():
    calls = {"n": 0}

    def fake_stream(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "get_weather",
                            "arguments": {"city": "北京"},
                        }
                    ],
                },
            )
            return
        tool_msg = messages[-1]
        yield ("text", tool_msg["content"])
        yield ("result", {"content": tool_msg["content"], "tool_calls": []})

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        result = run("北京天气怎么样？")

    assert calls["n"] == 2
    assert "北京" in result
    assert "20度" in result


def test_loop_keeps_session_history():
    seen: list[list[str]] = []

    def fake_stream(messages, tools):
        seen.append([m["role"] for m in messages])
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        run("第一句", chat_session_id="s1")
        run("第二句", chat_session_id="s1")

    assert seen[0] == ["system", "user"]
    assert seen[1] == ["system", "user", "assistant", "user"]
    history = short_term_memory.load("s1")
    assert [m["role"] for m in history] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert history[-1]["content"] == "ok"


def test_loop_compacts_short_term_on_save():
    original = short_term_memory.max_turns
    short_term_memory.max_turns = 2
    try:
        with patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream):
            run("第一句", chat_session_id="s-compact")
            run("第二句", chat_session_id="s-compact")
        history = short_term_memory.load("s-compact")
        assert [m["content"] for m in history if m["role"] != "system"] == [
            "第二句",
            "ok",
        ]
    finally:
        short_term_memory.max_turns = original


def test_loop_injects_profile_memory():
    profile = MemoryItem(
        id="1",
        user_id="u1",
        text="喜欢 pytest",
        created_at="t",
    )
    captured: list[str] = []

    def fake_stream(messages, tools):
        captured.append(messages[0]["content"])
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.store.list_profile", return_value=[profile]),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
    ):
        run("测试习惯", user_id="u1", chat_session_id="s2")

    assert "喜欢 pytest" in captured[0]


def test_run_stream_survives_context_switch_between_yields():
    def fake_stream(messages, tools):
        yield ("text", "你好")
        yield ("result", {"content": "你好", "tool_calls": []})

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        gen = run_stream("你好", user_id="u1", chat_session_id="s3")
        first = next(gen)
        rest = contextvars.Context().run(lambda: list(gen))

    assert first["type"] == "text"
    assert rest[-1]["type"] == "done"


def test_run_stream_appends_new_messages_only():
    appended: list[list[str]] = []

    def fake_append(user_id, chat_session_id, messages):
        appended.append([m["role"] for m in messages])

    with (
        patch("ai_agent.agents.loop.store.append_chat", side_effect=fake_append),
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        run("第一句", user_id="u1", chat_session_id="s1")
        run("第二句", user_id="u1", chat_session_id="s1")

    assert appended[0] == ["user", "assistant"]
    assert appended[1] == ["user", "assistant"]


def test_run_stream_appends_tool_turn():
    calls = {"n": 0}
    appended: list[list[dict]] = []

    def fake_stream(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "get_weather",
                            "arguments": {"city": "北京"},
                        }
                    ],
                },
            )
            return
        yield ("text", "20度")
        yield ("result", {"content": "20度", "tool_calls": []})

    def fake_append(user_id, chat_session_id, messages):
        appended.append(messages)

    with (
        patch("ai_agent.agents.loop.store.append_chat", side_effect=fake_append),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
    ):
        run("北京天气怎么样？", user_id="u1", chat_session_id="s-tool")

    roles = [m["role"] for m in appended[0]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert appended[0][1]["tool_calls"][0]["name"] == "get_weather"
    assert appended[0][2]["tool_call_id"] == "call_1"
    assert "20度" in appended[0][3]["content"]


def test_run_stream_hydrates_from_db_when_memory_empty():
    history = [
        {"role": "user", "content": "上一句"},
        {"role": "assistant", "content": "上次答"},
    ]
    seen: list[list[str]] = []

    def fake_stream(messages, tools):
        seen.append([m["role"] for m in messages])
        yield from _ok_stream(messages, tools)

    with (
        patch("ai_agent.agents.loop.store.load_chat", return_value=history),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
    ):
        run("下一句", user_id="u1", chat_session_id="s-miss")

    assert seen[0] == ["system", "user", "assistant", "user"]


def test_run_stream_skips_db_load_when_short_term_exists():
    poison = [{"role": "user", "content": "不该出现"}]
    seen: list[list[str]] = []

    def fake_stream(messages, tools):
        seen.append([m.get("content") for m in messages if m["role"] != "system"])
        yield from _ok_stream(messages, tools)

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        run("第一句", user_id="u1", chat_session_id="s-hot")

    with (
        patch("ai_agent.agents.loop.store.load_chat", return_value=poison),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
    ):
        run("第二句", user_id="u1", chat_session_id="s-hot")

    assert "不该出现" not in seen[1]
    assert seen[1] == ["第一句", "ok", "第二句"]


def test_run_stream_survives_append_chat_failure():
    with (
        patch(
            "ai_agent.agents.loop.store.append_chat",
            side_effect=RuntimeError("db down"),
        ),
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        assert run("你好", user_id="u1", chat_session_id="s-fail") == "ok"


def test_run_stream_survives_load_chat_failure():
    with (
        patch(
            "ai_agent.agents.loop.store.load_chat",
            side_effect=RuntimeError("db down"),
        ),
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        assert run("你好", user_id="u1", chat_session_id="s-load-fail") == "ok"


def test_run_stream_requests_chat_window():
    with (
        patch("ai_agent.agents.loop.store.load_chat", return_value=[]) as load_chat,
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        run("hi", user_id="u1", chat_session_id="s-lim")
    load_chat.assert_called_once_with(
        "u1", "s-lim", limit=short_term_memory.max_turns
    )


def test_run_stream_compacts_hydrated_history():
    history = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "3"},
        {"role": "assistant", "content": "a3"},
    ]
    seen: list[list[str]] = []
    original = short_term_memory.max_turns
    short_term_memory.max_turns = 2

    def fake_stream(messages, tools):
        seen.append([m["content"] for m in messages if m["role"] != "system"])
        yield from _ok_stream(messages, tools)

    try:
        with (
            patch("ai_agent.agents.loop.store.load_chat", return_value=history),
            patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
        ):
            run("新", user_id="u1", chat_session_id="s-win")
    finally:
        short_term_memory.max_turns = original

    assert seen[0] == ["3", "a3", "新"]


def test_loop_isolates_short_term_by_user():
    with patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream):
        run("给a", user_id="a", chat_session_id="same")
        run("给b", user_id="b", chat_session_id="same")
    assert short_term_memory.load("same", user_id="a")[-2]["content"] == "给a"
    assert short_term_memory.load("same", user_id="b")[-2]["content"] == "给b"