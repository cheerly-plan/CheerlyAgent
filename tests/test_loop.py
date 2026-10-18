import contextvars
from unittest.mock import patch

import pytest

from ai_agent.agents.loop import run, run_stream
from ai_agent.memory.short_term import short_term_memory
from ai_agent.memory.store import MemoryItem


@pytest.fixture(autouse=True)
def _isolate_loop_memory():
    short_term_memory.clear()
    with patch("ai_agent.agents.loop.store.list_profile", return_value=[]):
        yield
    short_term_memory.clear()


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
        run("第一句", session_id="s1")
        run("第二句", session_id="s1")

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
        run("测试习惯", user_id="u1", session_id="s2")

    assert "喜欢 pytest" in captured[0]


def test_run_stream_survives_context_switch_between_yields():
    def fake_stream(messages, tools):
        yield ("text", "你好")
        yield ("result", {"content": "你好", "tool_calls": []})

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        gen = run_stream("你好", user_id="u1", session_id="s3")
        first = next(gen)
        rest = contextvars.Context().run(lambda: list(gen))

    assert first["type"] == "text"
    assert rest[-1]["type"] == "done"