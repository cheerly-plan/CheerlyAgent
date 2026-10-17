from unittest.mock import patch

from ai_agent.agents.loop import run


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
