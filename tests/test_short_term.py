from ai_agent.memory.short_term import compact


def test_compact_keeps_system_and_recent():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": "a2"},
    ]
    out = compact(messages, keep=2)
    assert out[0]["role"] == "system"
    assert [m["content"] for m in out[1:]] == ["2", "a2"]


def test_compact_drops_leading_tool():
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "1", "name": "x", "arguments": {}}],
        },
        {"role": "tool", "tool_call_id": "1", "name": "x", "content": "ok"},
        {"role": "user", "content": "hi"},
    ]
    out = compact(messages, keep=2)
    assert out[0]["role"] == "system"
    assert out[1]["role"] != "tool"