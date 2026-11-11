"""历史轮次的工具原文老化：当前轮完整，更早的压成一行引用。"""

from ai_agent.agents.messages import age_tool_results, messages_chars


def _turn(n: int, tool_chars: int = 800) -> list[dict]:
    return [
        {"role": "user", "content": f"问题{n}"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call{n}",
                    "name": "kb_search",
                    "arguments": {"query": f"查询{n}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"call{n}", "name": "kb_search",
         "content": f"[文件{n} · 第{n}页]\n" + "数" * tool_chars},
    ]


def test_current_turn_keeps_full_text():
    messages = _turn(1) + _turn(2) + _turn(3)
    aged = age_tool_results(messages, keep_turns=2)
    assert aged[-1]["content"].endswith("数" * 800)      # 当前轮完整
    assert aged[-4]["content"].endswith("数" * 800)      # 上一轮完整


def test_older_turn_is_replaced_by_reference():
    messages = _turn(1) + _turn(2) + _turn(3)
    aged = age_tool_results(messages, keep_turns=2)
    old = aged[2]["content"]
    assert len(old) < 200
    assert "原文已省略" in old
    assert "kb_search" in old          # 带上当时的调用，方便照着再查
    assert "数" * 100 not in old


def test_saves_most_of_the_window():
    messages = _turn(1) + _turn(2) + _turn(3) + _turn(4)
    before = messages_chars(messages)
    after = messages_chars(age_tool_results(messages, keep_turns=2))
    assert after < before * 0.6


def test_negative_disables_aging():
    messages = _turn(1) + _turn(2) + _turn(3)
    assert age_tool_results(messages, keep_turns=-1) is messages


def test_no_tool_messages_returns_same_object():
    messages = [
        {"role": "user", "content": "问1"},
        {"role": "assistant", "content": "答1"},
        {"role": "user", "content": "问2"},
    ]
    assert age_tool_results(messages, keep_turns=1) is messages
