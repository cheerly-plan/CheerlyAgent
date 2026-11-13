"""历史轮次的工具原文老化：当前轮完整，更早的压成一行引用。"""

from ai_agent.agents.messages import age_tool_results, messages_tokens


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
    before = messages_tokens(messages)
    after = messages_tokens(age_tool_results(messages, keep_turns=2))
    assert after < before * 0.65


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



def test_aged_stub_points_to_location():
    """老化行用「文件名 · 第N页」指路——不引入任何内部编号。"""
    content = "[某公司 · 第10页]\n" + "数" * 900
    messages = [
        {"role": "user", "content": "问1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "kb_search", "arguments": {"query": "q"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "kb_search", "content": content},
        {"role": "user", "content": "问2"},
        {"role": "user", "content": "问3"},
    ]
    stub = age_tool_results(messages, keep_turns=2)[2]["content"]
    assert "kb_read(location=" in stub
    assert "ref=" not in stub


def test_target_aging_prefers_large_uncited_and_keeps_current_turn():
    messages = [
        {"role": "user", "content": "问1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "big", "name": "web_search", "arguments": {}}],
        },
        {
            "role": "tool",
            "tool_call_id": "big",
            "name": "web_search",
            "content": "大" * 2000,
        },
        {"role": "user", "content": "问2"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "cited", "name": "kb_search", "arguments": {}}],
        },
        {
            "role": "tool",
            "tool_call_id": "cited",
            "name": "kb_search",
            "content": "小" * 500,
            "meta": {"citations": [{"filename": "a.pdf", "page": 1}]},
        },
        {"role": "user", "content": "问3"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "cur", "name": "web_search", "arguments": {}}],
        },
        {
            "role": "tool",
            "tool_call_id": "cur",
            "name": "web_search",
            "content": "当前" * 2000,
        },
    ]
    total = messages_tokens(messages)
    aged = age_tool_results(messages, target_tokens=total - 1000)
    assert "原文已省略" in aged[2]["content"]      # 大且未被引用，先收
    assert "小" * 500 in aged[5]["content"]          # 有 citation，保留
    assert "当前" * 2000 in aged[8]["content"]       # 当前轮永不老化
