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


def test_ref_marks_are_scrubbed_from_streaming_output():
    """模型会把内部引用标记复述出来，输出侧必须刮掉。"""
    from ai_agent.agents.messages import RefScrubber

    text = "一致行动人〔ref=5875a71d:17〕；股东总数〔ref=5875a71d〕。"
    for size in (3, 8, 20, 40):
        scrub = RefScrubber()
        out = ""
        for i in range(0, len(text), size):
            out += scrub.feed(text[i : i + size])
        out += scrub.flush()
        assert "ref=" not in out
        assert "一致行动人" in out and "股东总数" in out


def test_aged_stub_keeps_refs_for_read_back():
    """老化行里要保留引用，否则模型没法 kb_read 读回原文。"""
    content = "[\u5f20\u4e09\u62a5\u544a \u00b7 \u7b2c3\u9875]\n" + "数" * 900
    messages = [
        {"role": "user", "content": "问1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "kb_search", "arguments": {"query": "q"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "kb_search", "content": content,
         "refs": ["5875a71d:17", "5875a71d:13"]},
        {"role": "user", "content": "问2"},
        {"role": "user", "content": "问3"},
    ]
    aged = age_tool_results(messages, keep_turns=2)
    stub = aged[2]["content"]
    assert 'refs=["5875a71d:17","5875a71d:13"]' in stub
    assert "原文已省略" in stub
