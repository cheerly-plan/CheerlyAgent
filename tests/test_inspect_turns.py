from ai_agent.api.inspect_turns import turns_from_messages


def test_splits_each_user_question_and_keeps_process_order():
    turns = turns_from_messages(
        [
            {"role": "user", "content": "营收多少"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "kb_search",
                        "arguments": {"query": "营收"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "kb_search",
                "tool_call_id": "c1",
                "content": "12亿",
            },
            {"role": "assistant", "content": "营收是 12 亿"},
            {"role": "user", "content": "再看毛利率"},
            {"role": "assistant", "content": "毛利率是 20%"},
        ]
    )
    assert [turn.question for turn in turns] == ["营收多少", "再看毛利率"]
    assert [step.kind for step in turns[0].steps] == ["tool", "tool_result", "llm"]
    assert turns[0].steps[0].name == "kb_search"
    assert turns[0].steps[0].arguments == {"query": "营收"}
    assert turns[0].steps[1].name == "kb_search"
    assert turns[0].steps[1].tool_call_id == "c1"
    assert turns[0].steps[1].arguments == {"query": "营收"}
    assert turns[0].steps[1].content == "12亿"
    assert turns[0].steps[2].content == "营收是 12 亿"
    assert [step.kind for step in turns[1].steps] == ["llm"]


def test_string_arguments_are_parsed():
    turns = turns_from_messages(
        [
            {"role": "user", "content": "天气"},
            {
                "role": "assistant",
                "content": "我先查一下",
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "weather",
                        "arguments": '{"city": "上海"}',
                    }
                ],
            },
        ]
    )
    assert [step.kind for step in turns[0].steps] == ["llm", "tool"]
    assert turns[0].steps[1].arguments == {"city": "上海"}


def test_tool_result_keeps_the_matching_call_arguments():
    turns = turns_from_messages(
        [
            {"role": "user", "content": "对比两家"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "name": "kb_search", "arguments": {"query": "营收"}},
                    {"id": "b", "name": "kb_search", "arguments": {"query": "毛利率"}},
                ],
            },
            {"role": "tool", "tool_call_id": "b", "content": "20%"},
            {"role": "tool", "name": "kb_search", "tool_call_id": "a", "content": "12亿"},
        ]
    )
    results = {
        step.tool_call_id: step
        for step in turns[0].steps
        if step.kind == "tool_result"
    }
    assert results["b"].name == "kb_search"
    assert results["b"].arguments == {"query": "毛利率"}
    assert results["a"].arguments == {"query": "营收"}
