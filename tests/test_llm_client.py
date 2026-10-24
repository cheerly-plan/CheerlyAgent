import json

from ai_agent.llm.client import _to_openai_messages


def test_tool_call_arguments_are_json_string():
    payload = _to_openai_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "arguments": {"city": "北京"},
                    }
                ],
            }
        ]
    )
    arguments = payload[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"city": "北京"}



def test_to_openai_messages_passes_through_partial_flag():
    """消息级 partial 标记透传给服务端；普通消息不带该字段。"""
    payload = _to_openai_messages(
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "春天来了", "partial": True},
        ]
    )
    assert "partial" not in payload[0]
    assert payload[1]["partial"] is True
