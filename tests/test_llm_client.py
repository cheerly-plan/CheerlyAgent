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
