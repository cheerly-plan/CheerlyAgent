import json

from ai_agent.core.config import settings
from ai_agent.llm import client as llm_client
from ai_agent.llm.client import _to_openai_messages, lightweight_llm


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


def test_lightweight_llm_uses_local_gateway(monkeypatch):
    """SMALL_MODEL 打到 LOCAL_BASE_URL，不跟主对话走线上。"""
    monkeypatch.setattr(settings, "small_model", "qwen3:4b")
    monkeypatch.setattr(settings, "model", "qwen3.7-plus")
    monkeypatch.setattr(settings, "local_base_url", "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(settings, "local_api_key", "ollama")
    monkeypatch.setattr(settings, "base_url", "https://cloud.example/v1")
    monkeypatch.setattr(settings, "api_key", "sk-cloud")

    model, chat = lightweight_llm("")

    assert model == "qwen3:4b"
    assert chat is not llm_client._client
    assert "127.0.0.1:11434" in str(chat.base_url)


def test_lightweight_llm_override_stays_on_main_gateway(monkeypatch):
    """显式指定了别的模型名时，仍走主网关。"""
    monkeypatch.setattr(settings, "small_model", "qwen3:4b")
    monkeypatch.setattr(settings, "local_base_url", "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(settings, "local_api_key", "ollama")

    model, chat = lightweight_llm("qwen3.7-plus")

    assert model == "qwen3.7-plus"
    assert chat is llm_client._client


def test_lightweight_llm_without_small_model_uses_main(monkeypatch):
    monkeypatch.setattr(settings, "small_model", "")
    monkeypatch.setattr(settings, "model", "qwen3.7-plus")
    monkeypatch.setattr(settings, "local_base_url", "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(settings, "local_api_key", "ollama")

    model, chat = lightweight_llm("")

    assert model == "qwen3.7-plus"
    assert chat is llm_client._client
