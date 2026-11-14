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


def test_thinking_off_constant_never_sends_top_level_enable_thinking():
    """部分网关把顶层 enable_thinking 限制为 True，传 False 会 400。

    所有关思考的调用点必须共用同一个常量，且不含该顶层字段。
    """
    from ai_agent.llm.client import THINKING_OFF

    assert "enable_thinking" not in THINKING_OFF
    assert THINKING_OFF["chat_template_kwargs"] == {"enable_thinking": False}
    assert THINKING_OFF["think"] is False


def test_no_module_inlines_top_level_enable_thinking():
    """回归：不允许任何模块重新内联顶层 enable_thinking。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "ai_agent"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            # 允许出现在 chat_template_kwargs 内部的写法
            if stripped.startswith('"enable_thinking"') and "chat_template_kwargs" not in stripped:
                offenders.append(f"{path.name}: {stripped}")
    assert not offenders, offenders
