from unittest.mock import patch

from ai_agent.core.exceptions import LLMError
from ai_agent.memory.summarize import summarize_session


def test_summarize_session_uses_previous_and_overflow():
    captured: list[list[dict]] = []

    def fake_complete(messages, **kwargs):
        captured.append(messages)
        return "用户正在规划旅行"

    with patch("ai_agent.memory.summarize.complete_chat", side_effect=fake_complete):
        text = summarize_session(
            "用户叫小明",
            [
                {"role": "user", "content": "想去杭州"},
                {"role": "assistant", "content": "可以走周末两日"},
            ],
        )

    assert text == "用户正在规划旅行"
    assert "已有摘要" in captured[0][1]["content"]
    assert "想去杭州" in captured[0][1]["content"]


def test_summarize_session_falls_back_on_llm_error():
    with patch(
        "ai_agent.memory.summarize.complete_chat",
        side_effect=LLMError("timeout"),
    ):
        text = summarize_session(
            "旧摘要",
            [{"role": "user", "content": "我叫小明"}, {"role": "assistant", "content": "你好"}],
        )
    assert "旧摘要" in text
    assert "我叫小明" in text
