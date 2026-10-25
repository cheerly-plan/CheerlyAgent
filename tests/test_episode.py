from unittest.mock import patch

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.memory.episode_merge import compact_episode_texts

RELATED = [
    "最近在学习nodejs bff",
    "学习 Node.js BFF 使用 Koa 框架",
    "用 Koa 实现 BFF 数据聚合",
]


def test_compact_episode_texts_falls_back_to_small_model(monkeypatch):
    monkeypatch.setattr(settings, "memory_model", "")
    monkeypatch.setattr(settings, "small_model", "qwen-small")
    captured: dict = {}

    def fake_complete(messages, **kwargs):
        captured.update(kwargs)
        return "用户正在用 Koa 做 Node.js BFF 数据聚合"

    with patch("ai_agent.memory.parse.complete_chat", side_effect=fake_complete):
        compact_episode_texts(RELATED)
    assert captured["model"] == "qwen-small"


def test_compact_episode_texts_uses_memory_model(monkeypatch):
    monkeypatch.setattr(settings, "memory_model", "qwen-memory")
    captured: dict = {}

    def fake_complete(messages, **kwargs):
        captured.update(kwargs)
        return "用户正在用 Koa 做 Node.js BFF 数据聚合"

    with patch("ai_agent.memory.parse.complete_chat", side_effect=fake_complete):
        compact_episode_texts(RELATED)
    assert captured["model"] == "qwen-memory"


def test_compact_episode_texts_uses_small_model():
    captured: list[list[dict]] = []

    def fake_complete(messages, **kwargs):
        captured.append(messages)
        return "最近在学习 Node.js BFF，使用 Koa 框架实现数据聚合"

    with patch("ai_agent.memory.episode_merge.complete_memory_chat", side_effect=fake_complete):
        text = compact_episode_texts(RELATED)

    assert text == "最近在学习 Node.js BFF，使用 Koa 框架实现数据聚合"
    assert "；" not in text
    user = captured[0][1]["content"]
    assert "最近在学习nodejs bff" in user
    assert "Koa" in user
    assert "数据聚合" in user


def test_compact_episode_texts_falls_back_on_llm_error():
    with patch(
        "ai_agent.memory.episode_merge.complete_memory_chat",
        side_effect=LLMError("timeout"),
    ):
        text = compact_episode_texts(RELATED)
    assert "；" in text
    assert "数据聚合" in text


def test_compact_episode_texts_falls_back_on_empty_result():
    with patch("ai_agent.memory.episode_merge.complete_memory_chat", return_value=""):
        text = compact_episode_texts(RELATED)
    assert "；" in text


def test_compact_episode_texts_skips_llm_when_contained():
    with patch("ai_agent.memory.episode_merge.complete_memory_chat") as complete:
        text = compact_episode_texts(
            ["学习 Node.js BFF 使用 Koa 框架", "学习 Node.js BFF"]
        )
    complete.assert_not_called()
    assert text == "学习 Node.js BFF 使用 Koa 框架"


def test_compact_episode_texts_keeps_detailed_snapshot():
    detailed = (
        "用户正在用 Node.js + Koa 做 BFF，把多个后端接口聚合成一层给前端使用，"
        "目前已确定用 Koa 做数据聚合。"
    )
    with patch("ai_agent.memory.episode_merge.complete_memory_chat", return_value=detailed):
        assert compact_episode_texts(RELATED) == detailed


def test_compact_episode_texts_single_item_skips_llm():
    with patch("ai_agent.memory.episode_merge.complete_memory_chat") as complete:
        text = compact_episode_texts(["最近在学习 Node.js BFF"])
    complete.assert_not_called()
    assert text == "最近在学习 Node.js BFF"
