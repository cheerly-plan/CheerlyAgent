from unittest.mock import patch

from ai_agent.agents.loop import _capture_long_term
from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.memory.extract import ExtractedMemory, extract_turn_memories


TURN = [
    {"role": "user", "content": "我想用 Koa 做 Node.js BFF，把多个接口聚合成一层给前端"},
    {
        "role": "assistant",
        "content": "可以。你确认用 Koa 做数据聚合层的话，建议先定路由和下游接口。",
    },
]


def test_extract_turn_memories_falls_back_to_small_model(monkeypatch):
    monkeypatch.setattr(settings, "memory_model", "")
    monkeypatch.setattr(settings, "small_model", "qwen-small")
    captured: dict = {}

    def fake_complete(messages, **kwargs):
        captured.update(kwargs)
        return '{"memories":[]}'

    with patch("ai_agent.memory.extract.complete_chat", side_effect=fake_complete):
        extract_turn_memories(TURN)
    assert captured["model"] == "qwen-small"


def test_extract_turn_memories_uses_memory_model(monkeypatch):
    monkeypatch.setattr(settings, "memory_model", "qwen-memory")
    captured: dict = {}

    def fake_complete(messages, **kwargs):
        captured.update(kwargs)
        return '{"memories":[]}'

    with patch("ai_agent.memory.extract.complete_chat", side_effect=fake_complete):
        extract_turn_memories(TURN)
    assert captured["model"] == "qwen-memory"


def test_extract_turn_memories_parses_episode():
    captured: list[list[dict]] = []

    def fake_complete(messages, **kwargs):
        captured.append(messages)
        return (
            '{"memories":[{"kind":"episode","content":'
            '"用户正在用 Node.js + Koa 做 BFF，把多个后端接口聚合成一层给前端。"}]}'
        )

    with patch("ai_agent.memory.extract.complete_chat", side_effect=fake_complete):
        items = extract_turn_memories(TURN, summary="此前在讨论后端")

    assert len(items) == 1
    assert items[0].kind == "episode"
    assert "Koa" in items[0].content
    assert "BFF" in items[0].content
    user = captured[0][1]["content"]
    assert "Koa" in user
    assert "此前在讨论后端" in user


def test_extract_turn_memories_accepts_fenced_json():
    raw = """```json
{"memories":[{"kind":"profile","content":"用户是后端工程师"}]}
```"""
    with patch("ai_agent.memory.extract.complete_chat", return_value=raw):
        items = extract_turn_memories(TURN)
    assert items == [ExtractedMemory(kind="profile", content="用户是后端工程师")]


def test_extract_turn_memories_empty_when_nothing_to_store():
    with patch(
        "ai_agent.memory.extract.complete_chat",
        return_value='{"memories":[]}',
    ):
        assert extract_turn_memories(TURN) == []


def test_extract_turn_memories_returns_empty_on_llm_error():
    with patch(
        "ai_agent.memory.extract.complete_chat",
        side_effect=LLMError("timeout"),
    ):
        assert extract_turn_memories(TURN) == []


def test_extract_turn_memories_skips_without_assistant():
    with patch("ai_agent.memory.extract.complete_chat") as complete:
        items = extract_turn_memories([{"role": "user", "content": "你好"}])
    complete.assert_not_called()
    assert items == []


def test_capture_long_term_writes_extracted_episode():
    item = ExtractedMemory(
        kind="episode",
        content="用户正在用 Node.js + Koa 做 BFF，把多个后端接口聚合成一层给前端。",
    )
    with (
        patch("ai_agent.agents.loop.store.search_episode", return_value=[]),
        patch("ai_agent.agents.loop.extract_turn_memories", return_value=[item]),
        patch("ai_agent.agents.loop.store.write_episode") as write_episode,
        patch("ai_agent.agents.loop.store.write_profile") as write_profile,
    ):
        _capture_long_term("u1", TURN, "", TURN[0]["content"])
    write_episode.assert_called_once_with(item.content, user_id="u1")
    write_profile.assert_not_called()
