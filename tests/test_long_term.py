import logging
from unittest.mock import patch

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.memory.long_term import (
    ExtractedMemory,
    capture_long_term,
    extract_turn_memories,
)


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

    with patch("ai_agent.memory.parse.complete_chat", side_effect=fake_complete):
        extract_turn_memories(TURN)
    assert captured["model"] == "qwen-small"


def test_extract_turn_memories_uses_memory_model(monkeypatch):
    monkeypatch.setattr(settings, "memory_model", "qwen-memory")
    captured: dict = {}

    def fake_complete(messages, **kwargs):
        captured.update(kwargs)
        return '{"memories":[]}'

    with patch("ai_agent.memory.parse.complete_chat", side_effect=fake_complete):
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

    with patch("ai_agent.memory.long_term.complete_memory_chat", side_effect=fake_complete):
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
    with patch("ai_agent.memory.long_term.complete_memory_chat", return_value=raw):
        items = extract_turn_memories(TURN)
    assert items == [ExtractedMemory(kind="profile", content="用户是后端工程师")]


def test_extract_turn_memories_parses_profile_update():
    raw = (
        '{"memories":[{"kind":"profile","op":"update","facet":"occupation",'
        '"content":"用户是后端工程师"}]}'
    )
    with patch("ai_agent.memory.long_term.complete_memory_chat", return_value=raw):
        items = extract_turn_memories(TURN)
    assert items == [
        ExtractedMemory(
            kind="profile",
            content="用户是后端工程师",
            op="update",
            facet="occupation",
        )
    ]


def test_extract_turn_memories_coalesces_same_facet():
    raw = (
        '{"memories":['
        '{"kind":"profile","op":"add","facet":"occupation","content":"画家"},'
        '{"kind":"profile","op":"update","facet":"occupation",'
        '"content":"曾是画家，现为前端开发工程师"}'
        "]}"
    )
    with patch("ai_agent.memory.long_term.complete_memory_chat", return_value=raw):
        items = extract_turn_memories(TURN)
    assert len(items) == 1
    assert items[0].content == "曾是画家，现为前端开发工程师"


def test_extract_turn_memories_warns_on_facet_op_conflict(caplog):
    raw = (
        '{"memories":['
        '{"kind":"profile","op":"update","facet":"locale","content":"上海"},'
        '{"kind":"profile","op":"delete","facet":"locale"}'
        "]}"
    )
    with (
        caplog.at_level(logging.WARNING),
        patch("ai_agent.memory.long_term.complete_memory_chat", return_value=raw),
    ):
        items = extract_turn_memories(TURN)
    assert items == [
        ExtractedMemory(kind="profile", content="", op="delete", facet="locale")
    ]
    assert "conflict" in caplog.text


def test_extract_turn_memories_parses_profile_delete():
    raw = '{"memories":[{"kind":"profile","op":"delete","facet":"locale"}]}'
    with patch("ai_agent.memory.long_term.complete_memory_chat", return_value=raw):
        items = extract_turn_memories(TURN)
    assert items == [
        ExtractedMemory(kind="profile", content="", op="delete", facet="locale")
    ]


def test_extract_turn_memories_includes_existing_profile():
    captured: list[str] = []

    def fake_complete(messages, **kwargs):
        captured.append(messages[1]["content"])
        return '{"memories":[]}'

    with patch("ai_agent.memory.long_term.complete_memory_chat", side_effect=fake_complete):
        extract_turn_memories(
            TURN, existing_profile=["- 职业：用户是后端工程师"]
        )
    assert "已有用户画像" in captured[0]
    assert "用户是后端工程师" in captured[0]


def test_extract_turn_memories_empty_when_nothing_to_store():
    with patch(
        "ai_agent.memory.long_term.complete_memory_chat",
        return_value='{"memories":[]}',
    ):
        assert extract_turn_memories(TURN) == []


def test_extract_turn_memories_returns_empty_on_llm_error():
    with patch(
        "ai_agent.memory.long_term.complete_memory_chat",
        side_effect=LLMError("timeout"),
    ):
        assert extract_turn_memories(TURN) == []


def test_extract_turn_memories_skips_without_assistant():
    with patch("ai_agent.memory.long_term.complete_memory_chat") as complete:
        items = extract_turn_memories([{"role": "user", "content": "你好"}])
    complete.assert_not_called()
    assert items == []


def test_capture_long_term_writes_extracted_episode():
    item = ExtractedMemory(
        kind="episode",
        content="用户正在用 Node.js + Koa 做 BFF，把多个后端接口聚合成一层给前端。",
    )
    with (
        patch("ai_agent.memory.long_term.store.search_episode", return_value=[]),
        patch("ai_agent.memory.long_term.store.list_profile", return_value=[]),
        patch("ai_agent.memory.long_term.extract_turn_memories", return_value=[item]),
        patch("ai_agent.memory.long_term.store.write_episode") as write_episode,
        patch("ai_agent.memory.long_term.apply_profile_action") as apply_profile,
    ):
        capture_long_term("u1", TURN, "", TURN[0]["content"])
    write_episode.assert_called_once_with(item.content, user_id="u1")
    apply_profile.assert_not_called()


def test_capture_long_term_applies_profile_update():
    item = ExtractedMemory(
        kind="profile",
        content="用户是后端工程师",
        op="update",
        facet="occupation",
    )
    with (
        patch("ai_agent.memory.long_term.store.search_episode", return_value=[]),
        patch("ai_agent.memory.long_term.store.list_profile", return_value=[]),
        patch("ai_agent.memory.long_term.extract_turn_memories", return_value=[item]),
        patch("ai_agent.memory.long_term.store.write_episode") as write_episode,
        patch("ai_agent.memory.long_term.apply_profile_action") as apply_profile,
    ):
        capture_long_term("u1", TURN, "", "我是后端工程师")
    write_episode.assert_not_called()
    apply_profile.assert_called_once()
    action = apply_profile.call_args.args[0]
    assert action.op == "update"
    assert action.facet == "occupation"
    assert action.content == "用户是后端工程师"
