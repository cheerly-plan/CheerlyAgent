import logging
from unittest.mock import patch

from ai_agent.core.exceptions import LLMError
from ai_agent.memory.facets import format_profile_line, normalize_facet
from ai_agent.memory.profile import (
    ProfileAction,
    apply_profile_action,
    apply_profile_text,
    resolve_profile_write,
    sanitize_profile_action,
)
from ai_agent.memory.store import MemoryItem


def _item(text: str, facet: str) -> MemoryItem:
    return MemoryItem(
        id="1",
        user_id="a",
        text=text,
        created_at="t",
        facet=facet,
    )


def test_normalize_facet_aliases():
    assert normalize_facet("职业") == "occupation"
    assert normalize_facet("occupation") == "occupation"
    assert normalize_facet("unknown") == "other"


def test_sanitize_keeps_llm_facet():
    action = ProfileAction(op="add", facet="preference", content="想学习后端")
    cleaned = sanitize_profile_action(action, [])
    assert cleaned.facet == "preference"


def test_sanitize_warns_but_does_not_block_redundant_occupation(caplog):
    existing = [_item("曾是汽修工，现为水电修理工（计划中）", "occupation")]
    action = ProfileAction(
        op="update",
        facet="occupation",
        content="计划从汽修工转行做水电修理工",
    )
    with caplog.at_level(logging.WARNING):
        cleaned = sanitize_profile_action(action, existing)
    assert cleaned == action
    assert "redundant" in caplog.text


def test_sanitize_does_not_drop_richer_occupation_update():
    existing = [_item("后端", "occupation")]
    action = ProfileAction(
        op="update",
        facet="occupation",
        content="5 年后端开发工程师",
    )
    cleaned = sanitize_profile_action(action, existing)
    assert cleaned == action


def test_apply_profile_action_writes_llm_update():
    existing = [_item("后端", "occupation")]
    item = _item("5 年后端开发工程师", "occupation")
    action = ProfileAction(
        op="update",
        facet="occupation",
        content="5 年后端开发工程师",
    )
    with (
        patch("ai_agent.memory.profile.store.list_profile", return_value=existing),
        patch(
            "ai_agent.memory.profile.store.write_profile", return_value=item
        ) as write_profile,
    ):
        result = apply_profile_action(action, user_id="a")
    assert result is item
    write_profile.assert_called_once_with(
        "5 年后端开发工程师", user_id="a", facet="occupation"
    )


def test_format_profile_line_uses_chinese_label():
    assert format_profile_line(_item("用户是后端", "occupation")) == "- 职业：用户是后端"


def test_resolve_profile_write_parses_update():
    existing = [_item("我是画家", "occupation")]
    with patch(
        "ai_agent.memory.profile.complete_memory_chat",
        return_value='{"op":"update","facet":"occupation","content":"用户是后端工程师"}',
    ):
        actions = resolve_profile_write("我现在做后端", existing)
    assert actions == [
        ProfileAction(op="update", facet="occupation", content="用户是后端工程师")
    ]


def test_resolve_profile_write_splits_mixed_other_on_career_change():
    existing = [_item("用户名为哈里，是一名画家", "other")]
    raw = (
        '{"actions":['
        '{"op":"update","facet":"occupation","content":"曾是画家，现为前端开发工程师"},'
        '{"op":"add","facet":"name","content":"哈里"},'
        '{"op":"delete","facet":"other"}'
        "]}"
    )
    with patch("ai_agent.memory.profile.complete_memory_chat", return_value=raw):
        actions = resolve_profile_write("用户职业是前端开发工程师", existing)
    assert [action.facet for action in actions] == ["occupation", "name", "other"]
    assert actions[0].content == "曾是画家，现为前端开发工程师"
    assert actions[2].op == "delete"


def test_resolve_profile_write_warns_on_facet_op_conflict(caplog):
    raw = (
        '{"actions":['
        '{"op":"update","facet":"locale","content":"上海"},'
        '{"op":"delete","facet":"locale"}'
        "]}"
    )
    with (
        caplog.at_level(logging.WARNING),
        patch("ai_agent.memory.profile.complete_memory_chat", return_value=raw),
    ):
        actions = resolve_profile_write("我不在上海了", [])
    assert len(actions) == 1
    assert actions[0].op == "delete"
    assert actions[0].facet == "locale"
    assert "conflict" in caplog.text


def test_resolve_profile_write_falls_back_on_llm_error():
    with patch(
        "ai_agent.memory.profile.complete_memory_chat",
        side_effect=LLMError("timeout"),
    ):
        actions = resolve_profile_write("喜欢 pytest", [])
    assert actions == [ProfileAction(op="add", facet="other", content="喜欢 pytest")]


def test_apply_profile_text_skips():
    with (
        patch("ai_agent.memory.profile.store.list_profile", return_value=[]),
        patch(
            "ai_agent.memory.profile.resolve_profile_write",
            return_value=[ProfileAction(op="skip")],
        ),
        patch("ai_agent.memory.profile.store.write_profile") as write_profile,
    ):
        result = apply_profile_text("喜欢 pytest", user_id="a")
    assert result == "用户画像无需更新"
    write_profile.assert_not_called()


def test_apply_profile_text_writes_with_facet():
    item = _item("用户是后端工程师", "occupation")
    with (
        patch("ai_agent.memory.profile.store.list_profile", return_value=[]),
        patch(
            "ai_agent.memory.profile.resolve_profile_write",
            return_value=[
                ProfileAction(
                    op="update", facet="occupation", content="用户是后端工程师"
                )
            ],
        ),
        patch(
            "ai_agent.memory.profile.store.write_profile", return_value=item
        ) as write_profile,
    ):
        result = apply_profile_text("我是后端工程师", user_id="a")
    assert "用户是后端工程师" in result
    write_profile.assert_called_once_with(
        "用户是后端工程师", user_id="a", facet="occupation"
    )


def test_apply_profile_text_deletes_facet():
    with (
        patch("ai_agent.memory.profile.store.list_profile", return_value=[]),
        patch(
            "ai_agent.memory.profile.resolve_profile_write",
            return_value=[ProfileAction(op="delete", facet="locale")],
        ),
        patch(
            "ai_agent.memory.profile.store.delete_profile_facet", return_value=True
        ) as delete_facet,
    ):
        result = apply_profile_text("我不在北京了", user_id="a")
    assert "删除" in result
    delete_facet.assert_called_once_with("locale", user_id="a")
