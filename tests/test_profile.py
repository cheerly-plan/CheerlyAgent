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


def test_sanitize_skips_redundant_preference():
    """累积 facet：与已有条目高度相似直接 skip，防同义偏好堆积。"""
    existing = [_item("喜欢简洁回答", "preference")]
    action = ProfileAction(op="add", facet="preference", content="喜欢简洁回答。")
    cleaned = sanitize_profile_action(action, existing)
    assert cleaned.op == "skip"


def test_apply_profile_action_skips_jev_duplicate():
    """Jev 判定与已有条目同一偏好：不写入。"""
    existing = [_item("喜欢简洁回答", "preference")]
    with (
        patch("ai_agent.memory.profile.store.list_profile", return_value=existing),
        patch(
            "ai_agent.memory.profile.jev_noul_batch",
            return_value={"dup_0": 0.9},
        ) as jev,
        patch("ai_agent.memory.profile.store.write_profile") as write_profile,
    ):
        result = apply_profile_action(
            ProfileAction(op="add", facet="preference", content="希望回复简短一些"),
            user_id="a",
        )
    assert result is None
    jev.assert_called_once()
    write_profile.assert_not_called()


def test_apply_profile_action_jev_fail_open_writes():
    """Jev 不可用（未配 key/失败）：放行写入，行为与无去重一致。"""
    existing = [_item("喜欢简洁回答", "preference")]
    item = _item("希望回复简短一些", "preference")
    with (
        patch("ai_agent.memory.profile.store.list_profile", return_value=existing),
        patch(
            "ai_agent.memory.profile.jev_noul_batch", return_value=None
        ) as jev,
        patch(
            "ai_agent.memory.profile.store.write_profile", return_value=item
        ) as write_profile,
    ):
        result = apply_profile_action(
            ProfileAction(op="add", facet="preference", content="希望回复简短一些"),
            user_id="a",
        )
    assert result is item
    jev.assert_called_once()
    write_profile.assert_called_once()


def test_apply_profile_action_jev_low_prob_writes():
    """Jev 判定不是重复：正常写入。"""
    existing = [_item("喜欢简洁回答", "preference")]
    item = _item("周末不想被打扰", "preference")
    with (
        patch("ai_agent.memory.profile.store.list_profile", return_value=existing),
        patch(
            "ai_agent.memory.profile.jev_noul_batch",
            return_value={"dup_0": 0.15},
        ),
        patch(
            "ai_agent.memory.profile.store.write_profile", return_value=item
        ) as write_profile,
    ):
        result = apply_profile_action(
            ProfileAction(op="add", facet="preference", content="周末不想被打扰"),
            user_id="a",
        )
    assert result is item
    write_profile.assert_called_once()


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
    delete_facet.assert_called_once_with("locale", user_id="a", text=None)


def test_apply_profile_action_preference_adds_are_kept_separately():
    """累积 facet 的多条 add 不被合并，逐条写入。"""
    item_a = _item("喜欢简洁回答", "preference")
    item_b = _item("对历史话题感兴趣", "preference")
    raw = (
        '{"actions":['
        '{"op":"add","facet":"preference","content":"喜欢简洁回答"},'
        '{"op":"add","facet":"preference","content":"对历史话题感兴趣"}'
        "]}"
    )
    with (
        patch("ai_agent.memory.profile.store.list_profile", return_value=[]),
        patch("ai_agent.memory.profile.complete_memory_chat", return_value=raw),
        patch(
            "ai_agent.memory.profile.store.write_profile",
            side_effect=[item_a, item_b],
        ) as write_profile,
    ):
        result = apply_profile_text("我喜欢简洁的回答，也对历史感兴趣", user_id="a")
    assert write_profile.call_count == 2
    assert "喜欢简洁回答" in result and "对历史话题感兴趣" in result


def test_apply_profile_action_preference_delete_is_scoped():
    """累积 facet 的 delete 带 content，只删被否定的那一条。"""
    with (
        patch("ai_agent.memory.profile.store.list_profile", return_value=[]),
        patch(
            "ai_agent.memory.profile.store.delete_profile_facet",
            return_value=True,
        ) as delete_facet,
    ):
        apply_profile_action(
            ProfileAction(op="delete", facet="preference", content="喜欢简洁回答"),
            user_id="a",
        )
    delete_facet.assert_called_once_with(
        "preference", user_id="a", text="喜欢简洁回答"
    )
