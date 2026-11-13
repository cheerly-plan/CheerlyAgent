"""search_history: 摘要之外的旧对话原文检索。"""

from unittest.mock import patch

from ai_agent.storage.db import user_scope
from ai_agent.tools.context import session_scope
from ai_agent.tools.misc.history import search_history


def test_search_history_formats_rows_and_requires_scope():
    rows = [
        {"seq": 3, "role": "user", "content": "合同负债多少"},
        {"seq": 4, "role": "assistant", "content": "合同负债 100 万"},
    ]
    with (
        patch(
            "ai_agent.storage.store.store.search_chat_history",
            return_value=rows,
        ) as search,
        user_scope("u1"),
        session_scope("s1"),
    ):
        out = search_history("合同负债")
    assert "[seq=3] 用户：合同负债多少" in out
    assert "[seq=4] 助手：合同负债 100 万" in out
    search.assert_called_once_with("u1", "s1", "合同负债", limit=8)


def test_search_history_handles_empty_query_and_no_hits():
    with user_scope("u1"), session_scope("s1"):
        assert "请提供" in search_history("  ")
        with patch(
            "ai_agent.storage.store.store.search_chat_history",
            return_value=[],
        ):
            assert "没有找到" in search_history("不存在")


def test_read_history_marks_anchor_and_clips():
    from ai_agent.tools.misc.history import read_history

    rows = [
        {"seq": 2, "role": "user", "content": "前一句"},
        {"seq": 3, "role": "user", "content": "合同负债多少"},
        {"seq": 4, "role": "assistant", "content": "合同负债 100 万"},
    ]
    with (
        patch(
            "ai_agent.storage.store.store.load_chat_around",
            return_value=rows,
        ) as load,
        user_scope("u1"),
        session_scope("s1"),
    ):
        out = read_history(3, before=1, after=1)
    assert "[seq=3] ← 用户：合同负债多少" in out
    assert "[seq=4] 助手：合同负债 100 万" in out
    load.assert_called_once_with("u1", "s1", 3, before=1, after=1)


def test_read_history_without_scope_or_rows():
    from ai_agent.tools.misc.history import read_history

    with user_scope("u1"), session_scope(""):
        assert "没有可读取" in read_history(1)
    with (
        patch(
            "ai_agent.storage.store.store.load_chat_around",
            return_value=[],
        ),
        user_scope("u1"),
        session_scope("s1"),
    ):
        assert "附近没有可读" in read_history(99)
