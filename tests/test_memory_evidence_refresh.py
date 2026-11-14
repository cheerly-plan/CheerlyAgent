"""回归：自动召回的跨会话证据闭环 + 画像载入上限的诚实文案。

两条底线：
1. 不传 chat_session_id / 画像未触及 20 条上限时，调用形式与输出逐字不变；
2. 新链路任何异常都 fail-open，不影响召回与工具返回。
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from ai_agent.agents import loop
from ai_agent.core.config import settings
from ai_agent.memory import promotion
from ai_agent.memory.progressive import build_memory_index
from ai_agent.storage.db import MemoryItem
from ai_agent.storage.store import MemoryStore, store
from ai_agent.tools.memory.memory import expand_memory


def _profiles(n: int) -> list[MemoryItem]:
    return [
        MemoryItem(
            id=f"p{i}",
            user_id="u1",
            text=f"画像条目 {i}",
            facet="preference",
            created_at="2026-01-01",
        )
        for i in range(n)
    ]


def _episodes(n: int = 2) -> list[MemoryItem]:
    return [
        MemoryItem(
            id=f"e{i}",
            user_id="u1",
            text=f"情景 {i}",
            created_at="2026-01-01",
            score=0.9,
        )
        for i in range(n)
    ]


class _FakeResult:
    def __init__(self, rowcount: int = 0):
        self.rowcount = rowcount

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _FakeConn:
    def __init__(self, sink: list, rowcount: int = 0):
        self._sink = sink
        self._rowcount = rowcount

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self._sink.append((sql, params))
        return _FakeResult(self._rowcount)

    def commit(self):
        return None


class _FakePool:
    def __init__(self, sink: list, rowcount: int = 0):
        self._sink = sink
        self._rowcount = rowcount

    @contextmanager
    def connection(self):
        yield _FakeConn(self._sink, self._rowcount)


def _fake_store(monkeypatch, *, rowcount: int = 0):
    sink: list = []
    obj = MemoryStore()
    monkeypatch.setattr(obj, "_get_pool", lambda: _FakePool(sink, rowcount))
    return obj, sink


# ---------- 1. 自动召回登记会话证据 ----------


def _search_store(monkeypatch, hits):
    obj, _ = _fake_store(monkeypatch)
    monkeypatch.setattr(
        "ai_agent.memory.episode_store.embed_query", lambda q: [0.1] * 8
    )
    monkeypatch.setattr(obj, "_similar_episodes", lambda *a, **k: hits)
    record = MagicMock()
    touch = MagicMock()
    monkeypatch.setattr(obj, "record_episode_hits", record)
    monkeypatch.setattr(obj, "touch_episode_session", touch)
    return obj, record, touch


def test_search_without_session_keeps_legacy_path(monkeypatch):
    """不传会话 id：只记热度，不碰会话证据，调用形式与改动前一致。"""
    obj, record, touch = _search_store(monkeypatch, _episodes())
    hits = obj.search_episode("在学 BFF", user_id="u1")

    assert len(hits) == 2
    record.assert_called_once_with(["e0", "e1"], user_id="u1")
    touch.assert_not_called()


def test_search_registers_session_evidence_per_hit(monkeypatch):
    obj, record, touch = _search_store(monkeypatch, _episodes())
    obj.search_episode("在学 BFF", user_id="u1", chat_session_id="s1")

    record.assert_called_once_with(["e0", "e1"], user_id="u1")
    assert touch.call_count == 2
    assert [c.args for c in touch.call_args_list] == [
        ("e0", "s1"),
        ("e1", "s1"),
    ]
    assert all(c.kwargs == {"user_id": "u1"} for c in touch.call_args_list)


def test_search_session_failure_does_not_break_recall(monkeypatch):
    """证据登记失败只记日志，召回结果必须照常返回。"""
    obj, _record, touch = _search_store(monkeypatch, _episodes())
    touch.side_effect = RuntimeError("db down")
    hits = obj.search_episode("在学 BFF", user_id="u1", chat_session_id="s1")
    assert len(hits) == 2


# ---------- 2. 候选证据快照回填 ----------


def test_refresh_candidate_evidence_sql_is_monotonic_and_scoped(monkeypatch):
    obj, sink = _fake_store(monkeypatch, rowcount=3)
    assert obj.refresh_candidate_evidence("u1") == 3

    sql, params = sink[0]
    assert "UPDATE agent_memory_candidate" in sql
    assert "GREATEST" in sql
    assert "status = 'pending'" in sql
    assert "MAX(cardinality(" in sql
    assert params == ("u1", "u1")


def test_refresh_candidate_evidence_uses_scope_when_no_user(monkeypatch):
    obj, sink = _fake_store(monkeypatch)
    monkeypatch.setattr(
        "ai_agent.memory.promotion_store.current_user_id", lambda: "ctx-user"
    )
    obj.refresh_candidate_evidence()
    assert sink[0][1] == ("ctx-user", "ctx-user")


def test_refresh_all_continues_when_one_user_fails(monkeypatch):
    monkeypatch.setattr(
        store, "list_candidate_users", lambda **k: ["u1", "u2", "u3"]
    )
    refresh = MagicMock(side_effect=[2, RuntimeError("db down"), 1])
    monkeypatch.setattr(store, "refresh_candidate_evidence", refresh)

    assert promotion.refresh_all_candidate_evidence() == 3
    assert [c.args for c in refresh.call_args_list] == [("u1",), ("u2",), ("u3",)]


def test_refresh_all_fails_open_on_user_list_error(monkeypatch):
    monkeypatch.setattr(
        store,
        "list_candidate_users",
        MagicMock(side_effect=RuntimeError("db down")),
    )
    assert promotion.refresh_all_candidate_evidence() == 0


# ---------- 3. 会话 id 透传（默认调用形式逐字不变） ----------


def test_recall_block_omits_new_kwarg_by_default():
    with patch("ai_agent.agents.loop.store.search_episode", return_value=[]) as spy:
        assert loop._recall_block("u", "研报约定") == ""
    spy.assert_called_once_with(
        "研报约定",
        k=max(settings.memory_auto_recall_top_k * 3, 3),
        user_id="u",
    )


def test_recall_block_passes_session_when_present():
    with patch("ai_agent.agents.loop.store.search_episode", return_value=[]) as spy:
        loop._recall_block("u", "研报约定", "s1")
    assert spy.call_args.kwargs["chat_session_id"] == "s1"
    assert spy.call_args.kwargs["user_id"] == "u"


def test_system_prompt_forwards_session_to_recall():
    with (
        patch("ai_agent.agents.loop.skill_catalog", return_value=""),
        patch("ai_agent.agents.loop._profile_prompt", return_value=""),
        patch("ai_agent.agents.loop._today_line", return_value=""),
        patch("ai_agent.agents.loop._kb_capability_block", return_value=""),
        patch("ai_agent.agents.loop._recall_block", return_value="") as spy,
    ):
        loop._system_prompt(
            "u",
            use_kb=False,
            use_memory=True,
            user_input="研报约定",
            chat_session_id="s1",
        )
    assert spy.call_args.args == ("u", "研报约定", "s1")


# ---------- 4. 画像上限的诚实文案（默认逐字不变） ----------


def test_index_without_total_keeps_legacy_note():
    items = _profiles(20)
    block, _ = build_memory_index(
        items, user_input="画像", top_k=3, max_chars=4000, expand_hint=True
    )
    assert "画像共 20 条" in block
    assert "需要完整条目时调用 expand_memory(query)。" in block
    assert "本次只载入" not in block


def test_index_with_total_equal_items_keeps_legacy_note():
    items = _profiles(20)
    block, _ = build_memory_index(
        items,
        user_input="画像",
        top_k=3,
        max_chars=4000,
        expand_hint=True,
        total=20,
    )
    assert "画像共 20 条" in block
    assert "本次只载入" not in block


def test_index_honest_when_profile_truncated():
    items = _profiles(20)
    # 用默认 800 字符预算：新说明行不能被 clip_text 的中段省略吃掉
    block, _ = build_memory_index(
        items, user_input="画像", top_k=3, expand_hint=True, total=37
    )
    assert len(block) <= 800
    assert "画像共 37 条" in block
    assert "本次只载入 20 条" in block
    assert "单次最多返回 20 条" in block
    assert "expand_memory(query)" in block
    assert "需要完整条目时" not in block


def test_profile_prompt_counts_only_at_limit(monkeypatch):
    count = MagicMock(return_value=37)
    index = MagicMock(return_value=("BLOCK", MagicMock()))
    with (
        patch("ai_agent.agents.loop.store.list_profile", return_value=_profiles(20)),
        patch("ai_agent.agents.loop.store.count_active_profile", count),
        patch("ai_agent.agents.loop.build_memory_index", index),
    ):
        assert loop._profile_prompt("u1", user_input="画像") == "BLOCK"
    count.assert_called_once_with("u1")
    assert index.call_args.kwargs["total"] == 37


def test_profile_prompt_skips_count_under_limit(monkeypatch):
    count = MagicMock(return_value=5)
    index = MagicMock(return_value=("BLOCK", MagicMock()))
    with (
        patch("ai_agent.agents.loop.store.list_profile", return_value=_profiles(5)),
        patch("ai_agent.agents.loop.store.count_active_profile", count),
        patch("ai_agent.agents.loop.build_memory_index", index),
    ):
        loop._profile_prompt("u1", user_input="画像")
    count.assert_not_called()
    assert index.call_args.kwargs["total"] is None


def test_profile_prompt_fails_open_on_count_error(monkeypatch):
    index = MagicMock(return_value=("BLOCK", MagicMock()))
    with (
        patch("ai_agent.agents.loop.store.list_profile", return_value=_profiles(20)),
        patch(
            "ai_agent.agents.loop.store.count_active_profile",
            MagicMock(side_effect=RuntimeError("db down")),
        ),
        patch("ai_agent.agents.loop.build_memory_index", index),
    ):
        assert loop._profile_prompt("u1", user_input="画像") == "BLOCK"
    assert index.call_args.kwargs["total"] is None


def test_expand_memory_marks_limit_when_truncated():
    with (
        patch(
            "ai_agent.tools.memory.memory.store.list_profile",
            return_value=_profiles(20),
        ),
        patch(
            "ai_agent.tools.memory.memory.store.count_active_profile",
            return_value=37,
        ),
    ):
        out = expand_memory("画像", kind="profile")
    assert "完整用户画像条目" not in out
    assert "库中共 37 条" in out
    assert "本次返回 20 条" in out
    assert "单次上限 20 条" in out


def test_expand_memory_keeps_full_label_under_limit():
    count = MagicMock(return_value=5)
    with (
        patch(
            "ai_agent.tools.memory.memory.store.list_profile",
            return_value=_profiles(5),
        ),
        patch("ai_agent.tools.memory.memory.store.count_active_profile", count),
    ):
        out = expand_memory("画像", kind="profile")
    assert "完整用户画像条目" in out
    count.assert_not_called()
