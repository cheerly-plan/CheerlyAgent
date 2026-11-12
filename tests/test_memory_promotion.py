"""渐进式记忆 · 第 4 步回归。

两条最重要的断言：

1. ``memory_promotion_enabled=False``（默认）时，检索 / TTL 清理 / 抽取 /
   画像删除 / ensure_schema 的 SQL 与调用逐字不变，不碰任何新表新列；
2. 候选**默认不自动升 L1 画像**，任何新链路异常都 fail-open。

其余用例覆盖证据门槛、出处门禁、否定删除、衰减/续期与后台审核。
单测不连 Postgres：DB 层用假连接池记录 SQL 与参数。
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

from ai_agent.core.config import settings
from ai_agent.memory import promotion
from ai_agent.memory.promotion import (
    Candidate,
    approve_candidate,
    decay_hit_count,
    evaluate_candidate,
    evaluate_episode,
    has_source,
    is_explicit_remember,
    is_explicit_reject,
    needs_source,
    observe_episode,
    reject_candidates,
    review_all_pending_candidates,
    review_pending_candidates,
    should_retain,
)
from ai_agent.storage.db import MemoryItem
from ai_agent.storage.store import MemoryStore, store


class _FakeResult:
    def __init__(self, rows=None, rowcount=None):
        self._rows = rows or []
        self.rowcount = len(self._rows) if rowcount is None else rowcount

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, sink, rows=None, rowcount=None):
        self._sink = sink
        self._rows = rows
        self._rowcount = rowcount

    def execute(self, sql, params=None):
        self._sink.append((sql, params))
        return _FakeResult(self._rows, self._rowcount)

    def commit(self):
        return None


class _FakePool:
    def __init__(self, sink, rows=None, rowcount=None):
        self._sink = sink
        self._rows = rows
        self._rowcount = rowcount

    @contextmanager
    def connection(self):
        yield _FakeConn(self._sink, self._rows, self._rowcount)


def _fake_store(monkeypatch, *, rows=None, rowcount=None, sink=None):
    sink = sink if sink is not None else []
    obj = MemoryStore()
    monkeypatch.setattr(obj, "_get_pool", lambda: _FakePool(sink, rows, rowcount))
    return obj, sink


def _item(text="用户偏好简洁的中文回答", mid="e1"):
    return MemoryItem(id=mid, user_id="u1", text=text, created_at="2026-01-01")


def _candidate(**kwargs):
    base = dict(
        id="c1",
        user_id="u1",
        text="用户偏好简洁的中文回答",
        facet="preference",
        source="",
        hit_count=5,
        session_seen=4,
        status="pending",
    )
    base.update(kwargs)
    return Candidate(**base)


# ---------- 纯函数：出处 / 意图门禁 ----------


def test_needs_source_only_for_verifiable_facts():
    assert needs_source("营收增长 12.5%")
    assert needs_source("市占率同比提升")
    assert not needs_source("喜欢简洁的中文回答")
    # 泛词「报告」不该把普通近况卡死在 hold
    assert not needs_source("用户在写年度报告")


def test_has_source_and_explicit_intents():
    assert has_source("据 2025 年报第 30 页")
    assert not has_source("   ")
    assert is_explicit_remember("记住，以后都用中文")
    assert not is_explicit_remember("随便聊聊")
    assert is_explicit_reject("别记这条，我说错了")
    assert not is_explicit_reject("这个不是我想问的")


# ---------- 纯函数：观察态晋升判定 ----------


def test_evaluate_episode_stays_observe_without_evidence():
    d = evaluate_episode(text="最近在学 BFF", hit_count=1, session_seen=1)
    assert (d.stage, d.reason) == ("observe", "insufficient_evidence")


def test_evaluate_episode_promotes_to_candidate_on_recurrence():
    d = evaluate_episode(text="最近在学 BFF", hit_count=5, session_seen=3)
    assert (d.stage, d.reason) == ("candidate", "recurred")


def test_evaluate_episode_explicit_remember_becomes_candidate():
    d = evaluate_episode(text="用中文回答", user_input="记住，以后都用中文")
    assert (d.stage, d.reason) == ("candidate", "explicit_remember")


def test_evaluate_episode_blocks_numbers_without_source_even_if_remembered():
    d = evaluate_episode(
        text="毛利率是 42%", user_input="记住这个", hit_count=9, session_seen=9
    )
    assert (d.stage, d.reason) == ("hold", "missing_source")


def test_evaluate_episode_allows_number_with_source():
    d = evaluate_episode(text="毛利率是 42%", source="2025 年报第 30 页",
                         hit_count=5, session_seen=3)
    assert d.stage == "candidate"


def test_evaluate_episode_reject_beats_everything():
    d = evaluate_episode(text="毛利率是 42%", source="年报", hit_count=9,
                         session_seen=9, user_input="别记这条，我记错了")
    assert (d.stage, d.reason) == ("reject", "explicit_reject")


# ---------- 纯函数：候选审核与衰减 ----------


def test_candidate_holds_on_duplicate_or_missing_source():
    assert evaluate_candidate(_candidate(), duplicate=True).reason == "duplicate"
    assert (
        evaluate_candidate(_candidate(text="占比 42%", source="")).reason
        == "missing_source"
    )
    assert (
        evaluate_candidate(_candidate(session_seen=1)).reason
        == "insufficient_evidence"
    )


def test_candidate_promotes_on_evidence_without_approval():
    """跨会话复现够数就晋升 —— 不再需要人工逐条批准。"""
    d = evaluate_candidate(_candidate())
    assert (d.stage, d.reason) == ("promote", "evidence")


def test_decay_reduces_heat_instead_of_deleting():
    assert decay_hit_count(4, 20 * 86400, 14 * 86400) == 3
    assert decay_hit_count(4, 3600, 14 * 86400) == 4
    assert decay_hit_count(0, 99 * 86400, 14 * 86400) == 0
    assert decay_hit_count(4, 99 * 86400, 0) == 4


def test_should_retain_keeps_hot_episodes():
    assert should_retain(3, 3)
    assert not should_retain(2, 3)
    assert not should_retain(9, 0)


# ---------- 编排层：observe / reject ----------


def test_observe_records_candidate_when_evidence_met(monkeypatch):
    monkeypatch.setattr(store, "touch_episode_session", lambda *a, **k: (5, 3))
    upsert = MagicMock()
    monkeypatch.setattr(store, "upsert_candidate", upsert)
    d = observe_episode(_item(), user_id="u1", chat_session_id="s1")
    assert d.stage == "candidate"
    upsert.assert_called_once()
    assert upsert.call_args.kwargs["session_seen"] == 3
    # episode 没带 facet 时按文本线索推断，不再一律 other
    assert upsert.call_args.kwargs["facet"] == "preference"


def test_observe_uses_explicit_candidate_facet(monkeypatch):
    monkeypatch.setattr(store, "touch_episode_session", lambda *a, **k: (5, 3))
    upsert = MagicMock()
    monkeypatch.setattr(store, "upsert_candidate", upsert)
    observe_episode(
        _item("在做 BFF 聚合层"),
        user_id="u1",
        chat_session_id="s1",
        facet="occupation",
    )
    assert upsert.call_args.kwargs["facet"] == "occupation"


def test_observe_stays_observe_when_evidence_weak(monkeypatch):
    monkeypatch.setattr(store, "touch_episode_session", lambda *a, **k: (1, 1))
    upsert = MagicMock()
    monkeypatch.setattr(store, "upsert_candidate", upsert)
    d = observe_episode(_item(), user_id="u1", chat_session_id="s1")
    assert d.stage == "observe"
    upsert.assert_not_called()


def test_observe_deletes_candidate_on_explicit_reject(monkeypatch):
    monkeypatch.setattr(store, "touch_episode_session", lambda *a, **k: (5, 5))
    delete = MagicMock(return_value=1)
    monkeypatch.setattr(store, "delete_candidates_matching", delete)
    d = observe_episode(
        _item(), user_id="u1", chat_session_id="s1", user_input="别记这条"
    )
    assert d.stage == "reject"
    delete.assert_called_once()


def test_observe_fails_open_on_db_error(monkeypatch):
    monkeypatch.setattr(
        store, "touch_episode_session", MagicMock(side_effect=RuntimeError("db down"))
    )
    assert observe_episode(_item(), user_id="u1", chat_session_id="s1") is None


def test_review_holds_candidates_by_default(monkeypatch):
    monkeypatch.setattr(
        store,
        "list_candidates",
        lambda *a, **k: [
            dict(
                id="c1", text="用中文回答", facet="preference", source="",
                hit_count=5, session_seen=4,
            )
        ],
    )
    mark = MagicMock()
    monkeypatch.setattr(store, "mark_candidate", mark)
    assert review_pending_candidates(user_id="u1") == 0
    mark.assert_not_called()


def test_review_promotes_when_approved_and_profile_accepts(monkeypatch):
    monkeypatch.setattr(
        store,
        "list_candidates",
        lambda *a, **k: [
            dict(
                id="c1", text="用中文回答", facet="preference", source="",
                hit_count=5, session_seen=4,
            )
        ],
    )
    mark = MagicMock(return_value=True)
    monkeypatch.setattr(store, "mark_candidate", mark)
    monkeypatch.setattr(
        "ai_agent.memory.profile.apply_profile_action", lambda *a, **k: _item("用中文回答")
    )
    assert review_pending_candidates(user_id="u1") == 1
    mark.assert_called_once_with("c1", "promoted", user_id="u1")


def test_review_does_not_mark_promoted_when_deduped(monkeypatch):
    monkeypatch.setattr(
        store,
        "list_candidates",
        lambda *a, **k: [
            dict(
                id="c1", text="用中文回答", facet="preference", source="",
                hit_count=5, session_seen=4,
            )
        ],
    )
    mark = MagicMock()
    monkeypatch.setattr(store, "mark_candidate", mark)
    monkeypatch.setattr(
        "ai_agent.memory.profile.apply_profile_action", lambda *a, **k: None
    )
    assert review_pending_candidates(user_id="u1") == 0
    mark.assert_not_called()


def _pending_row(**kwargs):
    row = dict(
        id="c1", text="用中文回答", facet="preference", source="",
        hit_count=5, session_seen=4, status="pending",
    )
    row.update(kwargs)
    return row


def test_approve_candidate_writes_profile_and_marks_promoted(monkeypatch):
    monkeypatch.setattr(store, "get_candidate", lambda *a, **k: _pending_row())
    monkeypatch.setattr(
        "ai_agent.memory.profile.apply_profile_action", lambda *a, **k: _item("用中文回答")
    )
    mark = MagicMock(return_value=True)
    monkeypatch.setattr(store, "mark_candidate", mark)
    assert approve_candidate("c1", user_id="u1") is True
    mark.assert_called_once_with("c1", "promoted", user_id="u1")


def test_approve_candidate_still_blocks_numbers_without_source(monkeypatch):
    monkeypatch.setattr(
        store, "get_candidate", lambda *a, **k: _pending_row(text="占比 42%", source="")
    )
    write = MagicMock()
    monkeypatch.setattr("ai_agent.memory.profile.apply_profile_action", write)
    assert approve_candidate("c1", user_id="u1") is False
    write.assert_not_called()


def test_approve_candidate_rejects_non_pending_and_dedup(monkeypatch):
    monkeypatch.setattr(
        store, "get_candidate", lambda *a, **k: _pending_row(status="rejected")
    )
    assert approve_candidate("c1", user_id="u1") is False
    monkeypatch.setattr(store, "get_candidate", lambda *a, **k: _pending_row())
    monkeypatch.setattr(
        "ai_agent.memory.profile.apply_profile_action", lambda *a, **k: None
    )
    mark = MagicMock()
    monkeypatch.setattr(store, "mark_candidate", mark)
    assert approve_candidate("c1", user_id="u1") is False
    mark.assert_not_called()


def test_review_all_iterates_users(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(store, "list_candidate_users", lambda **k: ["u1", "u2"])
    monkeypatch.setattr(
        promotion,
        "review_pending_candidates",
        lambda *, user_id, limit: seen.append(user_id) or 0,
    )
    assert review_all_pending_candidates() == 0
    assert seen == ["u1", "u2"]


# ---------- DB 层：默认关闭时 SQL 逐字不变 ----------


def test_search_episode_has_no_hit_sql_when_disabled(monkeypatch):
    obj, sink = _fake_store(monkeypatch, rows=[])
    monkeypatch.setattr(
        "ai_agent.memory.episode_store.embed_query", lambda q: [0.1] * 8
    )
    obj.search_episode("随便问问", user_id="u1")
    assert not any("UPDATE agent_episode" in sql for sql, _ in sink)


def test_search_episode_records_hits_when_enabled(monkeypatch):
    rows = [
        dict(id="e1", user_id="u1", text="在学 BFF", created_at="2026-01-01", score=0.9)
    ]
    obj, sink = _fake_store(monkeypatch, rows=rows)
    monkeypatch.setattr(
        "ai_agent.memory.episode_store.embed_query", lambda q: [0.1] * 8
    )
    hits = obj.search_episode("BFF", user_id="u1")
    assert [h.id for h in hits] == ["e1"]
    updates = [p for sql, p in sink if "UPDATE agent_episode" in sql]
    assert updates
    assert updates[0][2] == ["e1"]


def test_search_episode_fails_open_when_hit_recording_breaks(monkeypatch):
    rows = [
        dict(id="e1", user_id="u1", text="在学 BFF", created_at="2026-01-01", score=0.9)
    ]
    obj, _ = _fake_store(monkeypatch, rows=rows)
    monkeypatch.setattr(
        "ai_agent.memory.episode_store.embed_query", lambda q: [0.1] * 8
    )
    monkeypatch.setattr(
        obj, "record_episode_hits", MagicMock(side_effect=RuntimeError("db down"))
    )
    assert [h.id for h in obj.search_episode("BFF", user_id="u1")] == ["e1"]


def test_similar_episodes_ttl_uses_last_seen_when_enabled(monkeypatch):
    obj, sink = _fake_store(monkeypatch, rows=[])
    with obj._get_pool().connection() as conn:
        obj._similar_episodes(conn, MagicMock(), "u1", 3, 0.5)
    sql = "\n".join(s for s, _ in sink)
    assert "OR COALESCE(last_seen_at, created_at) >= now()" in sql


def test_cleanup_uses_ttl_renewal_and_decay_when_enabled(monkeypatch):
    obj, sink = _fake_store(monkeypatch, rowcount=2)
    assert obj.cleanup_expired_episodes(user_id="u1") == 2
    sql = "\n".join(s for s, _ in sink)
    assert "last_decay_at" in sql
    assert "COALESCE(last_seen_at, created_at)" in sql
    assert "DELETE FROM agent_episode WHERE created_at <" not in sql


def test_cleanup_fails_open_without_deleting_when_progressive_breaks(monkeypatch):
    obj, sink = _fake_store(monkeypatch, rowcount=2)
    monkeypatch.setattr(
        obj, "decay_and_cleanup_episodes", MagicMock(side_effect=RuntimeError("db down"))
    )
    assert obj.cleanup_expired_episodes(user_id="u1") == 0
    assert not any("DELETE FROM agent_episode" in s for s, _ in sink)


def test_touch_episode_session_dedupes_same_session_in_sql(monkeypatch):
    obj, sink = _fake_store(
        monkeypatch,
        rows=[dict(hit_count=3, session_seen=2)],
    )
    assert obj.touch_episode_session("e1", "s1", user_id="u1") == (3, 2)
    sql = "\n".join(s for s, _ in sink)
    assert "= ANY(COALESCE(sessions_seen" in sql
    assert "array_append" in sql


def test_decay_and_cleanup_issues_decay_then_delete(monkeypatch):
    obj, sink = _fake_store(monkeypatch, rowcount=1)
    n = obj.decay_and_cleanup_episodes(
        ttl_seconds=100, decay_seconds=50, min_hits=3, user_id="u1"
    )
    assert n == 1
    sql = "\n".join(s for s, _ in sink)
    assert "GREATEST(COALESCE(hit_count, 0) - 1, 0)" in sql
    assert "COALESCE(hit_count, 0) < %s" in sql
    assert sink[-1][1][:2] == (100, 3)


# ---------- schema 与接线 ----------


def test_ensure_schema_adds_columns_and_candidate_table_when_enabled(monkeypatch):
    obj, sink = _fake_store(monkeypatch)
    obj.ensure_schema()
    sql = "\n".join(s for s, _ in sink)
    assert "ADD COLUMN IF NOT EXISTS hit_count INT NOT NULL DEFAULT 0" in sql
    assert "sessions_seen TEXT[] NOT NULL DEFAULT '{}'" in sql
    assert "CREATE TABLE IF NOT EXISTS agent_memory_candidate" in sql


def test_capture_turn_forwards_session_id(monkeypatch):
    from ai_agent.agents import turn

    captured = {}

    def fake_capture(uid, messages, summary, user_input, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(turn, "capture_long_term", fake_capture)
    turn._capture_turn("u1", [], "", "问句", "session-42")
    assert captured["chat_session_id"] == "session-42"


def test_write_extracted_observes_episode_with_session_id(monkeypatch):
    from ai_agent.memory import long_term

    written = _item("在学 BFF")
    monkeypatch.setattr(store, "write_episode", lambda *a, **k: written)
    observed = MagicMock(return_value=None)
    monkeypatch.setattr(long_term, "observe_episode", observed)
    long_term._write_extracted(
        "u1",
        long_term.ExtractedMemory(kind="episode", content="在学 BFF"),
        chat_session_id="s1",
        user_input="BFF 怎么用",
    )
    observed.assert_called_once()
    assert observed.call_args.kwargs["chat_session_id"] == "s1"


def test_long_term_capture_passes_session_id_to_write(monkeypatch):
    from ai_agent.memory import long_term

    seen = {}

    def fake_write(uid, item, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(long_term, "_gate_blocks", lambda *a, **k: False)
    monkeypatch.setattr(store, "search_episode", lambda *a, **k: [])
    monkeypatch.setattr(store, "list_profile", lambda *a, **k: [])
    monkeypatch.setattr(
        long_term,
        "extract_turn_memories",
        lambda *a, **k: [long_term.ExtractedMemory(kind="episode", content="在学 BFF")],
    )
    monkeypatch.setattr(long_term, "_write_extracted", fake_write)
    long_term.capture_long_term(
        "u1",
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        "",
        "x",
        "session-7",
    )
    assert seen["chat_session_id"] == "session-7"
