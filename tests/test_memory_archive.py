"""渐进式记忆 · 第 5 步回归：候选人工审核入口 + L4 蒸馏归档。

四条硬约束：

1. 默认全关时候选/归档端点一律 409、零 DB 访问，工具列表与 prompt 不变；
2. 普通用户强制只看/审自己的记忆，管理员才能指定 user_id；
3. 含数字/量化事实的候选**必须**由人工填 source 才能批准，服务端绝不
   用 LLM 补出处；人工批准仍过 evaluate_candidate + 画像去重门禁；
4. 归档只是副本：不删 episode、不进 system、幂等、异常 fail-open。

单测不连 Postgres：路由直接调用（显式传 user，绕开 Depends），DB 层用
假连接池记录 SQL 与参数。
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from ai_agent.api.routers import memory as memory_router
from ai_agent.api.schemas import MemoryApproveRequest, MemoryRejectRequest
from ai_agent.core.config import settings
from ai_agent.memory import archive, promotion
from ai_agent.memory.archive import (
    archive_enabled,
    distill_all_users,
    distill_memory,
    search_archive,
)
from ai_agent.memory.promotion import approve_candidate, reject_candidate
from ai_agent.storage.db import UserRecord
from ai_agent.storage.store import MemoryStore, store
from ai_agent.tools.memory.memory import expand_memory


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


def _user(username="alice", uid="u1"):
    return UserRecord(user_id=uid, username=username, display_name=username)


def _pending_row(**kwargs):
    row = dict(
        id="c1",
        user_id="u1",
        text="用中文回答",
        facet="preference",
        source="",
        hit_count=5,
        session_seen=4,
        status="pending",
    )
    row.update(kwargs)
    return row


def _enable_promotion(monkeypatch):
    """渐进式记忆已无开关，恒开；保留这个空实现让既有用例的调用点不动。"""


def _enable_archive(monkeypatch):
    _enable_promotion(monkeypatch)


# ---------- 审核端点：默认关闭 / 权限隔离 ----------


def test_candidate_users_is_admin_only(monkeypatch):
    _enable_promotion(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        memory_router.list_candidate_users(user=_user())
    assert exc.value.status_code == 403

    users = MagicMock(return_value=["u1", "u2"])
    monkeypatch.setattr(store, "list_candidate_users", users)
    with patch("ai_agent.api.routers.memory._is_admin", return_value=True):
        out = memory_router.list_candidate_users(user=_user("admin"))
    assert out == {"total": 2, "user_ids": ["u1", "u2"]}


def test_normal_user_cannot_request_other_users_candidates(monkeypatch):
    _enable_promotion(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        memory_router.list_candidates(
            status="pending", limit=20, user_id="u2", user=_user()
        )
    assert exc.value.status_code == 403


def test_normal_user_is_scoped_to_own_user_id(monkeypatch):
    _enable_promotion(monkeypatch)
    seen = {}
    monkeypatch.setattr(
        store,
        "list_candidates",
        lambda uid, **kw: seen.update(uid=uid, kw=kw) or [_pending_row()],
    )
    out = memory_router.list_candidates(
        status="pending", limit=20, user_id="u1", user=_user()
    )
    assert seen["uid"] == "u1"
    assert out["items"][0]["requires_source"] is False


def test_admin_can_scope_to_another_user(monkeypatch):
    _enable_promotion(monkeypatch)
    seen = {}
    monkeypatch.setattr(
        store, "list_candidates", lambda uid, **kw: seen.update(uid=uid) or []
    )
    with patch("ai_agent.api.routers.memory._is_admin", return_value=True):
        memory_router.list_candidates(
            status="pending", limit=20, user_id="u2", user=_user("admin")
        )
    assert seen["uid"] == "u2"


def test_list_marks_candidates_that_need_manual_source(monkeypatch):
    _enable_promotion(monkeypatch)
    rows = [_pending_row(), _pending_row(id="c2", text="毛利率 42%")]
    monkeypatch.setattr(store, "list_candidates", lambda uid, **kw: rows)
    out = memory_router.list_candidates(
        status="pending", limit=20, user_id=None, user=_user()
    )
    assert [item["requires_source"] for item in out["items"]] == [False, True]


# ---------- 审核端点：批准 / 驳回 ----------


def test_approve_blocks_numbered_fact_without_source(monkeypatch):
    _enable_promotion(monkeypatch)
    monkeypatch.setattr(
        store,
        "get_candidate",
        lambda *a, **k: _pending_row(text="毛利率 42%", source=""),
    )
    write = MagicMock()
    monkeypatch.setattr("ai_agent.memory.profile.apply_profile_action", write)
    with pytest.raises(HTTPException) as exc:
        memory_router.approve_candidate(
            "c1", MemoryApproveRequest(), user_id=None, user=_user()
        )
    assert exc.value.status_code == 409
    assert "出处" in exc.value.detail
    write.assert_not_called()


def test_approve_accepts_human_supplied_source(monkeypatch):
    _enable_promotion(monkeypatch)
    monkeypatch.setattr(
        store,
        "get_candidate",
        lambda *a, **k: _pending_row(text="毛利率 42%", source=""),
    )
    updated = MagicMock(
        return_value=_pending_row(text="毛利率 42%", source="2026Q3 年报第 12 页")
    )
    monkeypatch.setattr(store, "update_candidate_source", updated)
    monkeypatch.setattr(
        "ai_agent.memory.profile.apply_profile_action",
        lambda *a, **k: MagicMock(text="毛利率 42%"),
    )
    mark = MagicMock(return_value=True)
    monkeypatch.setattr(store, "mark_candidate", mark)

    out = memory_router.approve_candidate(
        "c1",
        MemoryApproveRequest(source="2026Q3 年报第 12 页", note="已核对原文"),
        user_id=None,
        user=_user(),
    )
    assert out["status"] == "promoted"
    assert out["reviewed_by"] == "alice"
    updated.assert_called_once_with(
        "c1", "2026Q3 年报第 12 页", user_id="u1"
    )
    assert mark.call_args.kwargs["reviewed_by"] == "alice"
    assert mark.call_args.kwargs["note"] == "已核对原文"


def test_approve_uses_existing_source_when_request_omits_it(monkeypatch):
    _enable_promotion(monkeypatch)
    monkeypatch.setattr(
        store,
        "get_candidate",
        lambda *a, **k: _pending_row(text="占比 42%", source="台账 F-001"),
    )
    monkeypatch.setattr(store, "update_candidate_source", MagicMock())
    monkeypatch.setattr(
        "ai_agent.memory.profile.apply_profile_action",
        lambda *a, **k: MagicMock(text="占比 42%"),
    )
    monkeypatch.setattr(store, "mark_candidate", MagicMock(return_value=True))
    out = memory_router.approve_candidate(
        "c1", MemoryApproveRequest(), user_id=None, user=_user()
    )
    assert out["source"] == "台账 F-001"


def test_approve_returns_404_for_non_pending(monkeypatch):
    _enable_promotion(monkeypatch)
    monkeypatch.setattr(
        store, "get_candidate", lambda *a, **k: _pending_row(status="promoted")
    )
    with pytest.raises(HTTPException) as exc:
        memory_router.approve_candidate(
            "c1", MemoryApproveRequest(), user_id=None, user=_user()
        )
    assert exc.value.status_code == 404


def test_approve_returns_409_when_profile_dedup_blocks(monkeypatch):
    _enable_promotion(monkeypatch)
    monkeypatch.setattr(store, "get_candidate", lambda *a, **k: _pending_row())
    monkeypatch.setattr(
        "ai_agent.memory.profile.apply_profile_action", lambda *a, **k: None
    )
    monkeypatch.setattr(store, "mark_candidate", MagicMock())
    with pytest.raises(HTTPException) as exc:
        memory_router.approve_candidate(
            "c1", MemoryApproveRequest(), user_id=None, user=_user()
        )
    assert exc.value.status_code == 409
    assert "去重" in exc.value.detail


def test_reject_marks_rejected_with_reviewer(monkeypatch):
    _enable_promotion(monkeypatch)
    monkeypatch.setattr(store, "get_candidate", lambda *a, **k: _pending_row())
    mark = MagicMock(return_value=True)
    monkeypatch.setattr(store, "mark_candidate", mark)
    out = memory_router.reject_candidate(
        "c1", MemoryRejectRequest(note="不是长期偏好"), user_id=None, user=_user()
    )
    assert out["status"] == "rejected"
    assert mark.call_args.args[:2] == ("c1", "rejected")
    assert mark.call_args.kwargs["note"] == "不是长期偏好"


# ---------- 编排层：approve / reject ----------


def test_approve_candidate_without_source_keeps_step4_call_shape(monkeypatch):
    """第 4 步的存量断言：默认调用 mark_candidate(id, status, user_id=...)。"""
    _enable_promotion(monkeypatch)
    monkeypatch.setattr(store, "get_candidate", lambda *a, **k: _pending_row())
    monkeypatch.setattr(
        "ai_agent.memory.profile.apply_profile_action",
        lambda *a, **k: MagicMock(text="用中文回答"),
    )
    mark = MagicMock(return_value=True)
    monkeypatch.setattr(store, "mark_candidate", mark)
    assert approve_candidate("c1", user_id="u1") is True
    mark.assert_called_once_with("c1", "promoted", user_id="u1")


def test_approve_candidate_routes_source_to_store(monkeypatch):
    _enable_promotion(monkeypatch)
    updated = MagicMock(return_value=_pending_row(source="年报第 8 页"))
    monkeypatch.setattr(store, "update_candidate_source", updated)
    monkeypatch.setattr(
        "ai_agent.memory.profile.apply_profile_action",
        lambda *a, **k: MagicMock(text="用中文回答"),
    )
    monkeypatch.setattr(store, "mark_candidate", MagicMock(return_value=True))
    assert approve_candidate("c1", user_id="u1", source="年报第 8 页") is True
    updated.assert_called_once_with("c1", "年报第 8 页", user_id="u1")


def test_reject_candidate_ignores_already_decided(monkeypatch):
    _enable_promotion(monkeypatch)
    monkeypatch.setattr(
        store, "get_candidate", lambda *a, **k: _pending_row(status="promoted")
    )
    mark = MagicMock()
    monkeypatch.setattr(store, "mark_candidate", mark)
    assert reject_candidate("c1", user_id="u1") is False
    mark.assert_not_called()


# ---------- L4：开关、蒸馏、检索 ----------


def test_distill_archives_promoted_candidates_and_stable_episodes(monkeypatch):
    _enable_archive(monkeypatch)
    monkeypatch.setattr(
        store,
        "list_stable_candidates",
        lambda *a, **k: [
            dict(
                id="c1",
                text="用中文回答",
                facet="preference",
                source="人工核对",
                hit_count=6,
                session_seen=4,
                reviewed_by="alice",
            )
        ],
    )
    monkeypatch.setattr(
        store,
        "list_stable_episodes",
        lambda *a, **k: [
            dict(id="e1", text="在做研报深度层", hit_count=5, session_seen=3)
        ],
    )
    written = {}

    def fake_write(entries, user_id=None):
        written["entries"] = entries
        written["uid"] = user_id
        return 2

    monkeypatch.setattr(store, "write_archive_entries", fake_write)
    report = distill_memory(user_id="u1")
    assert report == {
        "archived": 2,
        "scanned_candidates": 1,
        "scanned_episodes": 1,
        "disabled": False,
    }
    kinds = {e["kind"] for e in written["entries"]}
    assert kinds == {"promoted", "episode"}
    promoted = next(e for e in written["entries"] if e["kind"] == "promoted")
    assert promoted["source"] == "人工核对"
    assert promoted["reviewed_by"] == "alice"
    assert written["uid"] == "u1"


def test_distill_fail_open_when_one_source_fails(monkeypatch):
    _enable_archive(monkeypatch)
    monkeypatch.setattr(
        store,
        "list_stable_candidates",
        MagicMock(side_effect=RuntimeError("db down")),
    )
    monkeypatch.setattr(
        store,
        "list_stable_episodes",
        lambda *a, **k: [dict(id="e1", text="稳定近况", hit_count=5, session_seen=3)],
    )
    written = []
    monkeypatch.setattr(
        store,
        "write_archive_entries",
        lambda entries, user_id=None: written.extend(entries) or len(entries),
    )
    report = distill_memory(user_id="u1")
    assert report["archived"] == 1
    assert written[0]["kind"] == "episode"


def test_distill_write_failure_is_fail_open(monkeypatch):
    _enable_archive(monkeypatch)
    monkeypatch.setattr(store, "list_stable_candidates", lambda *a, **k: [])
    monkeypatch.setattr(store, "list_stable_episodes", lambda *a, **k: [])
    monkeypatch.setattr(
        store, "write_archive_entries", MagicMock(side_effect=RuntimeError("db down"))
    )
    assert distill_memory(user_id="u1")["archived"] == 0


def test_search_archive_ranks_by_deterministic_relevance(monkeypatch):
    _enable_archive(monkeypatch)
    rows = [
        dict(id="a1", kind="episode", text="喜欢简洁的中文回答", facet="other"),
        dict(
            id="a2",
            kind="promoted",
            text="研报深度层红框结论需人工审核",
            facet="preference",
        ),
    ]
    monkeypatch.setattr(store, "list_archive_rows", lambda *a, **k: rows)
    hits = search_archive("研报红框", user_id="u1", top_k=1)
    assert [h["id"] for h in hits] == ["a2"]
    assert hits[0]["score"] > 0


def test_search_archive_fail_open_on_db_error(monkeypatch):
    _enable_archive(monkeypatch)
    monkeypatch.setattr(
        store, "list_archive_rows", MagicMock(side_effect=RuntimeError("db down"))
    )
    assert search_archive("任意", user_id="u1") == []


def test_distill_all_users_merges_and_deduplicates(monkeypatch):
    _enable_archive(monkeypatch)
    monkeypatch.setattr(store, "list_episode_users", lambda: ["u1", "u2"])
    monkeypatch.setattr(
        store, "list_candidate_users", lambda **k: ["u2", "u3"]
    )
    seen = []
    monkeypatch.setattr(
        archive,
        "distill_memory",
        lambda **kw: seen.append(kw["user_id"]) or {"archived": 1},
    )
    assert distill_all_users() == 3
    assert seen == ["u1", "u2", "u3"]


def test_archive_endpoints_return_payload(monkeypatch):
    _enable_archive(monkeypatch)
    monkeypatch.setattr(
        archive, "search_archive", lambda q, **k: [{"id": "a1", "text": "研报约定"}]
    )
    monkeypatch.setattr(
        archive, "distill_memory", lambda **k: {"archived": 2, "disabled": False}
    )
    out = memory_router.search_archive(
        query="研报", limit=5, user_id=None, user=_user()
    )
    assert out["total"] == 1 and out["user_id"] == "u1"
    distilled = memory_router.distill_archive(user_id=None, user=_user())
    assert distilled["archived"] == 2


# ---------- L4 展开：expand_memory ----------


def test_expand_memory_appends_archive_when_enabled(monkeypatch):
    _enable_archive(monkeypatch)
    monkeypatch.setattr(
        archive,
        "search_archive",
        lambda q, **k: [{"id": "a1", "text": "研报深度层红框需人工审核"}],
    )
    with (
        patch("ai_agent.tools.memory.memory.store.list_profile", return_value=[]),
        patch("ai_agent.tools.memory.memory.store.search_episode", return_value=[]),
    ):
        out = expand_memory("研报红框", kind="all")
    assert "L4 归档（按需展开，共 1 条）" in out
    assert "研报深度层红框需人工审核" in out


def test_expand_memory_archive_failure_does_not_break_other_sections(monkeypatch):
    _enable_archive(monkeypatch)
    monkeypatch.setattr(
        archive, "search_archive", MagicMock(side_effect=RuntimeError("db down"))
    )
    episode = MagicMock(id="e1", text="研报深度层 PDF 需人工批准")
    with (
        patch("ai_agent.tools.memory.memory.store.list_profile", return_value=[]),
        patch(
            "ai_agent.tools.memory.memory.store.search_episode",
            return_value=[episode],
        ),
    ):
        out = expand_memory("研报", kind="all")
    assert "相关情景记忆全文" in out
    assert "L4 归档读取暂不可用" in out


# ---------- DB 层：DDL / 幂等 / 清理 ----------


def test_ensure_schema_creates_archive_table_and_review_columns(monkeypatch):
    _enable_archive(monkeypatch)
    obj, sink = _fake_store(monkeypatch)
    obj.ensure_schema()
    sql = "\n".join(s for s, _ in sink)
    assert "CREATE TABLE IF NOT EXISTS agent_memory_archive" in sql
    assert "UNIQUE (user_id, kind, origin_id)" in sql
    assert "ADD COLUMN IF NOT EXISTS reviewed_by" in sql


def test_write_archive_entries_is_idempotent_and_skips_empty(monkeypatch):
    obj, sink = _fake_store(monkeypatch, rowcount=1)
    assert obj.write_archive_entries([], user_id="u1") == 0
    assert sink == []
    n = obj.write_archive_entries(
        [
            dict(kind="episode", origin_id="e1", text="稳定近况"),
            dict(kind="promoted", origin_id="", text="无 origin 应跳过"),
        ],
        user_id="u1",
    )
    assert n == 1
    assert "ON CONFLICT (user_id, kind, origin_id) DO NOTHING" in sink[0][0]


def test_list_stable_episodes_sql_uses_evidence_columns(monkeypatch):
    obj, sink = _fake_store(monkeypatch, rows=[])
    obj.list_stable_episodes("u1", min_hits=4, min_sessions=3, limit=10)
    sql, params = sink[0]
    assert "COALESCE(hit_count, 0) >= %s" in sql
    assert "cardinality(COALESCE(sessions_seen" in sql
    assert params == ("u1", 4, 3, 10)


def test_update_candidate_source_only_touches_pending(monkeypatch):
    obj, sink = _fake_store(monkeypatch, rows=[_pending_row()])
    row = obj.update_candidate_source("c1", "年报第 8 页", user_id="u1")
    assert row["id"] == "c1"
    sql, params = sink[0]
    assert "status = 'pending'" in sql
    assert params == ("年报第 8 页", "c1", "u1")


def test_mark_candidate_default_sql_is_unchanged(monkeypatch):
    obj, sink = _fake_store(monkeypatch, rows=[{"id": "c1"}])
    assert obj.mark_candidate("c1", "promoted", user_id="u1") is True
    sql, params = sink[0]
    assert "reviewed_by" not in sql
    assert params == ("promoted", "c1", "u1")


def test_mark_candidate_records_reviewer_when_supplied(monkeypatch):
    obj, sink = _fake_store(monkeypatch, rows=[{"id": "c1"}])
    assert obj.mark_candidate(
        "c1", "rejected", user_id="u1", reviewed_by="alice", note="不是长期偏好"
    ) is True
    sql, params = sink[0]
    assert "reviewed_by = %s" in sql and "reviewed_at = now()" in sql
    assert params == ("rejected", "alice", "不是长期偏好", "c1", "u1")


def test_clear_long_term_scopes_optional_tables_by_user(monkeypatch):
    _enable_archive(monkeypatch)
    # 假连接池让两次 to_regclass 都返回非空：两张可选表都存在时，
    # 删除必须都按 user_id 收敛，不能出现全表 DELETE。
    obj, sink = _fake_store(monkeypatch, rows=[{"t": "agent_memory_candidate"}])
    obj.clear_long_term("u1")
    sql = "\n".join(s for s, _ in sink)
    assert "to_regclass('agent_memory_archive')" in sql
    deletes = [s.strip() for s, _ in sink if "DELETE FROM agent_memory" in s]
    assert len(deletes) == 2
    assert all("WHERE user_id = %s" in s for s in deletes)
    delete_params = [p for s, p in sink if "DELETE FROM agent_memory" in s]
    assert delete_params == [("u1",), ("u1",)]
