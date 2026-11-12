"""L1/L2 管理、记忆开关、导出、归档分页与更新的回归。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from ai_agent.api.routers import memory as memory_router
from ai_agent.api.schemas import MemorySettingsRequest
from ai_agent.core.config import settings
from ai_agent.memory.episode_merge import merge_episode_texts
from ai_agent.memory.facets import infer_facet
from ai_agent.memory.long_term import _memory_from_row, capture_long_term
from ai_agent.storage.db import UserRecord, user_scope
from ai_agent.storage.store import MemoryStore

UID = "memory_management_user"


def _fake_embed(texts: list[str]) -> list[list[float]]:
    dim = settings.embedding_dim
    vecs: list[list[float]] = []
    for text in texts:
        v = [0.0] * dim
        for word in text.lower().split():
            v[hash(word) % dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        vecs.append([x / norm for x in v])
    return vecs


def _cleanup(s: MemoryStore) -> None:
    s.clear_long_term(user_id=UID)
    s.clear_chat(UID)
    s.clear_sessions(UID)
    s.clear_checkpoints(UID)
    with s._get_pool().connection() as conn:
        conn.execute("DELETE FROM agent_memory_setting WHERE user_id = %s", (UID,))
        conn.commit()


@pytest.fixture()
def mem_store(monkeypatch):
    monkeypatch.setattr("ai_agent.memory.episode_store.embed_texts", _fake_embed)
    monkeypatch.setattr(
        "ai_agent.memory.episode_store.embed_query", lambda q: _fake_embed([q])[0]
    )
    monkeypatch.setattr(
        "ai_agent.memory.episode_store.compact_episode_texts", merge_episode_texts
    )
    s = MemoryStore()
    try:
        s.ensure_schema()
    except Exception as e:  # pragma: no cover - 无库时跳过
        pytest.skip(f"Postgres 不可用: {e}")
    _cleanup(s)
    with user_scope(UID):
        yield s
    _cleanup(s)
    s.close()


def _user(uid: str = UID, username: str = "alice") -> UserRecord:
    return UserRecord(user_id=uid, username=username, display_name=username)


def _insert_archive(s: MemoryStore, kind: str, origin_id: str, text: str) -> None:
    s.write_archive_entries(
        [
            dict(
                kind=kind,
                origin_id=origin_id,
                text=text,
                facet="other",
                source="",
                hit_count=1,
                session_seen=1,
                reviewed_by="",
            )
        ],
        user_id=UID,
    )


# ---------- 纯函数 / 开关 ----------


@pytest.mark.parametrize(
    ("text", "facet"),
    [
        ("用户偏好简洁的中文回答", "preference"),
        ("用户是后端工程师", "occupation"),
        ("我住在北京", "locale"),
        ("以后都用英文回答", "constraint"),
        ("我叫林晓", "name"),
        ("在做知识库文档路由", "other"),
    ],
)
def test_infer_facet_uses_conservative_clues(text, facet):
    assert infer_facet(text) == facet


def test_memory_setting_defaults_on_and_roundtrips(mem_store):
    assert mem_store.memory_enabled(UID) is True
    assert mem_store.set_memory_enabled(False, user_id=UID) is False
    assert mem_store.memory_enabled(UID) is False
    assert mem_store.set_memory_enabled(True, user_id=UID) is True


def test_capture_long_term_skips_when_memory_disabled():
    with patch("ai_agent.memory.long_term.extract_turn_memories") as extract:
        capture_long_term(
            "u1",
            [{"role": "assistant", "content": "ok"}],
            "",
            "hi",
            memory_enabled=False,
        )
    extract.assert_not_called()


# ---------- L1 / L2 管理 ----------


def test_profile_page_and_delete_cascades(mem_store):
    for text in ("喜欢蓝色", "喜欢绿色", "喜欢红色"):
        mem_store.write_profile(text, user_id=UID, facet="preference")

    first, has_more = mem_store.list_profile_page(UID, limit=2)
    assert len(first) == 2 and has_more
    second, has_more2 = mem_store.list_profile_page(UID, limit=2, offset=2)
    assert len(second) == 1 and not has_more2

    target = first[0]
    mem_store.upsert_candidate(
        text=target["text"],
        facet="preference",
        hit_count=5,
        session_seen=4,
        user_id=UID,
    )
    _insert_archive(mem_store, "promoted", "c1", target["text"])

    assert mem_store.delete_profile_item(target["id"], user_id=UID) is True
    assert mem_store.delete_profile_item(target["id"], user_id=UID) is False
    assert not [
        row
        for row in mem_store.list_candidates(UID, status="pending")
        if row["text"] == target["text"]
    ]
    assert not [
        row
        for row in mem_store.list_archive_rows(UID)
        if row["text"] == target["text"]
    ]


def test_clear_profile_counts_only_profile_rows(mem_store):
    for text in ("喜欢蓝色", "喜欢绿色"):
        mem_store.write_profile(text, user_id=UID, facet="preference")
    assert mem_store.clear_profile(UID) == 2
    assert mem_store.list_profile_page(UID)[0] == []


def test_episode_page_and_delete_cascades(mem_store):
    first = mem_store.write_episode("alpha beta gamma", user_id=UID)
    mem_store.write_episode("delta epsilon zeta", user_id=UID)

    page, has_more = mem_store.list_episodes_page(UID, limit=1)
    assert len(page) == 1 and has_more
    target = page[0]
    mem_store.upsert_candidate(
        text=target["text"],
        facet="other",
        hit_count=5,
        session_seen=4,
        user_id=UID,
    )
    _insert_archive(mem_store, "episode", target["id"], target["text"])

    assert mem_store.delete_episode(target["id"], user_id=UID) is True
    assert mem_store.delete_episode(target["id"], user_id=UID) is False
    remaining = [item["id"] for item in mem_store.list_episodes_page(UID)[0]]
    assert target["id"] not in remaining
    assert len(remaining) == 1
    assert not [
        row
        for row in mem_store.list_candidates(UID, status="pending")
        if row["text"] == target["text"]
    ]
    assert not [
        row
        for row in mem_store.list_archive_rows(UID)
        if row["origin_id"] == target["id"]
    ]


def test_clear_episodes_removes_pending_candidates_and_archive_copies(mem_store):
    item = mem_store.write_episode("alpha beta gamma", user_id=UID)
    mem_store.upsert_candidate(
        text=item.text, facet="other", hit_count=5, session_seen=4, user_id=UID
    )
    _insert_archive(mem_store, "episode", item.id, item.text)
    assert mem_store.clear_episodes(UID) == 1
    assert mem_store.list_episodes_page(UID)[0] == []
    assert mem_store.list_candidates(UID, status="pending") == []
    assert mem_store.list_archive_rows(UID) == []


def test_background_user_lists_skip_disabled_users(mem_store):
    item = mem_store.write_episode("alpha beta gamma", user_id=UID)
    mem_store.upsert_candidate(
        text=item.text, facet="other", hit_count=5, session_seen=4, user_id=UID
    )
    assert UID in mem_store.list_episode_users()
    assert UID in mem_store.list_candidate_users(status="pending")

    mem_store.set_memory_enabled(False, user_id=UID)
    assert UID not in mem_store.list_episode_users(enabled_only=True)
    assert UID not in mem_store.list_candidate_users(
        status="pending", enabled_only=True
    )


# ---------- 归档更新 / 分页 ----------


def test_archive_upsert_updates_existing_copy_and_pages(mem_store):
    _insert_archive(mem_store, "episode", "e1", "旧文本")
    rows = mem_store.list_archive_rows(UID)
    assert rows[0]["text"] == "旧文本"

    # 同一 origin 再次蒸馏：不应新增一行，但文本/证据要更新
    n = mem_store.write_archive_entries(
        [
            dict(
                kind="episode",
                origin_id="e1",
                text="新文本",
                facet="other",
                source="出处",
                hit_count=4,
                session_seen=3,
                reviewed_by="alice",
            )
        ],
        user_id=UID,
    )
    assert n == 0
    rows = mem_store.list_archive_rows(UID)
    assert len(rows) == 1
    assert rows[0]["text"] == "新文本"
    assert rows[0]["hit_count"] == 4
    assert rows[0]["source"] == "出处"

    _insert_archive(mem_store, "episode", "e2", "第二条")
    _insert_archive(mem_store, "episode", "e3", "第三条")
    page1 = mem_store.list_archive_rows(UID, limit=2)
    page2 = mem_store.list_archive_rows(UID, limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 1
    assert {row["origin_id"] for row in page1} & {
        row["origin_id"] for row in page2
    } == set()


# ---------- 全量导出 ----------


def test_export_memory_includes_all_layers(mem_store):
    mem_store.write_profile("喜欢蓝色", user_id=UID, facet="preference")
    item = mem_store.write_episode("在做记忆评测", user_id=UID)
    mem_store.upsert_candidate(
        text=item.text, facet="other", hit_count=5, session_seen=4, user_id=UID
    )
    _insert_archive(mem_store, "episode", item.id, item.text)

    data = mem_store.export_memory(UID)
    assert data["schema_version"] == 1
    assert data["user_id"] == UID
    assert data["memory_enabled"] is True
    assert len(data["profile"]) == 1
    assert len(data["episodes"]) == 1
    assert len(data["candidates"]) == 1
    assert len(data["archive"]) == 1


# ---------- 路由层 ----------


def test_settings_route_roundtrip(monkeypatch):
    monkeypatch.setattr(memory_router.store, "memory_enabled", lambda uid: False)
    monkeypatch.setattr(
        memory_router.store, "set_memory_enabled", lambda enabled, user_id=None: enabled
    )
    out = memory_router.get_memory_settings(user_id=None, user=_user())
    assert out == {"user_id": UID, "enabled": False}
    out = memory_router.update_memory_settings(
        MemorySettingsRequest(enabled=True), user_id=None, user=_user()
    )
    assert out == {"user_id": UID, "enabled": True}


def test_profile_route_reports_next_offset(monkeypatch):
    monkeypatch.setattr(
        memory_router.store,
        "list_profile_page",
        lambda uid, **kw: ([{"id": "p1", "text": "喜欢蓝色"}], True),
    )
    out = memory_router.list_profile(
        limit=1, offset=0, include_inactive=False, user_id=None, user=_user()
    )
    assert out["total"] == 1
    assert out["has_more"] is True
    assert out["next_offset"] == 1


def test_episode_delete_route_404(monkeypatch):
    monkeypatch.setattr(
        memory_router.store, "delete_episode", lambda *a, **k: False
    )
    with pytest.raises(HTTPException) as exc:
        memory_router.delete_episode("missing", user_id=None, user=_user())
    assert exc.value.status_code == 404


def test_archive_route_paginates_empty_query(monkeypatch):
    rows = [
        {"id": "a1", "text": "一"},
        {"id": "a2", "text": "二"},
        {"id": "a3", "text": "三"},
    ]
    monkeypatch.setattr(
        memory_router.store,
        "list_archive_rows",
        lambda uid, **kw: rows,
    )
    out = memory_router.search_archive(
        query="", limit=2, offset=0, user_id=None, user=_user()
    )
    assert out["total"] == 2
    assert out["has_more"] is True
    assert out["next_offset"] == 2


def test_export_and_clear_routes(monkeypatch):
    monkeypatch.setattr(
        memory_router.store,
        "export_memory",
        lambda uid: {"user_id": uid, "profile": []},
    )
    assert memory_router.export_memory(user_id=None, user=_user())["profile"] == []
    called = MagicMock()
    monkeypatch.setattr(memory_router.store, "clear_long_term", called)
    out = memory_router.clear_memory(user_id=None, user=_user())
    called.assert_called_once_with(UID)
    assert out["deleted"] is True


def test_run_loop_filters_memory_tools_when_disabled(monkeypatch):
    from ai_agent.agents import loop

    seen: list[list[str]] = []

    def fake_stream(messages, tools, **kwargs):
        seen.append([t["function"]["name"] for t in tools])
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    monkeypatch.setattr(
        loop,
        "tool_schemas",
        lambda: [
            {"function": {"name": name}}
            for name in (
                "get_weather",
                "search_memory",
                "write_memory",
                "expand_memory",
            )
        ],
    )
    monkeypatch.setattr(loop, "stream_chat", fake_stream)
    list(loop.run_loop([{"role": "user", "content": "hi"}], use_kb=False, use_memory=False))
    assert seen and seen[0] == ["get_weather"]

    list(loop.run_loop([{"role": "user", "content": "hi"}], use_kb=False, use_memory=True))
    assert {"search_memory", "write_memory", "expand_memory"} <= set(seen[-1])


def test_system_prompt_omits_memory_when_disabled(monkeypatch):
    from ai_agent.agents import loop

    profile = MagicMock(return_value="PROFILE")
    recall = MagicMock(return_value="RECALL")
    monkeypatch.setattr(loop, "_profile_prompt", profile)
    monkeypatch.setattr(loop, "_recall_block", recall)
    out = loop._system_prompt("u1", use_kb=False, use_memory=False)
    assert "记忆纪律" not in out
    profile.assert_not_called()
    recall.assert_not_called()


def test_extracted_episode_can_carry_facet():
    item = _memory_from_row(
        {"kind": "episode", "content": "在做 BFF", "facet": "occupation"}
    )
    assert item is not None
    assert item.facet == "occupation"


def test_extracted_episode_without_facet_leaves_it_for_inference():
    item = _memory_from_row({"kind": "episode", "content": "在做 BFF"})
    assert item is not None
    assert item.facet == ""