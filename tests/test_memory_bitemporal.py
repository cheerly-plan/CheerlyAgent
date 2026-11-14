"""画像双时间轴：事实的生效区间要能回查，而不是只剩最后一条。"""

from datetime import datetime, timedelta, timezone

import psycopg

from ai_agent.core.config import settings
from ai_agent.memory.profile import ProfileAction, apply_profile_action
from ai_agent.storage.store import MemoryStore


def _clean(user_id: str) -> None:
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("DELETE FROM agent_user_profile WHERE user_id = %s", (user_id,))
        conn.commit()


def _rows(user_id: str):
    with psycopg.connect(settings.database_url) as conn:
        return conn.execute(
            """
            SELECT text, valid_from, valid_to, superseded_at
            FROM agent_user_profile
            WHERE user_id = %s
            ORDER BY valid_from
            """,
            (user_id,),
        ).fetchall()


def test_mutex_facet_keeps_history_with_valid_range():
    store = MemoryStore()
    uid = "bt-mutex"
    _clean(uid)
    try:
        apply_profile_action(
            ProfileAction(op="add", facet="locale", content="我住在深圳"), user_id=uid
        )
        past = datetime.now(timezone.utc) - timedelta(days=400)
        with psycopg.connect(settings.database_url) as conn:
            conn.execute(
                "UPDATE agent_user_profile SET valid_from = %s, created_at = %s "
                "WHERE user_id = %s",
                (past, past, uid),
            )
            conn.commit()

        apply_profile_action(
            ProfileAction(op="add", facet="locale", content="我现在住在上海"),
            user_id=uid,
        )
        rows = _rows(uid)
        assert [row[0] for row in rows] == ["我住在深圳", "我现在住在上海"]
        old, new = rows
        assert old[2] is not None            # 旧值有 valid_to
        assert old[3] is not None            # 且被标为 superseded
        assert new[2] is None                # 新值仍生效
        assert new[3] is None

        # 现在只看到上海
        assert [item.text for item in store.list_profile(uid)] == ["我现在住在上海"]
        # 时间点回查能看到当时的深圳
        as_of = (past + timedelta(days=10)).isoformat()
        assert [item.text for item in store.list_profile(uid, as_of=as_of)] == [
            "我住在深圳"
        ]
    finally:
        _clean(uid)


def test_mutex_facet_only_one_active_row():
    store = MemoryStore()
    uid = "bt-active"
    _clean(uid)
    try:
        for text in ("我住在深圳", "我现在住在上海", "我住在北京"):
            apply_profile_action(
                ProfileAction(op="add", facet="locale", content=text), user_id=uid
            )
        active = [row for row in _rows(uid) if row[3] is None]
        assert len(active) == 1
        assert active[0][0] == "我住在北京"
        assert [item.text for item in store.list_profile(uid)] == ["我住在北京"]
    finally:
        _clean(uid)


def test_accumulating_facet_still_dedupes_by_text():
    store = MemoryStore()
    uid = "bt-acc"
    _clean(uid)
    try:
        for _ in range(3):
            apply_profile_action(
                ProfileAction(op="add", facet="preference", content="回答先给结论"),
                user_id=uid,
            )
        rows = _rows(uid)
        assert len(rows) == 1
    finally:
        _clean(uid)
