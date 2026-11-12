"""L1–L4 全量导出。

导出是隐私权的一部分：用户应当能拿到系统记住的**全部**内容，而不是分页
接口的一页。这里按四层逐表读出（profile / episode / candidate / archive），
时间统一转成字符串，交给 FastAPI 序列化。任一查询失败就整体失败，不返回
一份「看起来完整、实际缺层」的导出。
"""

from __future__ import annotations

from datetime import datetime, timezone

from ai_agent.storage.db import Database, current_user_id


class ExportStore(Database):
    def export_memory(self, user_id: str | None = None) -> dict:
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            profile = conn.execute(
                """
                SELECT id::text AS id, user_id, text, facet,
                       created_at::text AS created_at,
                       expires_at::text AS expires_at,
                       superseded_at::text AS superseded_at
                FROM agent_user_profile
                WHERE user_id = %s
                ORDER BY created_at DESC, id
                """,
                (uid,),
            ).fetchall()
            episodes = conn.execute(
                """
                SELECT id::text AS id, user_id, text,
                       created_at::text AS created_at,
                       COALESCE(last_seen_at, created_at)::text AS last_seen,
                       COALESCE(hit_count, 0)::int AS hit_count,
                       cardinality(COALESCE(sessions_seen, '{}'::text[]))
                           AS session_seen
                FROM agent_episode
                WHERE user_id = %s
                ORDER BY COALESCE(last_seen_at, created_at) DESC, id
                """,
                (uid,),
            ).fetchall()
            candidates = conn.execute(
                """
                SELECT id::text AS id, user_id, text, facet,
                       COALESCE(source, '') AS source,
                       COALESCE(hit_count, 0)::int AS hit_count,
                       COALESCE(session_seen, 0)::int AS session_seen,
                       status, created_at::text AS created_at,
                       updated_at::text AS updated_at,
                       COALESCE(reviewed_by, '') AS reviewed_by,
                       COALESCE(review_note, '') AS review_note,
                       reviewed_at::text AS reviewed_at
                FROM agent_memory_candidate
                WHERE user_id = %s
                ORDER BY updated_at DESC, id
                """,
                (uid,),
            ).fetchall()
            archive = conn.execute(
                """
                SELECT id::text AS id, user_id, kind, origin_id, text, facet,
                       COALESCE(source, '') AS source,
                       COALESCE(hit_count, 0)::int AS hit_count,
                       COALESCE(session_seen, 0)::int AS session_seen,
                       COALESCE(reviewed_by, '') AS reviewed_by,
                       archived_at::text AS archived_at
                FROM agent_memory_archive
                WHERE user_id = %s
                ORDER BY archived_at DESC, id
                """,
                (uid,),
            ).fetchall()
            setting = conn.execute(
                "SELECT enabled FROM agent_memory_setting WHERE user_id = %s",
                (uid,),
            ).fetchone()
        return {
            "schema_version": 1,
            "user_id": uid,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "memory_enabled": True if setting is None else bool(setting["enabled"]),
            "profile": [dict(row) for row in profile],
            "episodes": [dict(row) for row in episodes],
            "candidates": [dict(row) for row in candidates],
            "archive": [dict(row) for row in archive],
        }