"""渐进式记忆 · 第 5 步的 DB 层：L4 蒸馏归档（独立 mixin）。

只做三件事：挑出「够稳定」的记忆、幂等写入归档表、按用户读回归档。
所有 SQL 都由 archive 编排层调用，
**同时**打开后才会被编排层调用（episode 的证据列与候选表都由第 4 步创建）。

归档语义是「**只读副本 + 审计沉淀**」：不删 episode、不改画像、不改检索
结果，也不进 system 人格块。``UNIQUE (user_id, kind, origin_id)`` +
``ON CONFLICT DO NOTHING`` 保证后台任务反复跑也不会产生重复条目。
"""

from __future__ import annotations

from ai_agent.core.logging import get_logger
from ai_agent.storage.db import Database, current_user_id

logger = get_logger(__name__)


class ArchiveStore(Database):
    def list_stable_candidates(
        self, user_id: str | None = None, *, limit: int = 200
    ) -> list[dict]:
        """人工批准过的候选：结论 + 出处 + 审核人，是最可信的归档来源。"""
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT id::text AS id, text, facet,
                       COALESCE(source, '') AS source,
                       COALESCE(hit_count, 0)::int AS hit_count,
                       COALESCE(session_seen, 0)::int AS session_seen,
                       COALESCE(reviewed_by, '') AS reviewed_by
                FROM agent_memory_candidate
                WHERE user_id = %s AND status = 'promoted'
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (uid, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_stable_episodes(
        self,
        user_id: str | None = None,
        *,
        min_hits: int = 3,
        min_sessions: int = 3,
        limit: int = 200,
    ) -> list[dict]:
        """反复被用到、且跨会话复现的近况，才值得沉淀成可回溯归档。"""
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT id::text AS id, text,
                       COALESCE(hit_count, 0)::int AS hit_count,
                       cardinality(COALESCE(sessions_seen, '{}'::text[]))
                           AS session_seen,
                       created_at::text AS created_at
                FROM agent_episode
                WHERE user_id = %s
                  AND COALESCE(hit_count, 0) >= %s
                  AND cardinality(COALESCE(sessions_seen, '{}'::text[])) >= %s
                ORDER BY COALESCE(last_seen_at, created_at) DESC
                LIMIT %s
                """,
                (uid, max(0, min_hits), max(0, min_sessions), limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def write_archive_entries(
        self, entries: list[dict], user_id: str | None = None
    ) -> int:
        """幂等写入。返回真正新增的条数（已存在的不计）。"""
        if not entries:
            return 0
        uid = user_id or current_user_id()
        written = 0
        with self._get_pool().connection() as conn:
            for entry in entries:
                kind = str(entry.get("kind") or "episode")
                origin_id = str(entry.get("origin_id") or "")
                if not origin_id or not str(entry.get("text") or "").strip():
                    continue
                item_id = str(
                    self._point_id(
                        "agent_memory_archive", uid, f"{kind}:{origin_id}"
                    )
                )
                row = conn.execute(
                    """
                    INSERT INTO agent_memory_archive (
                        id, user_id, kind, origin_id, text, facet, source,
                        hit_count, session_seen, reviewed_by, archived_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (user_id, kind, origin_id) DO UPDATE
                        SET text = EXCLUDED.text,
                            facet = EXCLUDED.facet,
                            source = EXCLUDED.source,
                            hit_count = EXCLUDED.hit_count,
                            session_seen = EXCLUDED.session_seen,
                            reviewed_by = EXCLUDED.reviewed_by
                    RETURNING (xmax = 0) AS inserted
                    """,
                    (
                        item_id,
                        uid,
                        kind,
                        origin_id,
                        str(entry["text"]).strip(),
                        str(entry.get("facet") or "other"),
                        str(entry.get("source") or ""),
                        int(entry.get("hit_count") or 0),
                        int(entry.get("session_seen") or 0),
                        str(entry.get("reviewed_by") or ""),
                    ),
                ).fetchone()
                if row and row.get("inserted"):
                    written += 1
            conn.commit()
        return written

    def list_archive_rows(
        self,
        user_id: str | None = None,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        """取最近归档条目；相关性排序在编排层用确定性 bigram 做。

        不把排序塞进 SQL：归档查询要零向量、零 LLM，且与 ``progressive``
        的画像相关性口径完全一致（同一套 ``relevance``）。``offset`` 供
        管理页翻页；无 query 时直接按 archived_at 倒序稳定分页。
        """
        uid = user_id or current_user_id()
        page = max(1, min(int(limit), 1000))
        start = max(0, int(offset))
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT id::text AS id, kind, origin_id, text, facet,
                       COALESCE(source, '') AS source,
                       COALESCE(hit_count, 0)::int AS hit_count,
                       COALESCE(session_seen, 0)::int AS session_seen,
                       COALESCE(reviewed_by, '') AS reviewed_by,
                       archived_at::text AS archived_at
                FROM agent_memory_archive
                WHERE user_id = %s
                ORDER BY archived_at DESC, id
                LIMIT %s OFFSET %s
                """,
                (uid, page, start),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_episode_users(self, *, enabled_only: bool = False) -> list[str]:
        """后台线程没有 user context，按用户遍历，避免只处理 default。

        ``enabled_only`` 供后台蒸馏跳过明确关闭记忆的用户。
        """
        enabled_filter = ""
        if enabled_only:
            enabled_filter = """
              AND NOT EXISTS (
                  SELECT 1 FROM agent_memory_setting s
                  WHERE s.user_id = agent_episode.user_id
                    AND s.enabled = FALSE
              )
            """
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT user_id FROM agent_episode WHERE TRUE {enabled_filter}"
            ).fetchall()
        return [str(row["user_id"]) for row in rows]

    def clear_archive(self, user_id: str | None = None) -> int:
        """被 clear_long_term 调用；表不存在时由调用方先探测，避免事务 aborted。"""
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            result = conn.execute(
                "DELETE FROM agent_memory_archive WHERE user_id = %s", (uid,)
            )
            conn.commit()
            return result.rowcount or 0
