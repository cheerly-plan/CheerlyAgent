"""渐进式记忆 · 第 4 步的 DB 层（独立 mixin，零业务判断）。

只做四件事：命中计数 / 跨会话证据登记 / 候选表读写 / 衰减与续期清理。
所有 SQL 都只在 ``memory_promotion_enabled`` 打开后才会被调用；调用方
（``EpisodeStore`` 与 ``promotion.py``）负责开关判断与 fail-open 兜底。

与 ``ensure_schema`` 里的新列/新表配套：老库通过 ``ADD COLUMN IF NOT
EXISTS`` + 默认值平滑升级，无需数据迁移；开关关闭时连 DDL 都不执行。
"""

from __future__ import annotations

from ai_agent.core.logging import get_logger
from ai_agent.storage.db import Database, current_user_id

logger = get_logger(__name__)


class PromotionStore(Database):
    def record_episode_hits(
        self, ids: list[str], user_id: str | None = None
    ) -> int:
        """检索命中：热度 +1（同一时间窗内去重）并把 TTL 基准续到 now()。

        同一轮里自动召回与抽取前的相近检索会各查一次，不做时间窗去重会把
        热度虚高成两倍。去重只压 ``hit_count``，``last_seen_at`` 每次都刷新。
        """
        if not ids:
            return 0
        uid = user_id or current_user_id()
        from ai_agent.core.config import settings

        dedupe = max(0, settings.memory_promotion_hit_dedupe_seconds)
        with self._get_pool().connection() as conn:
            result = conn.execute(
                """
                UPDATE agent_episode
                SET hit_count = COALESCE(hit_count, 0) + CASE
                        WHEN last_seen_at IS NULL
                          OR last_seen_at < now() - (%s * interval '1 second')
                        THEN 1 ELSE 0 END,
                    last_seen_at = now()
                WHERE user_id = %s AND id::text = ANY(%s::text[])
                """,
                (dedupe, uid, ids),
            )
            conn.commit()
            return result.rowcount or 0

    def touch_episode_session(
        self, episode_id: str, session_id: str, user_id: str | None = None
    ) -> tuple[int, int]:
        """把本次会话登记进 episode 的去重会话集合，返回 (热度, 会话数)。

        同一会话里抽十次同一件事只算一个会话，避免「一个长会话刷满门槛」
        把一次性近况误升成长期约定。
        """
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                UPDATE agent_episode
                SET sessions_seen = CASE
                        WHEN %s = ANY(COALESCE(sessions_seen, '{}'::text[]))
                        THEN COALESCE(sessions_seen, '{}'::text[])
                        ELSE array_append(COALESCE(sessions_seen, '{}'::text[]), %s)
                    END
                WHERE id::text = %s AND user_id = %s
                RETURNING COALESCE(hit_count, 0)::int AS hit_count,
                          cardinality(
                              COALESCE(sessions_seen, '{}'::text[])
                          ) AS session_seen
                """,
                (session_id, session_id, episode_id, uid),
            ).fetchone()
            conn.commit()
        if row is None:
            return 0, 0
        return int(row["hit_count"]), int(row["session_seen"])

    def episode_evidence(
        self, episode_id: str, user_id: str | None = None
    ) -> tuple[int, int]:
        """没有 session id 时只读证据，不写库。"""
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(hit_count, 0)::int AS hit_count,
                       cardinality(COALESCE(sessions_seen, '{}'::text[]))
                           AS session_seen
                FROM agent_episode
                WHERE id::text = %s AND user_id = %s
                """,
                (episode_id, uid),
            ).fetchone()
        if row is None:
            return 0, 0
        return int(row["hit_count"]), int(row["session_seen"])

    def upsert_candidate(
        self,
        text: str,
        *,
        facet: str = "other",
        source: str = "",
        hit_count: int = 0,
        session_seen: int = 0,
        user_id: str | None = None,
    ) -> str:
        """登记 L3 候选。同文本幂等；已 rejected / promoted 的不复活。"""
        uid = user_id or current_user_id()
        item_id = str(self._point_id("agent_memory_candidate", uid, text.strip()))
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_memory_candidate (
                    id, user_id, text, facet, source,
                    hit_count, session_seen, status, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', now(), now())
                ON CONFLICT (id) DO UPDATE SET
                    hit_count = GREATEST(
                        agent_memory_candidate.hit_count, EXCLUDED.hit_count
                    ),
                    session_seen = GREATEST(
                        agent_memory_candidate.session_seen, EXCLUDED.session_seen
                    ),
                    source = CASE
                        WHEN EXCLUDED.source <> ''
                        THEN EXCLUDED.source
                        ELSE agent_memory_candidate.source END,
                    updated_at = now()
                WHERE agent_memory_candidate.status = 'pending'
                """,
                (item_id, uid, text.strip(), facet, source or "",
                 hit_count, session_seen),
            )
            conn.commit()
        return item_id

    def list_candidates(
        self,
        user_id: str | None = None,
        *,
        status: str = "pending",
        limit: int = 20,
    ) -> list[dict]:
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT id::text AS id, text, facet, source,
                       COALESCE(hit_count, 0)::int AS hit_count,
                       COALESCE(session_seen, 0)::int AS session_seen,
                       status
                FROM agent_memory_candidate
                WHERE user_id = %s AND status = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (uid, status, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_candidate(
        self, candidate_id: str, user_id: str | None = None
    ) -> dict | None:
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT id::text AS id, text, facet, source,
                       COALESCE(hit_count, 0)::int AS hit_count,
                       COALESCE(session_seen, 0)::int AS session_seen,
                       status
                FROM agent_memory_candidate
                WHERE id::text = %s AND user_id = %s
                """,
                (candidate_id, uid),
            ).fetchone()
        return dict(row) if row else None

    def list_candidate_users(self, *, status: str = "pending") -> list[str]:
        """后台清理线程没有 user context，按用户逐个审核，避免只处理 default。"""
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT user_id
                FROM agent_memory_candidate
                WHERE status = %s
                """,
                (status,),
            ).fetchall()
        return [str(row["user_id"]) for row in rows]

    def mark_candidate(
        self,
        candidate_id: str,
        status: str,
        user_id: str | None = None,
        *,
        reviewed_by: str = "",
        note: str = "",
    ) -> bool:
        """标记候选状态。审核元数据是第 5 步人工入口的可选增量：
        不传时 SQL 与第 4 步逐字一致，后台自动审核路径不受影响。"""
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            if reviewed_by or note:
                row = conn.execute(
                    """
                    UPDATE agent_memory_candidate
                    SET status = %s, reviewed_by = %s, review_note = %s,
                        reviewed_at = now(), updated_at = now()
                    WHERE id::text = %s AND user_id = %s AND status = 'pending'
                    RETURNING id
                    """,
                    (status, reviewed_by, note[:500], candidate_id, uid),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    UPDATE agent_memory_candidate
                    SET status = %s, updated_at = now()
                    WHERE id::text = %s AND user_id = %s AND status = 'pending'
                    RETURNING id
                    """,
                    (status, candidate_id, uid),
                ).fetchone()
            conn.commit()
        return row is not None

    def update_candidate_source(
        self, candidate_id: str, source: str, user_id: str | None = None
    ) -> dict | None:
        """人工核对原文后补录出处，返回更新后的候选行。

        **只接受人填的字符串**，不接 LLM 抽取结果；且只允许 pending 状态改。
        含数字的候选在补出处前会被 evaluate_candidate 挡住，这条 SQL 是
        让人工审核真正能闭环的落点，不是绕过门禁的后门。
        """
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                UPDATE agent_memory_candidate
                SET source = %s, updated_at = now()
                WHERE id::text = %s AND user_id = %s AND status = 'pending'
                RETURNING id::text AS id, text, facet, source,
                          COALESCE(hit_count, 0)::int AS hit_count,
                          COALESCE(session_seen, 0)::int AS session_seen,
                          status
                """,
                (source.strip()[:500], candidate_id, uid),
            ).fetchone()
            conn.commit()
        return dict(row) if row else None

    def delete_candidates_matching(
        self, text: str, user_id: str | None = None
    ) -> int:
        """用户明确否定：精确文本匹配才删，宁可漏删也不误删别的候选。"""
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            result = conn.execute(
                """
                DELETE FROM agent_memory_candidate
                WHERE user_id = %s AND text = %s
                """,
                (uid, text.strip()),
            )
            conn.commit()
            return result.rowcount or 0

    def clear_candidates(self, user_id: str | None = None) -> int:
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            result = conn.execute(
                "DELETE FROM agent_memory_candidate WHERE user_id = %s", (uid,)
            )
            conn.commit()
            return result.rowcount or 0

    def decay_and_cleanup_episodes(
        self,
        *,
        ttl_seconds: int,
        decay_seconds: int,
        min_hits: int,
        user_id: str | None = None,
    ) -> int:
        """渐进式淘汰：先降权，再按「最后见到」删除低热度且过期的行。

        高热度 episode 即使过期也不物理删除，交给下一轮衰减慢慢退场；
        ``last_decay_at`` 保证每个衰减周期最多降一次，不会一轮归零。
        """
        scope = " AND user_id = %s" if user_id else ""
        tail = (user_id,) if user_id else ()
        with self._get_pool().connection() as conn:
            if decay_seconds > 0:
                decayed = conn.execute(
                    f"""
                    UPDATE agent_episode
                    SET hit_count = GREATEST(COALESCE(hit_count, 0) - 1, 0),
                        last_decay_at = now()
                    WHERE COALESCE(hit_count, 0) > 0
                      AND COALESCE(last_seen_at, created_at)
                          < now() - (%s * interval '1 second')
                      AND (last_decay_at IS NULL
                           OR last_decay_at < now() - (%s * interval '1 second'))
                      {scope}
                    """,
                    (decay_seconds, decay_seconds, *tail),
                )
                if decayed.rowcount:
                    logger.info("记忆热度衰减 %s 条", decayed.rowcount)
            result = conn.execute(
                f"""
                DELETE FROM agent_episode
                WHERE COALESCE(last_seen_at, created_at)
                      < now() - (%s * interval '1 second')
                  AND COALESCE(hit_count, 0) < %s
                  {scope}
                """,
                (ttl_seconds, max(0, min_hits), *tail),
            )
            conn.commit()
            return result.rowcount or 0
