from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.storage.db import Database, MemoryItem, current_user_id
from ai_agent.storage.embeddings import embed_query, embed_texts
from ai_agent.memory.episode_merge import compact_episode_texts
from pgvector import Vector
from psycopg import Connection

logger = get_logger(__name__)


class EpisodeStore(Database):
    def write_episode(self, text: str, user_id: str | None = None) -> MemoryItem:
        uid = user_id or current_user_id()
        text = text.strip()
        src_vector = Vector(embed_texts([text])[0])
        # 锁事务只做相似检索；LLM 合并与 embedding 是网络调用，
        # 放在锁外，避免堵住同一用户的其他记忆写入
        with self._get_pool().connection() as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
                (uid, "agent_episode"),
            )
            hits = self._similar_episodes(
                conn,
                src_vector,
                uid,
                settings.memory_top_k,
                settings.memory_merge_threshold,
            )
            conn.commit()  # 事务结束即释放 advisory lock

        if not hits:
            item = MemoryItem(
                id=str(self._point_id("agent_episode", uid, text)),
                user_id=uid,
                text=text,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            with self._get_pool().connection() as conn:
                self._upsert_episode(conn, item, src_vector)
                conn.commit()
            return item

        keeper = hits[0]
        stale_ids = [uuid.UUID(hit.id) for hit in hits[1:]]
        # 网络/LLM 调用在锁外执行
        merged_text = compact_episode_texts([*(hit.text for hit in hits), text])
        merged_vector = (
            Vector(embed_texts([merged_text])[0])
            if merged_text != text
            else src_vector
        )
        now = datetime.now(timezone.utc).isoformat()
        with self._get_pool().connection() as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
                (uid, "agent_episode"),
            )
            # 拿到锁后复核 hits 未被并发写入改动；变了就走安全路径直接新增，
            # 避免用过期 hits 覆盖他人的合并结果
            current = self._episode_texts(conn, uid, [hit.id for hit in hits])
            if not all(current.get(hit.id) == hit.text for hit in hits):
                item = MemoryItem(
                    id=str(self._point_id("agent_episode", uid, text)),
                    user_id=uid,
                    text=text,
                    created_at=now,
                )
                self._upsert_episode(conn, item, src_vector)
                conn.commit()
                return item
            if stale_ids:
                conn.execute(
                    """
                    DELETE FROM agent_episode
                    WHERE user_id = %s AND id = ANY(%s)
                    """,
                    (uid, stale_ids),
                )
            if merged_text != keeper.text or stale_ids:
                conn.execute(
                    """
                    UPDATE agent_episode
                    SET text = %s, embedding = %s, created_at = %s
                    WHERE id = %s AND user_id = %s
                    """,
                    (
                        merged_text,
                        merged_vector,
                        now,
                        uuid.UUID(keeper.id),
                        uid,
                    ),
                )
                logger.info(
                    "merge_episode user_id=%s keeper=%s stale=%s text=%s",
                    uid,
                    keeper.id,
                    len(stale_ids),
                    merged_text,
                )
            conn.commit()
            return MemoryItem(
                id=keeper.id,
                user_id=uid,
                text=merged_text,
                created_at=keeper.created_at if merged_text == keeper.text else now,
            )

    def _episode_texts(
        self, conn: Connection, user_id: str, ids: list[str]
    ) -> dict[str, str]:
        rows = conn.execute(
            """
            SELECT id::text AS id, text
            FROM agent_episode
            WHERE user_id = %s AND id = ANY(%s)
            """,
            (user_id, [uuid.UUID(i) for i in ids]),
        ).fetchall()
        return {row["id"]: row["text"] for row in rows}

    def search_episode(
        self,
        query: str,
        k: int | None = None,
        user_id: str | None = None,
        threshold: float | None = None,
    ) -> list[MemoryItem]:
        uid = user_id or current_user_id()
        vector = Vector(embed_query(query))
        top_k = k or settings.memory_top_k
        cutoff = settings.memory_score_threshold if threshold is None else threshold
        with self._get_pool().connection() as conn:
            hits = self._similar_episodes(conn, vector, uid, top_k, cutoff)
        # 检索命中即登记观察证据、刷新 TTL 基准。只挂在 search_episode 上——
        # write_episode 内部为合并做的 _similar_episodes 不算「被记起」。
        # 失败只记日志，绝不影响召回结果。
        if hits:
            try:
                self.record_episode_hits([hit.id for hit in hits], user_id=uid)
            except Exception:
                logger.exception("情景记忆命中计数失败 user_id=%s", uid)
        return hits

    def _similar_episodes(
        self,
        conn: Connection,
        vector: Vector,
        user_id: str,
        k: int,
        threshold: float,
    ) -> list[MemoryItem]:
        ttl = settings.memory_episode_ttl_seconds
        sql = """
            SELECT
                id::text AS id,
                user_id,
                text,
                created_at::text AS created_at,
                COALESCE(last_seen_at, created_at)::text AS last_seen,
                COALESCE(hit_count, 0) AS hit_count,
                1 - (embedding <=> %s) AS score
            FROM agent_episode
            WHERE user_id = %s
              AND (%s <= 0 OR created_at >= now() - (%s * interval '1 second'))
              AND 1 - (embedding <=> %s) >= %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """
        # 命中续期写的是 last_seen_at，检索若仍只看 created_at，续期过的老记忆
        # 会留在库里却永远召不回，TTL 续期就成了空操作。
        sql = sql.replace(
            "OR created_at >=", "OR COALESCE(last_seen_at, created_at) >="
        )
        rows = conn.execute(
            sql, (vector, user_id, ttl, ttl, vector, threshold, vector, k)
        ).fetchall()
        return [
            MemoryItem(
                id=row["id"],
                user_id=row["user_id"],
                text=row["text"],
                created_at=row["created_at"],
                score=float(row["score"]),
                last_seen=row.get("last_seen"),
                hits=int(row.get("hit_count") or 0),
            )
            for row in rows
        ]

    def cleanup_expired_episodes(
        self, ttl_seconds: int | None = None, user_id: str | None = None
    ) -> int:
        """删掉超过 TTL 的情景记忆。0 / 负数 = 关闭，不删。"""
        ttl = (
            settings.memory_episode_ttl_seconds if ttl_seconds is None else ttl_seconds
        )
        if ttl <= 0:
            return 0
        # 命中续期 + 高热度保留 + 长期未命中降权；失败返回 0（宁可漏删，
        # 不误删刚被反复用到的记忆）。
        try:
            return self.decay_and_cleanup_episodes(
                ttl_seconds=ttl,
                decay_seconds=settings.memory_promotion_decay_seconds,
                min_hits=settings.memory_promotion_min_hits,
                user_id=user_id,
            )
        except Exception:
            logger.exception("渐进式情景记忆清理失败，本轮不删除")
            return 0

    def _upsert_episode(
        self, conn: Connection, item: MemoryItem, vector: Vector
    ) -> None:
        conn.execute(
            """
            INSERT INTO agent_episode (id, user_id, text, embedding, created_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                created_at = EXCLUDED.created_at
            """,
            (item.id, item.user_id, item.text, vector, item.created_at),
        )
