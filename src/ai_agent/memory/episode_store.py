from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.memory.db import Database, MemoryItem, current_user_id
from ai_agent.memory.embeddings import embed_query, embed_texts
from ai_agent.memory.episode import compact_episode_texts
from pgvector import Vector
from psycopg import Connection

logger = get_logger(__name__)


class EpisodeStore(Database):
    def write_episode(self, text: str, user_id: str | None = None) -> MemoryItem:
        uid = user_id or current_user_id()
        text = text.strip()
        src_vector = Vector(embed_texts([text])[0])
        hits = self._find_similar_episodes(src_vector, uid)
        merged_text = (
            compact_episode_texts([*(hit.text for hit in hits), text]) if hits else text
        )
        merged_vector = (
            Vector(embed_texts([merged_text])[0]) if merged_text != text else src_vector
        )
        now = datetime.now(timezone.utc).isoformat()
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
            if not hits:
                item = MemoryItem(
                    id=str(self._point_id("agent_episode", uid, merged_text)),
                    user_id=uid,
                    text=merged_text,
                    created_at=now,
                )
                self._upsert_episode(conn, item, merged_vector)
                conn.commit()
                return item

            keeper = hits[0]
            stale_ids = [uuid.UUID(hit.id) for hit in hits[1:]]
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
                keeper = MemoryItem(
                    id=keeper.id,
                    user_id=uid,
                    text=merged_text,
                    created_at=now,
                )
            conn.commit()
            return MemoryItem(
                id=keeper.id,
                user_id=uid,
                text=merged_text,
                created_at=keeper.created_at,
            )

    def _find_similar_episodes(self, vector: Vector, user_id: str) -> list[MemoryItem]:
        with self._get_pool().connection() as conn:
            return self._similar_episodes(
                conn,
                vector,
                user_id,
                settings.memory_top_k,
                settings.memory_merge_threshold,
            )

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
            return self._similar_episodes(conn, vector, uid, top_k, cutoff)

    def _similar_episodes(
        self,
        conn: Connection,
        vector: Vector,
        user_id: str,
        k: int,
        threshold: float,
    ) -> list[MemoryItem]:
        rows = conn.execute(
            """
            SELECT
                id::text AS id,
                user_id,
                text,
                created_at::text AS created_at,
                1 - (embedding <=> %s) AS score
            FROM agent_episode
            WHERE user_id = %s
              AND 1 - (embedding <=> %s) >= %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (vector, user_id, vector, threshold, vector, k),
        ).fetchall()
        return [
            MemoryItem(
                id=row["id"],
                user_id=row["user_id"],
                text=row["text"],
                created_at=row["created_at"],
                score=float(row["score"]),
            )
            for row in rows
        ]

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
