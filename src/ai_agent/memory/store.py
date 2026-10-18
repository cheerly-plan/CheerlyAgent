from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.memory.embeddings import embed_query, embed_texts
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = get_logger(__name__)
_current_user_id: ContextVar[str] = ContextVar("current_user_id", default="default")


@contextmanager
def user_scope(user_id: str):
    token = _current_user_id.set(user_id)
    try:
        yield
    finally:
        _current_user_id.reset(token)


def current_user_id() -> str:
    return _current_user_id.get()


@dataclass
class MemoryItem:
    id: str
    user_id: str
    text: str
    created_at: str
    score: float = 0.0


class MemoryStore:
    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = database_url or settings.database_url
        self._pool: ConnectionPool | None = None

    def _configure_connection(self, conn: Connection) -> None:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)

    def _get_pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(
                conninfo=self._database_url,
                min_size=1,
                max_size=8,
                kwargs={"row_factory": dict_row},
                configure=self._configure_connection,
                open=True,
            )
        return self._pool

    def ensure_schema(self) -> None:
        dim = int(settings.embedding_dim)
        if dim <= 0 or dim > 20000:
            raise ValueError("EMBEDDING_DIM 非法")
        with self._get_pool().connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector({dim}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_memory_user_id_idx
                ON agent_memory (user_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_memory_embedding_hnsw
                ON agent_memory USING hnsw (embedding vector_cosine_ops)
                """
            )
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS agent_profile (
                        id UUID PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_profile_user_id_idx
                ON agent_profile (user_id)
                """
            )
            conn.commit()

    def _point_id(self, table: str, user_id: str, text: str) -> uuid.UUID:
        return uuid.uuid5(uuid.NAMESPACE_URL, f"{table}:{user_id}:{text.strip()}")

    def write(self, text: str, user_id: str | None = None) -> MemoryItem:
        uid = user_id or current_user_id()
        text = text.strip()
        item = MemoryItem(
            id=str(self._point_id("agent_memory", uid, text)),
            user_id=uid,
            text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        vector = Vector(embed_texts([text])[0])
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_memory (id, user_id, text, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    created_at = EXCLUDED.created_at
                """,
                (item.id, item.user_id, item.text, vector, item.created_at),
            )
            conn.commit()
        return item

    def search(
        self,
        query: str,
        k: int | None = None,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        uid = user_id or current_user_id()
        vector = Vector(embed_query(query))
        top_k = k or settings.memory_top_k
        threshold = settings.memory_score_threshold
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    id::text AS id,
                    user_id,
                    text,
                    created_at::text AS created_at,
                    1 - (embedding <=> %s) AS score
                FROM agent_memory
                WHERE user_id = %s
                  AND 1 - (embedding <=> %s) >= %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (vector, uid, vector, threshold, vector, top_k),
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

    def write_profile(self, text: str, user_id: str | None = None) -> MemoryItem:
        uid = user_id or current_user_id()
        text = text.strip()
        item = MemoryItem(
            id=str(self._point_id("agent_profile", uid, text)),
            user_id=uid,
            text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_profile (id, user_id, text, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    created_at = EXCLUDED.created_at
                """,
                (item.id, item.user_id, item.text, item.created_at),
            )
            conn.commit()
        logger.info("write_profile user_id=%s text=%s", item.user_id, item.text)
        return item

    def list_profile(
        self, user_id: str | None = None, limit: int = 20
    ) -> list[MemoryItem]:
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT id::text AS id, user_id, text, created_at::text AS created_at
                FROM agent_profile
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (uid, limit),
            ).fetchall()
        return [
            MemoryItem(
                id=row["id"],
                user_id=row["user_id"],
                text=row["text"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def clear(self, user_id: str | None = None) -> None:
        with self._get_pool().connection() as conn:
            if user_id:
                conn.execute("DELETE FROM agent_memory WHERE user_id = %s", (user_id,))
                conn.execute("DELETE FROM agent_profile WHERE user_id = %s", (user_id,))
            else:
                conn.execute("DELETE FROM agent_memory")
                conn.execute("DELETE FROM agent_profile")
            conn.commit()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None


store = MemoryStore()
