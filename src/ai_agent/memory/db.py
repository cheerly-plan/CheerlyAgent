from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from ai_agent.core.config import settings
from ai_agent.llm.types import ChatMessage
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_current_user_id: ContextVar[str] = ContextVar("current_user_id", default="default")
DEFAULT_SESSION_TITLE = "新会话"


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
    facet: str = ""


@dataclass
class ChatHistoryPage:
    messages: list[ChatMessage]
    has_more: bool
    next_before_seq: int | None = None


@dataclass
class SessionRecord:
    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


class Database:
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

    def _point_id(self, table: str, user_id: str, text: str) -> uuid.UUID:
        return uuid.uuid5(uuid.NAMESPACE_URL, f"{table}:{user_id}:{text.strip()}")

    def ensure_schema(self) -> None:
        dim = int(settings.embedding_dim)
        if dim <= 0 or dim > 20000:
            raise ValueError("EMBEDDING_DIM 非法")
        with self._get_pool().connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS agent_episode (
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
                CREATE INDEX IF NOT EXISTS agent_episode_user_id_idx
                ON agent_episode (user_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_episode_embedding_hnsw
                ON agent_episode USING hnsw (embedding vector_cosine_ops)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_user_profile (
                    id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    facet TEXT NOT NULL DEFAULT 'other',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                ALTER TABLE agent_user_profile
                ADD COLUMN IF NOT EXISTS facet TEXT NOT NULL DEFAULT 'other'
                """
            )
            conn.execute(
                """
                DELETE FROM agent_user_profile a
                WHERE a.ctid NOT IN (
                    SELECT DISTINCT ON (user_id, facet) ctid
                    FROM agent_user_profile
                    ORDER BY user_id, facet, created_at DESC
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS agent_user_profile_user_facet_uidx
                ON agent_user_profile (user_id, facet)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_user_profile_user_id_idx
                ON agent_user_profile (user_id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_chat_history (
                    id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    chat_session_id TEXT NOT NULL,
                    seq INT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    tool_calls JSONB,
                    tool_call_id TEXT,
                    name TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (user_id, chat_session_id, seq)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_session (
                    id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '新会话',
                    summary TEXT NOT NULL DEFAULT '',
                    idempotency_key TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    deleted_at TIMESTAMPTZ,
                    UNIQUE (user_id, idempotency_key)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_session_user_updated_idx
                ON agent_session (user_id, updated_at DESC)
                WHERE deleted_at IS NULL
                """
            )
            conn.execute(
                """
                ALTER TABLE agent_session
                ALTER COLUMN title SET DEFAULT '新会话'
                """
            )
            conn.execute(
                """
                ALTER TABLE agent_session
                ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT ''
                """
            )
            conn.commit()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
