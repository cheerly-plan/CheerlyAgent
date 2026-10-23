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


@dataclass
class UserRecord:
    user_id: str  # 服务端生成的 UUID，不可变，业务数据以它为键
    username: str  # 登录名，仅用于认证
    display_name: str
    created_at: str = ""


class Database:
    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = database_url or settings.database_url
        self._pool: ConnectionPool | None = None

    def _configure_connection(self, conn: Connection) -> None:
        # 扩展创建放在 ensure_schema 里做一次即可；
        # 每条新连接都执行 CREATE EXTENSION 既慢又要求高权限
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
            # 迁移：旧版 agent_user 用用户名当主键，改为服务端生成的 UUID 主键，
            # 用户名降级为登录名唯一索引，业务数据（会话/历史/记忆）统一重映射到新 id
            conn.execute(
                """
                DO $$
                DECLARE
                    has_old_user_id boolean;
                    has_id boolean;
                BEGIN
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'agent_user' AND column_name = 'user_id'
                    ) INTO has_old_user_id;
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'agent_user' AND column_name = 'id'
                    ) INTO has_id;

                    IF has_old_user_id AND NOT has_id THEN
                        ALTER TABLE agent_user RENAME COLUMN user_id TO username;
                        ALTER TABLE agent_user ADD COLUMN id UUID;
                        UPDATE agent_user SET id = gen_random_uuid();
                        ALTER TABLE agent_user ALTER COLUMN id SET NOT NULL;
                        ALTER TABLE agent_user DROP CONSTRAINT agent_user_pkey;
                        ALTER TABLE agent_user ADD PRIMARY KEY (id);
                        -- 登录令牌是瞬态数据，迁移期间直接清空后换列类型并挂外键
                        DELETE FROM agent_auth_token;
                        ALTER TABLE agent_auth_token ALTER COLUMN user_id TYPE UUID
                            USING user_id::uuid;
                        ALTER TABLE agent_auth_token ADD CONSTRAINT agent_auth_token_user_fk
                            FOREIGN KEY (user_id) REFERENCES agent_user(id) ON DELETE CASCADE;
                        -- 历史 user_id 存的是用户名，重映射到新生成的 id
                        UPDATE agent_session s SET user_id = u.id::text
                            FROM agent_user u WHERE s.user_id = u.username;
                        UPDATE agent_chat_history h SET user_id = u.id::text
                            FROM agent_user u WHERE h.user_id = u.username;
                        UPDATE agent_episode e SET user_id = u.id::text
                            FROM agent_user u WHERE e.user_id = u.username;
                        UPDATE agent_user_profile p SET user_id = u.id::text
                            FROM agent_user u WHERE p.user_id = u.username;
                    END IF;
                END $$;
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_user (
                    id UUID PRIMARY KEY,
                    username TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS agent_user_username_uidx
                ON agent_user (username)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_auth_token (
                    token_hash TEXT PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES agent_user(id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    expires_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_auth_token_user_id_idx
                ON agent_auth_token (user_id)
                """
            )
            conn.commit()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
