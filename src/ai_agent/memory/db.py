from __future__ import annotations

import atexit
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
        self._atexit_hooked = False

    def _configure_connection(self, conn: Connection) -> None:
        # 扩展创建放在 ensure_schema 里做一次即可；
        # 每条新连接都执行 CREATE EXTENSION 既慢又要求高权限
        register_vector(conn)

    def _get_pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(
                conninfo=self._database_url,
                min_size=1,
                max_size=settings.db_pool_max_size,
                kwargs={"row_factory": dict_row},
                configure=self._configure_connection,
                open=True,
            )
            # 脚本/一次性进程没人调 close 时，ConnectionPool 会走 __del__，
            # 那里只能限时等 scheduler 线程退出，等不到就打
            # "couldn't stop thread 'pool-1-scheduler' within 5.0 seconds"。
            # 服务的 lifespan 已显式 close，这里只是兜底，close 幂等。
            if not self._atexit_hooked:
                atexit.register(self.close)
                self._atexit_hooked = True
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
            # 历史遗留：全 facet 唯一索引会把多条独立偏好挤成一行，
            # 只保留互斥型 facet 的单行约束，其余 facet 允许多条共存
            conn.execute(
                "DROP INDEX IF EXISTS agent_user_profile_user_facet_uidx"
            )
            conn.execute(
                """
                DELETE FROM agent_user_profile a
                WHERE a.facet IN ('name', 'language', 'locale', 'occupation')
                  AND a.ctid NOT IN (
                    SELECT DISTINCT ON (user_id, facet) ctid
                    FROM agent_user_profile
                    WHERE facet IN ('name', 'language', 'locale', 'occupation')
                    ORDER BY user_id, facet, created_at DESC
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS agent_user_profile_mutex_uidx
                ON agent_user_profile (user_id, facet)
                WHERE facet IN ('name', 'language', 'locale', 'occupation')
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
            # 存量库迁移：消息级附属信息（目前是知识库引用 sources）。
            # 单独一列而不是塞进 content：正文要原样回放给模型，附件信息
            # 只用于前端渲染，混在一起会让历史消息和模型输入失真
            conn.execute(
                """
                ALTER TABLE agent_chat_history
                ADD COLUMN IF NOT EXISTS meta JSONB
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_checkpoint (
                    run_id          UUID PRIMARY KEY,
                    user_id         TEXT NOT NULL,
                    chat_session_id TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'running',
                    step            INT NOT NULL DEFAULT 0,
                    messages        JSONB NOT NULL,
                    use_web_search  BOOLEAN NOT NULL DEFAULT TRUE,
                    use_kb          BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # 存量库迁移：补建 use_web_search 列（断点续跑时沿用原 run 的开关状态）
            conn.execute(
                """
                ALTER TABLE agent_checkpoint
                ADD COLUMN IF NOT EXISTS use_web_search BOOLEAN NOT NULL DEFAULT TRUE
                """
            )
            # 同上：知识库工具开关（工具是否挂载要跟原 run 保持一致）
            conn.execute(
                """
                ALTER TABLE agent_checkpoint
                ADD COLUMN IF NOT EXISTS use_kb BOOLEAN NOT NULL DEFAULT TRUE
                """
            )
            # 启动时不存在本进程的进行中 run（单进程部署）：残留的 running
            # 一律是上次进程异常退出留下的僵尸，标为 interrupted（进程被杀
            # 属于中断而非失败，两者均可被 /chat/resume 认领续跑），
            # 否则既无法恢复又会触发 session 互斥把会话锁死
            conn.execute(
                """
                UPDATE agent_checkpoint
                SET status = 'interrupted', updated_at = now()
                WHERE status = 'running'
                """
            )
            # DB 级互斥兜底：同会话同时最多一个 running checkpoint，
            # 路由层预检（has_running_checkpoint）之后的并发竞态由它裁决
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS agent_checkpoint_running_uidx
                ON agent_checkpoint (user_id, chat_session_id)
                WHERE status = 'running'
                """
            )
            # 知识库：元数据在 agent_kb_file，切片在 agent_kb_chunk
            # （外键级联删除，删文件即清空切片）
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_kb_file (
                    id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    size_bytes BIGINT NOT NULL,
                    mime TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'uploaded',
                    error TEXT NOT NULL DEFAULT '',
                    chunk_count INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_kb_file_user_id_idx
                ON agent_kb_file (user_id)
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS agent_kb_chunk (
                    id UUID PRIMARY KEY,
                    file_id UUID NOT NULL REFERENCES agent_kb_file(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL,
                    seq INT NOT NULL,
                    page INT,
                    kind TEXT NOT NULL DEFAULT 'text',
                    text TEXT NOT NULL,
                    embedding vector({dim}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (file_id, seq)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_kb_chunk_embedding_hnsw
                ON agent_kb_chunk USING hnsw (embedding vector_cosine_ops)
                """
            )
            # 既有库的增量列：CREATE IF NOT EXISTS 不会补列，靠 ALTER 兜底
            conn.execute(
                """
                ALTER TABLE agent_kb_chunk
                ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'text'
                """
            )
            # 知识图谱（P2）：实体 / 关系 / 提及三层。
            # 全部挂在文件上做外键级联，删文件即清空图谱，不留脏数据；
            # embedding 维度跟随 EMBEDDING_DIM（与切片向量同模型），供实体语义检索。
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS agent_kg_entity (
                    id UUID PRIMARY KEY,
                    file_id UUID NOT NULL REFERENCES agent_kb_file(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    norm_name TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'other',
                    description TEXT NOT NULL DEFAULT '',
                    embedding vector({dim}),
                    mention_count INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (file_id, norm_name, type)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_kg_entity_file_idx
                ON agent_kg_entity (file_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_kg_entity_norm_idx
                ON agent_kg_entity (norm_name)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_kg_entity_embedding_hnsw
                ON agent_kg_entity USING hnsw (embedding vector_cosine_ops)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_kg_relation (
                    id UUID PRIMARY KEY,
                    file_id UUID NOT NULL REFERENCES agent_kb_file(id) ON DELETE CASCADE,
                    source_id UUID NOT NULL REFERENCES agent_kg_entity(id) ON DELETE CASCADE,
                    target_id UUID NOT NULL REFERENCES agent_kg_entity(id) ON DELETE CASCADE,
                    relation TEXT NOT NULL DEFAULT '相关',
                    description TEXT NOT NULL DEFAULT '',
                    weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                    chunk_seq INT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_kg_relation_file_idx
                ON agent_kg_relation (file_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_kg_relation_source_idx
                ON agent_kg_relation (source_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_kg_relation_target_idx
                ON agent_kg_relation (target_id)
                """
            )
            # 提及表：实体出现在哪些切片，供前端「点实体 → 回溯原文」
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_kg_mention (
                    entity_id UUID NOT NULL REFERENCES agent_kg_entity(id) ON DELETE CASCADE,
                    file_id UUID NOT NULL REFERENCES agent_kb_file(id) ON DELETE CASCADE,
                    chunk_seq INT NOT NULL,
                    hits INT NOT NULL DEFAULT 1,
                    PRIMARY KEY (entity_id, chunk_seq)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_kg_mention_file_idx
                ON agent_kg_mention (file_id)
                """
            )
            # 文件表的图谱进度：'' 未构建 / building 构建中 / ready / stale 切片修订后待重建 / failed
            conn.execute(
                """
                ALTER TABLE agent_kb_file
                ADD COLUMN IF NOT EXISTS graph_status TEXT NOT NULL DEFAULT ''
                """
            )
            conn.execute(
                """
                ALTER TABLE agent_kb_file
                ADD COLUMN IF NOT EXISTS graph_entity_count INT NOT NULL DEFAULT 0
                """
            )
            # 管理能力：标签（读侧过滤/展示）与内容指纹（上传去重）
            conn.execute(
                """
                ALTER TABLE agent_kb_file
                ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT '{}'
                """
            )
            conn.execute(
                """
                ALTER TABLE agent_kb_file
                ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT ''
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_kb_file_hash_idx
                ON agent_kb_file (content_hash)
                """
            )
            conn.commit()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
