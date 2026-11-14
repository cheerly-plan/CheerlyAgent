from __future__ import annotations

import atexit
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from ai_agent.storage._schema.auth import (
    ensure_user,
    ensure_auth_token,
)
from ai_agent.storage._schema.chat import (
    ensure_chat_history,
    ensure_tool_result,
    ensure_session,
    ensure_checkpoint,
)
from ai_agent.storage._schema.core import (
    ensure_vector_extension,
    commit_schema,
)
from ai_agent.storage._schema.kb import (
    ensure_kb_file,
    ensure_kb_chunk,
    ensure_kb_file_more1,
    ensure_kb_chunk_more1,
    ensure_kb_file_more2,
)
from ai_agent.storage._schema.kg import (
    ensure_kg_entity,
    ensure_kg_relation,
    ensure_kg_mention,
    ensure_kg_alias,
)
from ai_agent.storage._schema.memory import (
    ensure_episode,
    ensure_user_profile,
    ensure_episode_more1,
    ensure_memory_archive,
    ensure_memory_setting,
)

from ai_agent.core.config import settings
from ai_agent.llm.types import ChatMessage
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    # 最近一次被检索命中的时间（episode 用；为空时回落到 created_at）
    last_seen: str | None = None
    # 命中次数（episode 的观察证据；画像恒为 0）
    hits: int = 0


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
    # 用户级记忆开关；读取失败/旧库无设置行时默认开启
    memory_enabled: bool = True


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
        """建表/补列（幂等）。

        DDL 按表拆在 ``storage/_schema`` 里：每张表一个函数，**调用顺序即
        原实现里的执行顺序**，SQL 逐字未改。这里只负责校验维度、取连接、
        按序调用与提交。

        规范：新增/修改表结构请写 Alembic 迁移，不要往这里加 ALTER TABLE
        （见 README「数据库与迁移」）；本方法作为回滚锚点保留。
        """
        dim = int(settings.embedding_dim)
        if dim <= 0 or dim > 20000:
            raise ValueError("EMBEDDING_DIM 非法")
        with self._get_pool().connection() as conn:
            ensure_vector_extension(conn, dim=dim)
            ensure_episode(conn, dim=dim)
            ensure_user_profile(conn, dim=dim)
            ensure_chat_history(conn, dim=dim)
            ensure_tool_result(conn, dim=dim)
            ensure_session(conn, dim=dim)
            ensure_user(conn, dim=dim)
            ensure_auth_token(conn, dim=dim)
            ensure_checkpoint(conn, dim=dim)
            ensure_kb_file(conn, dim=dim)
            ensure_kb_chunk(conn, dim=dim)
            ensure_kg_entity(conn, dim=dim)
            ensure_kg_relation(conn, dim=dim)
            ensure_kg_mention(conn, dim=dim)
            ensure_kg_alias(conn, dim=dim)
            ensure_kb_file_more1(conn, dim=dim)
            ensure_kb_chunk_more1(conn, dim=dim)
            ensure_kb_file_more2(conn, dim=dim)
            ensure_episode_more1(conn, dim=dim)
            ensure_memory_archive(conn, dim=dim)
            ensure_memory_setting(conn, dim=dim)
            commit_schema(conn, dim=dim)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None


def run_migrations() -> None:
    """通过 Alembic 执行 schema 迁移到最新版本。

    替代 ensure_schema()：schema 变更走版本化迁移，可追溯、可回滚。
    连接串由 env.py 从 settings.database_url 读取（与运行时同源）。
    现有库若已由 ensure_schema 建好，alembic 会自动建 alembic_version
    表并跑 baseline（IF NOT EXISTS 幂等，不会破坏既有数据）；也可预先
    手动 `alembic stamp head` 标记后跳过 baseline。

    ensure_schema() 保留作回滚锚点：万一迁移出问题，把 lifespan 改回
    调 store.ensure_schema() 即可恢复旧行为。

    用独立子进程跑 upgrade：uvicorn --reload 的 fork worker 里同进程调
    alembic/sqlalchemy 可能卡死（表现为启动停在 transactional DDL、登录超时），
    子进程与 CLI `alembic upgrade head` 同路径，可避开 fork 锁问题。
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=_PROJECT_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade head 失败 (exit={result.returncode})"
        )
