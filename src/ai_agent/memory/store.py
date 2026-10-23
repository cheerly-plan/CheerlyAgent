from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.compact import compact_episode_texts
from ai_agent.memory.embeddings import embed_query, embed_texts
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

logger = get_logger(__name__)
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


def _chat_message_from_row(row: dict) -> ChatMessage:
    msg: ChatMessage = {"role": row["role"], "content": row["content"] or ""}
    if row["tool_calls"]:
        msg["tool_calls"] = row["tool_calls"]
    if row["tool_call_id"]:
        msg["tool_call_id"] = row["tool_call_id"]
    if row["name"]:
        msg["name"] = row["name"]
    return msg


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
            # 情景记忆（向量检索）
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
            # 用户画像
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_user_profile (
                    id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_user_profile_user_id_idx
                ON agent_user_profile (user_id)
                """
            )
            # 会话对话历史
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

    def _point_id(self, table: str, user_id: str, text: str) -> uuid.UUID:
        return uuid.uuid5(uuid.NAMESPACE_URL, f"{table}:{user_id}:{text.strip()}")

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

    def write_profile(self, text: str, user_id: str | None = None) -> MemoryItem:
        uid = user_id or current_user_id()
        text = text.strip()
        item = MemoryItem(
            id=str(self._point_id("agent_user_profile", uid, text)),
            user_id=uid,
            text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_user_profile (id, user_id, text, created_at)
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
                FROM agent_user_profile
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

    def append_chat(
        self, user_id: str, chat_session_id: str, messages: list[ChatMessage]
    ) -> None:
        rows = [m for m in messages if m.get("role") != "system"]
        if not rows:
            return
        values = [
            (
                str(uuid.uuid4()),
                user_id,
                chat_session_id,
                msg["role"],
                msg.get("content") or "",
                Jsonb(msg["tool_calls"]) if msg.get("tool_calls") else None,
                msg.get("tool_call_id"),
                msg.get("name"),
            )
            for msg in rows
        ]
        with self._get_pool().connection() as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
                (user_id, chat_session_id),
            )
            last = conn.execute(
                """
                SELECT COALESCE(MAX(seq), 0) AS seq
                FROM agent_chat_history
                WHERE user_id = %s AND chat_session_id = %s
                """,
                (user_id, chat_session_id),
            ).fetchone()
            start = int(last["seq"]) if last else 0
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO agent_chat_history
                        (id, user_id, chat_session_id, seq, role, content,
                         tool_calls, tool_call_id, name)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            mid,
                            uid,
                            sid,
                            start + offset,
                            role,
                            content,
                            tool_calls,
                            tool_call_id,
                            name,
                        )
                        for offset, (
                            mid,
                            uid,
                            sid,
                            role,
                            content,
                            tool_calls,
                            tool_call_id,
                            name,
                        ) in enumerate(values, start=1)
                    ],
                )
            conn.execute(
                """
                UPDATE agent_session
                SET updated_at = now()
                WHERE user_id = %s AND id::text = %s AND deleted_at IS NULL
                """,
                (user_id, chat_session_id),
            )
            conn.commit()

    def load_chat(
        self,
        user_id: str,
        chat_session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        if limit is not None and limit <= 0:
            return []
        if limit is None:
            sql = """
                SELECT role, content, tool_calls, tool_call_id, name
                FROM agent_chat_history
                WHERE user_id = %s AND chat_session_id = %s
                ORDER BY seq
                """
            params: tuple = (user_id, chat_session_id)
        else:
            sql = """
                SELECT role, content, tool_calls, tool_call_id, name
                FROM (
                    SELECT role, content, tool_calls, tool_call_id, name, seq
                    FROM agent_chat_history
                    WHERE user_id = %s AND chat_session_id = %s
                    ORDER BY seq DESC
                    LIMIT %s
                ) recent
                ORDER BY seq
                """
            params = (user_id, chat_session_id, limit)
        with self._get_pool().connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_chat_message_from_row(row) for row in rows]

    def load_chat_page(
        self,
        user_id: str,
        chat_session_id: str,
        limit: int = 20,
        before_seq: int | None = None,
    ) -> ChatHistoryPage:
        if limit <= 0:
            return ChatHistoryPage([], False, None)
        page_size = min(limit, 100)
        with self._get_pool().connection() as conn:
            if before_seq is None:
                rows = conn.execute(
                    """
                    SELECT seq, role, content, tool_calls, tool_call_id, name
                    FROM agent_chat_history
                    WHERE user_id = %s AND chat_session_id = %s
                    ORDER BY seq DESC
                    LIMIT %s
                    """,
                    (user_id, chat_session_id, page_size),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT seq, role, content, tool_calls, tool_call_id, name
                    FROM agent_chat_history
                    WHERE user_id = %s AND chat_session_id = %s
                      AND seq < %s
                    ORDER BY seq DESC
                    LIMIT %s
                    """,
                    (user_id, chat_session_id, before_seq, page_size),
                ).fetchall()
            if not rows:
                return ChatHistoryPage([], False, None)

            # 页首若落在 tool 链中间，向前补到本轮 user，避免拆断调用链
            oldest = rows[-1]
            if oldest["role"] != "user":
                extra = conn.execute(
                    """
                    SELECT seq, role, content, tool_calls, tool_call_id, name
                    FROM agent_chat_history
                    WHERE user_id = %s AND chat_session_id = %s
                      AND seq < %s
                    ORDER BY seq DESC
                    LIMIT 32
                    """,
                    (user_id, chat_session_id, oldest["seq"]),
                ).fetchall()
                prefix: list[dict] = []
                for row in extra:
                    prefix.append(row)
                    if row["role"] == "user":
                        break
                rows = [*rows, *prefix]

            min_seq = rows[-1]["seq"]
            older = conn.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM agent_chat_history
                    WHERE user_id = %s AND chat_session_id = %s AND seq < %s
                ) AS has_more
                """,
                (user_id, chat_session_id, min_seq),
            ).fetchone()
            has_more = bool(older["has_more"]) if older else False

        chronological = list(reversed(rows))
        return ChatHistoryPage(
            messages=[_chat_message_from_row(row) for row in chronological],
            has_more=has_more,
            next_before_seq=int(chronological[0]["seq"]) if has_more else None,
        )

    def load_recent_turns(
        self, user_id: str, chat_session_id: str, turns: int
    ) -> list[ChatMessage]:
        if turns <= 0:
            return []
        with self._get_pool().connection() as conn:
            start = conn.execute(
                """
                SELECT MIN(seq) AS start_seq
                FROM (
                    SELECT seq
                    FROM agent_chat_history
                    WHERE user_id = %s AND chat_session_id = %s AND role = 'user'
                    ORDER BY seq DESC
                    LIMIT %s
                ) recent_users
                """,
                (user_id, chat_session_id, turns),
            ).fetchone()
            start_seq = start["start_seq"] if start else None
            if start_seq is None:
                return []
            rows = conn.execute(
                """
                SELECT role, content, tool_calls, tool_call_id, name
                FROM agent_chat_history
                WHERE user_id = %s AND chat_session_id = %s AND seq >= %s
                ORDER BY seq
                """,
                (user_id, chat_session_id, start_seq),
            ).fetchall()
        return [_chat_message_from_row(row) for row in rows]

    def get_session_summary(self, user_id: str, session_id: str) -> str:
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT summary
                FROM agent_session
                WHERE user_id = %s AND id::text = %s AND deleted_at IS NULL
                """,
                (user_id, session_id),
            ).fetchone()
        return (row["summary"] or "").strip() if row else ""

    def update_session_summary(
        self, user_id: str, session_id: str, summary: str
    ) -> None:
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                UPDATE agent_session
                SET summary = %s
                WHERE user_id = %s AND id::text = %s AND deleted_at IS NULL
                """,
                (summary, user_id, session_id),
            )
            conn.commit()

    def clear_long_term(self, user_id: str | None = None) -> None:
        with self._get_pool().connection() as conn:
            if user_id:
                conn.execute("DELETE FROM agent_episode WHERE user_id = %s", (user_id,))
                conn.execute(
                    "DELETE FROM agent_user_profile WHERE user_id = %s", (user_id,)
                )
            else:
                conn.execute("DELETE FROM agent_episode")
                conn.execute("DELETE FROM agent_user_profile")
            conn.commit()

    def clear_chat(self, user_id: str, chat_session_id: str | None = None) -> None:
        with self._get_pool().connection() as conn:
            if chat_session_id:
                conn.execute(
                    """
                    DELETE FROM agent_chat_history
                    WHERE user_id = %s AND chat_session_id = %s
                    """,
                    (user_id, chat_session_id),
                )
                conn.execute(
                    """
                    UPDATE agent_session
                    SET summary = ''
                    WHERE user_id = %s AND id::text = %s
                    """,
                    (user_id, chat_session_id),
                )
            else:
                conn.execute(
                    "DELETE FROM agent_chat_history WHERE user_id = %s",
                    (user_id,),
                )
                conn.execute(
                    "UPDATE agent_session SET summary = '' WHERE user_id = %s",
                    (user_id,),
                )
            conn.commit()

    def create_session(self, user_id: str, idempotency_key: str) -> SessionRecord:
        session_id = str(uuid.uuid4())
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                INSERT INTO agent_session (id, user_id, title, idempotency_key)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, idempotency_key) DO UPDATE
                    SET id = agent_session.id
                RETURNING
                    id::text AS id,
                    user_id,
                    title,
                    created_at::text AS created_at,
                    updated_at::text AS updated_at
                """,
                (session_id, user_id, DEFAULT_SESSION_TITLE, idempotency_key),
            ).fetchone()
            conn.commit()
        if row is None:
            raise RuntimeError("创建会话失败")
        return _session_from_row(row)

    def get_session(self, user_id: str, session_id: str) -> SessionRecord | None:
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT
                    id::text AS id,
                    user_id,
                    title,
                    created_at::text AS created_at,
                    updated_at::text AS updated_at
                FROM agent_session
                WHERE user_id = %s AND id::text = %s AND deleted_at IS NULL
                """,
                (user_id, session_id),
            ).fetchone()
        return _session_from_row(row) if row else None

    def list_sessions(self, user_id: str, limit: int = 100) -> list[SessionRecord]:
        page_size = min(max(limit, 1), 200)
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    id::text AS id,
                    user_id,
                    title,
                    created_at::text AS created_at,
                    updated_at::text AS updated_at
                FROM agent_session
                WHERE user_id = %s AND deleted_at IS NULL
                ORDER BY updated_at DESC, created_at DESC
                LIMIT %s
                """,
                (user_id, page_size),
            ).fetchall()
        return [_session_from_row(row) for row in rows]

    def update_session_title(
        self,
        user_id: str,
        session_id: str,
        title: str,
        only_if_default: bool = True,
    ) -> str | None:
        title = title.strip()
        if not title:
            return None
        with self._get_pool().connection() as conn:
            if only_if_default:
                row = conn.execute(
                    """
                    UPDATE agent_session
                    SET title = %s, updated_at = now()
                    WHERE user_id = %s
                      AND id::text = %s
                      AND deleted_at IS NULL
                      AND title = %s
                    RETURNING title
                    """,
                    (title, user_id, session_id, DEFAULT_SESSION_TITLE),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    UPDATE agent_session
                    SET title = %s, updated_at = now()
                    WHERE user_id = %s AND id::text = %s AND deleted_at IS NULL
                    RETURNING title
                    """,
                    (title, user_id, session_id),
                ).fetchone()
            conn.commit()
        return row["title"] if row else None

    def delete_session(self, user_id: str, session_id: str) -> bool:
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                UPDATE agent_session
                SET deleted_at = now()
                WHERE user_id = %s AND id::text = %s AND deleted_at IS NULL
                RETURNING id
                """,
                (user_id, session_id),
            ).fetchone()
            if row is None:
                conn.commit()
                return False
            conn.execute(
                """
                DELETE FROM agent_chat_history
                WHERE user_id = %s AND chat_session_id = %s
                """,
                (user_id, session_id),
            )
            conn.commit()
        return True

    def cleanup_empty_unnamed_sessions(self, max_age_seconds: int = 3600) -> int:
        age = max(int(max_age_seconds), 0)
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                DELETE FROM agent_session s
                WHERE s.title = %s
                  AND s.deleted_at IS NULL
                  AND s.created_at <= now() - (%s * interval '1 second')
                  AND NOT EXISTS (
                    SELECT 1
                    FROM agent_chat_history h
                    WHERE h.user_id = s.user_id
                      AND h.chat_session_id = s.id::text
                  )
                RETURNING s.id
                """,
                (DEFAULT_SESSION_TITLE, age),
            ).fetchall()
            conn.commit()
        return len(rows)

    def clear_sessions(self, user_id: str) -> None:
        with self._get_pool().connection() as conn:
            conn.execute("DELETE FROM agent_session WHERE user_id = %s", (user_id,))
            conn.commit()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None


def _session_from_row(row: dict) -> SessionRecord:
    return SessionRecord(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


store = MemoryStore()
