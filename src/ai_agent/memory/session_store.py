from __future__ import annotations

import uuid

from ai_agent.llm.types import ChatMessage
from ai_agent.memory.db import (
    DEFAULT_SESSION_TITLE,
    ChatHistoryPage,
    Database,
    SessionRecord,
)
from psycopg.types.json import Jsonb


def _chat_message_from_row(row: dict) -> ChatMessage:
    msg: ChatMessage = {"role": row["role"], "content": row["content"] or ""}
    if row["tool_calls"]:
        msg["tool_calls"] = row["tool_calls"]
    if row["tool_call_id"]:
        msg["tool_call_id"] = row["tool_call_id"]
    if row["name"]:
        msg["name"] = row["name"]
    # meta 只在有内容时回填：老数据这一列是 NULL，多写一个空对象没意义
    meta = row.get("meta")
    if meta:
        msg["meta"] = meta
    return msg


def _session_from_row(row: dict) -> SessionRecord:
    return SessionRecord(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class SessionStore(Database):
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
                Jsonb(msg["meta"]) if msg.get("meta") else None,
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
                         tool_calls, tool_call_id, name, meta)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                            meta,
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
                            meta,
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

    def replace_chat_after_last_user(
        self, user_id: str, chat_session_id: str, messages: list[ChatMessage]
    ) -> None:
        """重新生成成功后的原子落库：在同一事务内删除最后一条用户消息之后的
        旧回答，并写入新生成的内容。删除与插入同事务提交，任一步失败整体回滚，
        旧回答不会丢。messages 为空时不动历史（保留旧回答）。"""
        rows = [m for m in messages if m.get("role") != "system"]
        if not rows:
            return
        values = [
            (
                str(uuid.uuid4()),
                msg["role"],
                msg.get("content") or "",
                Jsonb(msg["tool_calls"]) if msg.get("tool_calls") else None,
                msg.get("tool_call_id"),
                msg.get("name"),
                Jsonb(msg["meta"]) if msg.get("meta") else None,
            )
            for msg in rows
        ]
        with self._get_pool().connection() as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
                (user_id, chat_session_id),
            )
            conn.execute(
                """
                DELETE FROM agent_chat_history
                WHERE user_id = %s AND chat_session_id = %s AND seq > (
                    SELECT COALESCE(MAX(seq), 0)
                    FROM agent_chat_history
                    WHERE user_id = %s AND chat_session_id = %s AND role = 'user'
                )
                """,
                (user_id, chat_session_id, user_id, chat_session_id),
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
                         tool_calls, tool_call_id, name, meta)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            mid,
                            user_id,
                            chat_session_id,
                            start + offset,
                            role,
                            content,
                            tool_calls,
                            tool_call_id,
                            name,
                            meta,
                        )
                        for offset, (
                            mid,
                            role,
                            content,
                            tool_calls,
                            tool_call_id,
                            name,
                            meta,
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
                SELECT role, content, tool_calls, tool_call_id, name, meta
                FROM agent_chat_history
                WHERE user_id = %s AND chat_session_id = %s
                ORDER BY seq
                """
            params: tuple = (user_id, chat_session_id)
        else:
            sql = """
                SELECT role, content, tool_calls, tool_call_id, name, meta
                FROM (
                    SELECT role, content, tool_calls, tool_call_id, name, meta, seq
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
                    SELECT seq, role, content, tool_calls, tool_call_id, name, meta
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
                    SELECT seq, role, content, tool_calls, tool_call_id, name, meta
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

            oldest = rows[-1]
            if oldest["role"] != "user":
                extra = conn.execute(
                    """
                    SELECT seq, role, content, tool_calls, tool_call_id, name, meta
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
                SELECT role, content, tool_calls, tool_call_id, name, meta
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
