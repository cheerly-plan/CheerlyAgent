from __future__ import annotations

import uuid

from ai_agent.core.config import settings
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.short_term import short_term_memory
from ai_agent.storage.db import (
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
    # seq 只在 load_all_turns 里取；给摘要游标用，_to_openai_messages 会忽略
    if row.get("seq") is not None:
        msg["seq"] = int(row["seq"])
    # 老行这两列是 NULL（加列前没采集），保持缺省，不要造出 0
    if row.get("prompt_tokens") is not None:
        msg["prompt_tokens"] = int(row["prompt_tokens"])
    if row.get("completion_tokens") is not None:
        msg["completion_tokens"] = int(row["completion_tokens"])
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
                msg.get("prompt_tokens"),
                msg.get("completion_tokens"),
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
                         tool_calls, tool_call_id, name, meta,
                         prompt_tokens, completion_tokens)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                            prompt_tokens,
                            completion_tokens,
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
                            prompt_tokens,
                            completion_tokens,
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

    def attach_report_delivery(self, job_id: str, text: str) -> int:
        """把已完成研报任务的交付文本写回发起它的那条回答。

        任务表是内存的，进程重启就没了；交付文本落到 ``meta.report_delivery``
        之后，前端刷新或服务重启都能直接从历史消息里把下载链接渲染出来，
        不必再依赖 ``GET /v1/report/jobs/{id}``。返回命中的消息条数。
        """
        if not job_id or not text:
            return 0
        with self._get_pool().connection() as conn:
            cur = conn.execute(
                """
                UPDATE agent_chat_history
                SET meta = jsonb_set(
                        coalesce(meta, '{}'::jsonb),
                        '{report_delivery}',
                        to_jsonb(%s::text),
                        true
                    )
                WHERE meta->>'report_job_id' = %s
                """,
                (text, job_id),
            )
            conn.commit()
            return int(cur.rowcount or 0)

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
                SELECT role, content, tool_calls, tool_call_id, name, meta,
                       prompt_tokens, completion_tokens
                FROM agent_chat_history
                WHERE user_id = %s AND chat_session_id = %s
                ORDER BY seq
                """
            params: tuple = (user_id, chat_session_id)
        else:
            sql = """
                SELECT role, content, tool_calls, tool_call_id, name, meta,
                       prompt_tokens, completion_tokens
                FROM (
                    SELECT role, content, tool_calls, tool_call_id, name, meta,
                           prompt_tokens, completion_tokens, seq
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
        # turns < 0 表示不按轮数截断：预算模式要看到全部历史，再由字符预算
        # 决定哪些轮次进摘要；0 仍表示不加载。
        if turns == 0:
            return []
        with self._get_pool().connection() as conn:
            if turns < 0:
                start = conn.execute(
                    """
                    SELECT MIN(seq) AS start_seq
                    FROM agent_chat_history
                    WHERE user_id = %s AND chat_session_id = %s
                      AND role = 'user'
                    """,
                    (user_id, chat_session_id),
                ).fetchone()
            else:
                start = conn.execute(
                    """
                    SELECT MIN(seq) AS start_seq
                    FROM (
                        SELECT seq
                        FROM agent_chat_history
                        WHERE user_id = %s AND chat_session_id = %s
                          AND role = 'user'
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

    def load_all_turns(
        self, user_id: str, chat_session_id: str, after_seq: int = 0
    ) -> list[ChatMessage]:
        """加载 ``after_seq`` 之后的消息；默认(0)加载全部历史。

        摘要已经覆盖的 seq 由 ``agent_session.summary_until_seq`` 提供，
        冷启动只取游标之后的部分，避免 summary 与原文重复进入上下文。
        """
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT seq, role, content, tool_calls, tool_call_id, name, meta
                FROM agent_chat_history
                WHERE user_id = %s AND chat_session_id = %s AND seq > %s
                ORDER BY seq
                """,
                (user_id, chat_session_id, max(0, int(after_seq))),
            ).fetchall()
        return [_chat_message_from_row(row) for row in rows]

    def get_session_state(self, user_id: str, session_id: str) -> tuple[str, int]:
        """返回 (summary, summary_until_seq)。"""
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT summary, summary_until_seq
                FROM agent_session
                WHERE user_id = %s AND id::text = %s AND deleted_at IS NULL
                """,
                (user_id, session_id),
            ).fetchone()
        if not row:
            return "", 0
        return (row["summary"] or "").strip(), int(row.get("summary_until_seq") or 0)

    def get_session_summary(self, user_id: str, session_id: str) -> str:
        return self.get_session_state(user_id, session_id)[0]

    def update_session_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        summary_until_seq: int | None = None,
    ) -> None:
        with self._get_pool().connection() as conn:
            if summary_until_seq is None:
                conn.execute(
                    """
                    UPDATE agent_session
                    SET summary = %s
                    WHERE user_id = %s AND id::text = %s AND deleted_at IS NULL
                    """,
                    (summary, user_id, session_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE agent_session
                    SET summary = %s,
                        summary_until_seq = GREATEST(summary_until_seq, %s)
                    WHERE user_id = %s AND id::text = %s AND deleted_at IS NULL
                    """,
                    (summary, int(summary_until_seq), user_id, session_id),
                )
            conn.commit()

    def load_tool_result(
        self, user_id: str, chat_session_id: str, tool_call_id: str
    ) -> dict | None:
        """按 call_id 从持久化历史取回工具结果（池子淘汰/重启后的兜底）。"""
        call_id = (tool_call_id or "").strip()
        if not call_id:
            return None
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT content, name, meta
                FROM agent_chat_history
                WHERE user_id = %s AND chat_session_id = %s
                  AND tool_call_id = %s
                ORDER BY seq DESC
                LIMIT 1
                """,
                (user_id, chat_session_id, call_id),
            ).fetchone()
        if not row:
            return None
        return {
            "content": row["content"] or "",
            "name": row["name"] or "tool",
            "meta": row["meta"] or {},
        }

    def save_raw_tool_result(
        self,
        user_id: str,
        chat_session_id: str,
        tool_call_id: str,
        name: str,
        content: str,
        meta: dict | None = None,
    ) -> None:
        """保存裁剪前工具全文；进程池淘汰/重启后 read_tool_result 用。"""
        call_id = (tool_call_id or "").strip()
        if not call_id:
            return
        body = str(content or "")
        cap = max(1, int(settings.tool_result_pool_max_content_chars))
        truncated = len(body) > cap
        if truncated:
            body = body[:cap]
        payload = dict(meta or {})
        payload["raw_truncated"] = truncated
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_tool_result
                    (id, user_id, chat_session_id, tool_call_id, name,
                     content, meta, created_at, expires_at)
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, now(),
                    CASE WHEN %s > 0
                         THEN now() + (%s * interval '1 second')
                         ELSE NULL END
                )
                ON CONFLICT (user_id, chat_session_id, tool_call_id)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    content = EXCLUDED.content,
                    meta = EXCLUDED.meta,
                    created_at = now(),
                    expires_at = EXCLUDED.expires_at
                """,
                (
                    str(uuid.uuid4()),
                    user_id,
                    chat_session_id,
                    call_id,
                    name or "tool",
                    body,
                    Jsonb(payload),
                    int(settings.tool_result_db_ttl_seconds),
                    int(settings.tool_result_db_ttl_seconds),
                ),
            )
            conn.commit()

    def load_raw_tool_result(
        self, user_id: str, chat_session_id: str, tool_call_id: str
    ) -> dict | None:
        call_id = (tool_call_id or "").strip()
        if not call_id:
            return None
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT content, name, meta
                FROM agent_tool_result
                WHERE user_id = %s AND chat_session_id = %s
                  AND tool_call_id = %s
                  AND (expires_at IS NULL OR expires_at >= now())
                LIMIT 1
                """,
                (user_id, chat_session_id, call_id),
            ).fetchone()
        if not row:
            return None
        meta = row["meta"] if isinstance(row["meta"], dict) else {}
        return {
            "content": row["content"] or "",
            "name": row["name"] or "tool",
            "meta": meta,
            "truncated": bool(meta.get("raw_truncated")),
        }

    def delete_raw_tool_results(
        self, user_id: str, chat_session_id: str | None = None
    ) -> int:
        with self._get_pool().connection() as conn:
            if chat_session_id is None:
                cur = conn.execute(
                    "DELETE FROM agent_tool_result WHERE user_id = %s",
                    (user_id,),
                )
            else:
                cur = conn.execute(
                    """
                    DELETE FROM agent_tool_result
                    WHERE user_id = %s AND chat_session_id = %s
                    """,
                    (user_id, chat_session_id),
                )
            conn.commit()
            return int(cur.rowcount or 0)

    def cleanup_expired_tool_results(self) -> int:
        with self._get_pool().connection() as conn:
            cur = conn.execute(
                """
                DELETE FROM agent_tool_result
                WHERE expires_at IS NOT NULL AND expires_at < now()
                """
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def search_chat_history(
        self,
        user_id: str,
        chat_session_id: str,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        """当前会话全文关键词检索；摘要之外的旧对话细节用它找回。"""
        text = (query or "").strip()
        if not text:
            return []
        escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT seq, role, content
                FROM agent_chat_history
                WHERE user_id = %s AND chat_session_id = %s
                  AND role IN ('user', 'assistant')
                  AND content ILIKE %s ESCAPE '\\'
                ORDER BY seq DESC
                LIMIT %s
                """,
                (user_id, chat_session_id, f"%{escaped}%", max(1, min(limit, 20))),
            ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "role": row["role"],
                "content": row["content"] or "",
            }
            for row in reversed(rows)
        ]

    def load_chat_around(
        self,
        user_id: str,
        chat_session_id: str,
        seq: int,
        before: int = 2,
        after: int = 2,
    ) -> list[dict]:
        """按 seq 读一段历史上下文，用于 search_history 命中后回看前后文。"""
        anchor = max(0, int(seq))
        back = max(0, min(int(before), 20))
        forward = max(0, min(int(after), 20))
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT seq, role, content
                FROM agent_chat_history
                WHERE user_id = %s AND chat_session_id = %s
                  AND seq BETWEEN %s AND %s
                  AND role IN ('user', 'assistant')
                ORDER BY seq
                """,
                (user_id, chat_session_id, anchor - back, anchor + forward),
            ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "role": row["role"],
                "content": row["content"] or "",
            }
            for row in rows
        ]

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
                    DELETE FROM agent_tool_result
                    WHERE user_id = %s AND chat_session_id = %s
                    """,
                    (user_id, chat_session_id),
                )
                conn.execute(
                    """
                    UPDATE agent_session
                    SET summary = '', summary_until_seq = 0
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
                    "DELETE FROM agent_tool_result WHERE user_id = %s",
                    (user_id,),
                )
                conn.execute(
                    "UPDATE agent_session "
                    "SET summary = '', summary_until_seq = 0 WHERE user_id = %s",
                    (user_id,),
                )
            conn.commit()
        # DB 已清空，进程内热缓存必须一起失效，否则 TTL 内 load 仍会命中旧窗口，
        # 用户删掉的会话会「自己回来」。事务提交成功后再失效，避免清了缓存但库没清。
        short_term_memory.invalidate(user_id, chat_session_id)
        try:
            from ai_agent.tools.result_pool import tool_result_pool

            tool_result_pool.clear(user_id, chat_session_id)
        except Exception:
            # 工具结果池是进程内缓存，清理失败不影响 DB 删除
            pass

    def create_session(self, user_id: str, idempotency_key: str) -> SessionRecord:
        session_id = str(uuid.uuid4())
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                INSERT INTO agent_session (id, user_id, title, idempotency_key)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, idempotency_key) DO UPDATE
                    SET updated_at = now(),
                        deleted_at = NULL
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

    def touch_session(self, user_id: str, session_id: str) -> bool:
        """把会话标成刚用过：空会话清理看 updated_at，聊天开始必须碰一下，
        否则「创建后立刻开聊、历史还没落库」会被当成空会话删掉。"""
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                UPDATE agent_session
                SET updated_at = now()
                WHERE user_id = %s AND id::text = %s AND deleted_at IS NULL
                RETURNING id
                """,
                (user_id, session_id),
            ).fetchone()
            conn.commit()
        return row is not None

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
            conn.execute(
                """
                DELETE FROM agent_checkpoint
                WHERE user_id = %s AND chat_session_id = %s
                """,
                (user_id, session_id),
            )
            conn.execute(
                """
                DELETE FROM agent_tool_result
                WHERE user_id = %s AND chat_session_id = %s
                """,
                (user_id, session_id),
            )
            conn.commit()
        short_term_memory.invalidate(user_id, session_id)
        try:
            from ai_agent.tools.result_pool import tool_result_pool

            tool_result_pool.clear(user_id, session_id)
        except Exception:
            pass
        return True

    def cleanup_empty_unnamed_sessions(self, max_age_seconds: int = 3600) -> int:
        age = max(int(max_age_seconds), 0)
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                DELETE FROM agent_session s
                WHERE s.title = %s
                  AND s.deleted_at IS NULL
                  AND s.updated_at <= now() - (%s * interval '1 second')
                  AND NOT EXISTS (
                    SELECT 1
                    FROM agent_chat_history h
                    WHERE h.user_id = s.user_id
                      AND h.chat_session_id = s.id::text
                  )
                  AND NOT EXISTS (
                    SELECT 1
                    FROM agent_checkpoint c
                    WHERE c.user_id = s.user_id
                      AND c.chat_session_id = s.id::text
                      AND c.status = 'running'
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
            conn.execute("DELETE FROM agent_checkpoint WHERE user_id = %s", (user_id,))
            conn.commit()
