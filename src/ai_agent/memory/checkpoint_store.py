from ai_agent.llm.types import ChatMessage
from ai_agent.memory.db import Database
from psycopg import errors as pg_errors
from psycopg.types.json import Jsonb


class CheckpointStore(Database):
    def create_checkpoint(
        self,
        run_id: str,
        user_id: str,
        chat_session_id: str,
        messages: list[ChatMessage],
        use_web_search: bool = True,
        use_kb: bool = True,
    ) -> bool:
        """返回 False 表示触发 DB 级互斥：同会话已有 running 的 checkpoint
        （部分唯一索引兜底，路由层预检后的并发竞态由这里裁决）。"""
        try:
            with self._get_pool().connection() as conn:
                # 新 run 开始即作废本会话旧的 failed/interrupted checkpoint：
                # "继续生成"只对紧接着最后一条消息的中断有意义，用户已改发
                # 新问题后若还允许恢复旧 run，替换落库会把中间的新问答删掉
                conn.execute(
                    """
                    DELETE FROM agent_checkpoint
                    WHERE user_id = %s AND chat_session_id = %s
                      AND status IN ('failed', 'interrupted')
                    """,
                    (user_id, chat_session_id),
                )
                conn.execute(
                    """
                    INSERT INTO agent_checkpoint
                        (run_id, user_id, chat_session_id, status, step, messages,
                         use_web_search, use_kb)
                    VALUES (%s, %s, %s, 'running', 0, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE
                        SET messages = EXCLUDED.messages,
                            step = EXCLUDED.step,
                            use_web_search = EXCLUDED.use_web_search,
                            use_kb = EXCLUDED.use_kb,
                            updated_at = now()
                    """,
                    (
                        run_id,
                        user_id,
                        chat_session_id,
                        Jsonb(messages),
                        use_web_search,
                        use_kb,
                    ),
                )
                conn.commit()
        except pg_errors.UniqueViolation:
            return False
        return True

    def update_checkpoint(
        self,
        run_id: str,
        messages: list[ChatMessage],
        step: int,
    ) -> None:
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                UPDATE agent_checkpoint
                SET messages = %s, step = %s, updated_at = now()
                WHERE run_id = %s AND status = 'running'
                """,
                (Jsonb(messages), step, run_id),
            )
            conn.commit()

    def mark_checkpoint(self, run_id: str, status: str) -> None:
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError(f"非法 checkpoint 状态 {status}")
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                    UPDATE agent_checkpoint
                    SET status = %s, updated_at = now()
                    WHERE run_id = %s
                    """,
                (status, run_id),
            )
            conn.commit()

    def delete_checkpoint(self, run_id: str) -> None:
        with self._get_pool().connection() as conn:
            conn.execute("DELETE FROM agent_checkpoint WHERE run_id = %s", (run_id,))
            conn.commit()

    def cleanup_stale_checkpoints(self, stale_seconds: int = 1800) -> int:
        """兜底：进程被 kill -9 时 finally 不会执行，留下僵尸 running 行。
        定期把超时未更新的 running 置为 failed。"""
        seconds = max(int(stale_seconds), 0)
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                UPDATE agent_checkpoint
                SET status = 'failed', updated_at = now()
                WHERE status = 'running'
                  AND updated_at <= now() - (%s * interval '1 second')
                RETURNING run_id
                """,
                (seconds,),
            ).fetchall()
            conn.commit()
        return len(rows)

    def cleanup_old_checkpoints(self, max_age_seconds: int = 7 * 24 * 3600) -> int:
        """failed/interrupted 行只为断点续跑服务，保留 7 天后删除，
        避免用户不点"继续生成"时无限累积。"""
        seconds = max(int(max_age_seconds), 0)
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                DELETE FROM agent_checkpoint
                WHERE status IN ('failed', 'interrupted')
                  AND updated_at <= now() - (%s * interval '1 second')
                RETURNING run_id
                """,
                (seconds,),
            ).fetchall()
            conn.commit()
        return len(rows)

    def load_resumable_checkpoint(self, run_id: str) -> dict | None:
        """路由层预检用：只读，不认领（认领由 run_stream 里 claim 完成）。"""
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT run_id, user_id, chat_session_id, messages,
                       use_web_search, use_kb
                FROM agent_checkpoint
                WHERE run_id = %s AND status IN ('failed', 'interrupted')
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "user_id": row["user_id"],
            "chat_session_id": row["chat_session_id"],
            "messages": row["messages"],
            "use_web_search": row["use_web_search"],
            "use_kb": row["use_kb"],
        }

    def claim_resumable_checkpoint(self, run_id: str) -> dict | None:
        """原子认领：DELETE ... RETURNING，只有一个并发请求能拿到。
        认领后 run_stream 用新 run_id 正常跑；再失败会留下新的
        failed 行，仍可再次恢复，不会死锁在旧行上。"""
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                DELETE FROM agent_checkpoint
                WHERE run_id = %s AND status IN ('failed', 'interrupted')
                RETURNING user_id, chat_session_id, messages, use_web_search, use_kb
                """,
                (run_id,),
            ).fetchone()
            conn.commit()
        if row is None:
            return None
        return {
            "user_id": row["user_id"],
            "chat_session_id": row["chat_session_id"],
            "messages": row["messages"],
            "use_web_search": row["use_web_search"],
            "use_kb": row["use_kb"],
        }

    def has_running_checkpoint(self, user_id: str, chat_session_id: str) -> bool:
        """session 互斥：同一会话同时只允许一个进行中的 run。"""
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM agent_checkpoint
                WHERE user_id = %s AND chat_session_id = %s AND status = 'running'
                LIMIT 1
                """,
                (user_id, chat_session_id),
            ).fetchone()
        return row is not None

    def latest_resumable_run(self, user_id: str, chat_session_id: str) -> str | None:
        """刷新页面后前端发现可恢复 run 用（可选）。"""
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT run_id FROM agent_checkpoint
                WHERE user_id = %s AND chat_session_id = %s
                  AND status IN ('failed', 'interrupted')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id, chat_session_id),
            ).fetchone()
        return row["run_id"] if row else None

    def discard_resumable_checkpoints(
        self, user_id: str, chat_session_id: str
    ) -> int:
        """用户发新消息或重新生成时丢掉该会话未完成的断点，
        避免旧的 interrupted 行在新回合结束后仍被 pending_run 捞出来。"""
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                DELETE FROM agent_checkpoint
                WHERE user_id = %s AND chat_session_id = %s
                  AND status IN ('failed', 'interrupted')
                RETURNING run_id
                """,
                (user_id, chat_session_id),
            ).fetchall()
            conn.commit()
        return len(rows)
