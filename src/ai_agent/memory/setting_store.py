"""用户级记忆开关。

一个布尔开关，按 user_id 存一行：关闭后**不写新记忆、不注入旧记忆、不挂
记忆工具**，但已有数据仍保留，用户可以通过 /v1/memory/* 查看/导出/删除。
这样「关闭记忆」不会隐式删除数据，隐私删除是另一个显式动作。

读路径 fail-open：表不存在 / DB 抖动时按「开启」处理，避免一次读失败让
所有用户突然失去记忆；写路径失败则把异常抛给路由（用户要看得到没保存成功）。
"""

from __future__ import annotations

from ai_agent.core.logging import get_logger
from ai_agent.storage.db import Database, current_user_id

logger = get_logger(__name__)


class MemorySettingStore(Database):
    def memory_enabled(self, user_id: str | None = None) -> bool:
        uid = user_id or current_user_id()
        try:
            with self._get_pool().connection() as conn:
                row = conn.execute(
                    """
                    SELECT enabled
                    FROM agent_memory_setting
                    WHERE user_id = %s
                    """,
                    (uid,),
                ).fetchone()
        except Exception:
            # 旧库还没跑迁移 / DB 抖动：宁可继续用记忆，也不要静默清空上下文
            logger.warning("记忆开关读取失败，按开启处理 user_id=%s", uid)
            return True
        return True if row is None else bool(row["enabled"])

    def set_memory_enabled(
        self, enabled: bool, user_id: str | None = None
    ) -> bool:
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                INSERT INTO agent_memory_setting (user_id, enabled, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (user_id) DO UPDATE
                    SET enabled = EXCLUDED.enabled,
                        updated_at = now()
                RETURNING enabled
                """,
                (uid, bool(enabled)),
            ).fetchone()
            conn.commit()
        return bool(row["enabled"]) if row else bool(enabled)