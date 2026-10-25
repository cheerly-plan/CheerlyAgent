from __future__ import annotations

from datetime import datetime, timezone

from ai_agent.core.logging import get_logger
from ai_agent.memory.db import Database, MemoryItem, current_user_id
from ai_agent.memory.facets import is_mutex_facet

logger = get_logger(__name__)


class ProfileStore(Database):
    def write_profile(
        self,
        text: str,
        user_id: str | None = None,
        *,
        facet: str = "other",
    ) -> MemoryItem:
        uid = user_id or current_user_id()
        text = text.strip()
        facet = facet.strip() or "other"
        mutex = is_mutex_facet(facet)
        # 互斥 facet 一个值一行（id 按 facet 确定性生成，天然 upsert）；
        # 累积 facet 每条独立事实一行（id 按文本确定性生成，同文本幂等去重）
        if mutex:
            item_id = str(self._point_id("agent_user_profile", uid, facet))
        else:
            item_id = str(self._point_id("agent_user_profile", uid, f"{facet}:{text}"))
        item = MemoryItem(
            id=item_id,
            user_id=uid,
            text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
            facet=facet,
        )
        with self._get_pool().connection() as conn:
            if mutex:
                row = conn.execute(
                    """
                    INSERT INTO agent_user_profile (id, user_id, text, facet, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, facet) WHERE facet IN
                        ('name', 'language', 'locale', 'occupation')
                    DO UPDATE SET
                        text = EXCLUDED.text,
                        created_at = EXCLUDED.created_at
                    RETURNING id::text AS id
                    """,
                    (item.id, item.user_id, item.text, item.facet, item.created_at),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    INSERT INTO agent_user_profile (id, user_id, text, facet, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    RETURNING id::text AS id
                    """,
                    (item.id, item.user_id, item.text, item.facet, item.created_at),
                ).fetchone()
                if row is None:
                    # 同文本已存在：取回原行，保持写入幂等
                    row = conn.execute(
                        """
                        SELECT id::text AS id,
                               created_at::text AS created_at
                        FROM agent_user_profile
                        WHERE id = %s
                        """,
                        (item.id,),
                    ).fetchone()
                    if row and row.get("created_at"):
                        item.created_at = row["created_at"]
            conn.commit()
        if row:
            item.id = row["id"]
        logger.info(
            "write_profile user_id=%s facet=%s mode=%s text=%s",
            item.user_id,
            item.facet,
            "replace" if mutex else "append",
            item.text,
        )
        return item

    def delete_profile_facet(
        self, facet: str, user_id: str | None = None, *, text: str | None = None
    ) -> bool:
        uid = user_id or current_user_id()
        facet = facet.strip() or "other"
        with self._get_pool().connection() as conn:
            if text:
                # 累积 facet 的单条否定：只删匹配的那几行，不动同 facet 其他条目；
                # 模型给的是原文转述时可能匹配不到，宁可漏删也不误删其他偏好
                row = conn.execute(
                    """
                    DELETE FROM agent_user_profile
                    WHERE user_id = %s AND facet = %s AND text = %s
                    RETURNING id
                    """,
                    (uid, facet, text.strip()),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    DELETE FROM agent_user_profile
                    WHERE user_id = %s AND facet = %s
                    RETURNING id
                    """,
                    (uid, facet),
                ).fetchone()
            conn.commit()
        return row is not None

    def list_profile(
        self, user_id: str | None = None, limit: int = 20
    ) -> list[MemoryItem]:
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    id::text AS id,
                    user_id,
                    text,
                    facet,
                    created_at::text AS created_at
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
                facet=row["facet"] or "other",
            )
            for row in rows
        ]
