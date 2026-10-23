from __future__ import annotations

from datetime import datetime, timezone

from ai_agent.core.logging import get_logger
from ai_agent.memory.db import Database, MemoryItem, current_user_id

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
        item = MemoryItem(
            id=str(self._point_id("agent_user_profile", uid, facet)),
            user_id=uid,
            text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
            facet=facet,
        )
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                INSERT INTO agent_user_profile (id, user_id, text, facet, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id, facet) DO UPDATE SET
                    text = EXCLUDED.text,
                    created_at = EXCLUDED.created_at
                RETURNING id::text AS id
                """,
                (item.id, item.user_id, item.text, item.facet, item.created_at),
            ).fetchone()
            conn.commit()
        if row:
            item.id = row["id"]
        logger.info(
            "write_profile user_id=%s facet=%s text=%s",
            item.user_id,
            item.facet,
            item.text,
        )
        return item

    def delete_profile_facet(
        self, facet: str, user_id: str | None = None
    ) -> bool:
        uid = user_id or current_user_id()
        facet = facet.strip() or "other"
        with self._get_pool().connection() as conn:
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
