from __future__ import annotations

from datetime import datetime, timezone

from ai_agent.core.logging import get_logger
from ai_agent.storage.db import Database, MemoryItem, current_user_id
from ai_agent.memory.facets import is_mutex_facet

logger = get_logger(__name__)


class ProfileStore(Database):
    def write_profile(
        self,
        text: str,
        user_id: str | None = None,
        *,
        facet: str = "other",
        expires_at: str | None = None,
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
                    INSERT INTO agent_user_profile
                        (id, user_id, text, facet, created_at, expires_at, superseded_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NULL)
                    ON CONFLICT (user_id, facet) WHERE facet IN
                        ('name', 'language', 'locale', 'occupation')
                    DO UPDATE SET
                        text = EXCLUDED.text,
                        created_at = EXCLUDED.created_at,
                        expires_at = EXCLUDED.expires_at,
                        superseded_at = NULL
                    RETURNING id::text AS id
                    """,
                    (item.id, item.user_id, item.text, item.facet, item.created_at, expires_at),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    INSERT INTO agent_user_profile
                        (id, user_id, text, facet, created_at, expires_at, superseded_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NULL)
                    ON CONFLICT (id) DO UPDATE SET
                        expires_at = EXCLUDED.expires_at,
                        superseded_at = NULL
                    RETURNING id::text AS id
                    """,
                    (item.id, item.user_id, item.text, item.facet, item.created_at, expires_at),
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

    def supersede_profile_items(
        self, item_ids: list[str], user_id: str | None = None
    ) -> int:
        """把若干画像条目标为「已被覆盖」。

        不删行：留档能回答「这个偏好以前是什么」，也方便排查是不是模型把
        用户的一句反问当成了纠正。
        """
        ids = [item for item in item_ids if item]
        if not ids:
            return 0
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            result = conn.execute(
                """
                UPDATE agent_user_profile
                SET superseded_at = now()
                WHERE user_id = %s AND id::text = ANY(%s) AND superseded_at IS NULL
                """,
                (uid, ids),
            )
            conn.commit()
            return result.rowcount or 0

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
        self,
        user_id: str | None = None,
        limit: int = 20,
        *,
        include_inactive: bool = False,
    ) -> list[MemoryItem]:
        """当前有效的画像条目。

        默认只返回「未被覆盖、未过期」的：被新事实覆盖的（``superseded_at``）
        和软过期且没再出现的（``expires_at``）都留在库里供审计，但不进 prompt。
        ``include_inactive=True`` 供排查/管理端查看全量。
        """
        uid = user_id or current_user_id()
        where = "user_id = %s"
        if not include_inactive:
            where += " AND superseded_at IS NULL"
            where += " AND (expires_at IS NULL OR expires_at > now())"
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    id::text AS id,
                    user_id,
                    text,
                    facet,
                    created_at::text AS created_at
                FROM agent_user_profile
                WHERE {where}
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

    # ---- L1 管理：分页查看 / 单条删除 / 清空 ----
    def list_profile_page(
        self,
        user_id: str | None = None,
        *,
        limit: int = 50,
        offset: int = 0,
        include_inactive: bool = False,
    ) -> tuple[list[dict], bool]:
        """分页列出画像。``include_inactive`` 打开时把被覆盖/过期的也带出。

        返回 (items, has_more)。items 是 dict，带 expires_at / superseded_at，
        管理页要能解释「为什么这条现在不生效」。
        """
        uid = user_id or current_user_id()
        page = max(1, min(int(limit), 200))
        start = max(0, int(offset))
        where = "user_id = %s"
        if not include_inactive:
            where += (
                " AND superseded_at IS NULL"
                " AND (expires_at IS NULL OR expires_at > now())"
            )
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id::text AS id, user_id, text, facet,
                       created_at::text AS created_at,
                       expires_at::text AS expires_at,
                       superseded_at::text AS superseded_at
                FROM agent_user_profile
                WHERE {where}
                ORDER BY created_at DESC, id
                LIMIT %s OFFSET %s
                """,
                (uid, page + 1, start),
            ).fetchall()
        items = [dict(row) for row in rows[:page]]
        return items, len(rows) > page

    def delete_profile_item(
        self, item_id: str, user_id: str | None = None
    ) -> bool:
        """删除一条画像，并清掉由它派生、可能把它重新带回来的记录。

        - pending/promoted 候选按原文精确匹配删除；
        - L4 里 kind='promoted' 的同文副本一起删（归档只是副本，不应在
          用户删除后还能通过导出/展开看到）。
        """
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT text
                FROM agent_user_profile
                WHERE user_id = %s AND id::text = %s
                """,
                (uid, item_id),
            ).fetchone()
            if row is None:
                return False
            text = str(row["text"])
            conn.execute(
                "DELETE FROM agent_user_profile WHERE user_id = %s AND id::text = %s",
                (uid, item_id),
            )
            conn.execute(
                "DELETE FROM agent_memory_candidate WHERE user_id = %s AND text = %s",
                (uid, text),
            )
            conn.execute(
                """
                DELETE FROM agent_memory_archive
                WHERE user_id = %s AND kind = 'promoted' AND text = %s
                """,
                (uid, text),
            )
            conn.commit()
        return True

    def clear_profile(self, user_id: str | None = None) -> int:
        """清空 L1 画像。返回删除行数。

        只删 profile 行；候选/归档是独立层，需要全量删除走
        ``/v1/memory``（clear_long_term）。单条删除已经做了派生记录清理，
        这里保持「清空 L1」语义简单可预期。
        """
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                "DELETE FROM agent_user_profile WHERE user_id = %s RETURNING id",
                (uid,),
            ).fetchall()
            conn.commit()
        return len(rows)
