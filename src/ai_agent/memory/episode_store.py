from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.storage.db import Database, MemoryItem, current_user_id
from ai_agent.storage.embeddings import embed_query, embed_texts
from ai_agent.memory.episode_merge import compact_episode_texts
from pgvector import Vector
from psycopg import Connection

logger = get_logger(__name__)


class EpisodeStore(Database):
    def write_episode(self, text: str, user_id: str | None = None) -> MemoryItem:
        uid = user_id or current_user_id()
        text = text.strip()
        src_vector = Vector(embed_texts([text])[0])
        # 锁事务只做相似检索；LLM 合并与 embedding 是网络调用，
        # 放在锁外，避免堵住同一用户的其他记忆写入
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
            conn.commit()  # 事务结束即释放 advisory lock

        if not hits:
            item = MemoryItem(
                id=str(self._point_id("agent_episode", uid, text)),
                user_id=uid,
                text=text,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            with self._get_pool().connection() as conn:
                self._upsert_episode(conn, item, src_vector)
                conn.commit()
            return item

        keeper = hits[0]
        stale_ids = [uuid.UUID(hit.id) for hit in hits[1:]]
        # 网络/LLM 调用在锁外执行
        merged_text = compact_episode_texts([*(hit.text for hit in hits), text])
        merged_vector = (
            Vector(embed_texts([merged_text])[0])
            if merged_text != text
            else src_vector
        )
        now = datetime.now(timezone.utc).isoformat()
        with self._get_pool().connection() as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
                (uid, "agent_episode"),
            )
            # 拿到锁后复核 hits 未被并发写入改动；变了就走安全路径直接新增，
            # 避免用过期 hits 覆盖他人的合并结果
            current = self._episode_texts(conn, uid, [hit.id for hit in hits])
            if not all(current.get(hit.id) == hit.text for hit in hits):
                item = MemoryItem(
                    id=str(self._point_id("agent_episode", uid, text)),
                    user_id=uid,
                    text=text,
                    created_at=now,
                )
                self._upsert_episode(conn, item, src_vector)
                conn.commit()
                return item
            # 合并前先把所有被并掉的 episode 的证据聚到 keeper 上：
            # hit_count 求和、sessions_seen 取并集。此前只删 stale 行，
            # 被并掉那条的跨会话证据一起消失，合并后的记忆可能永远
            # 达不到晋升/归档门槛。
            evidence_rows = conn.execute(
                """
                SELECT COALESCE(hit_count, 0)::int AS hit_count,
                       COALESCE(sessions_seen, '{}'::text[]) AS sessions_seen
                FROM agent_episode
                WHERE user_id = %s AND id::text = ANY(%s::text[])
                """,
                (uid, [hit.id for hit in hits]),
            ).fetchall()
            merged_hits = sum(int(row["hit_count"]) for row in evidence_rows)
            merged_sessions = sorted(
                {
                    str(session)
                    for row in evidence_rows
                    for session in (row["sessions_seen"] or [])
                }
            )
            if stale_ids:
                conn.execute(
                    """
                    DELETE FROM agent_episode
                    WHERE user_id = %s AND id = ANY(%s)
                    """,
                    (uid, stale_ids),
                )
            changed = merged_text != keeper.text or bool(stale_ids)
            if changed:
                conn.execute(
                    """
                    UPDATE agent_episode
                    SET text = %s, embedding = %s, created_at = %s,
                        last_seen_at = %s, hit_count = %s, sessions_seen = %s
                    WHERE id = %s AND user_id = %s
                    """,
                    (
                        merged_text,
                        merged_vector,
                        now,
                        now,
                        merged_hits,
                        merged_sessions,
                        uuid.UUID(keeper.id),
                        uid,
                    ),
                )
                logger.info(
                    "merge_episode user_id=%s keeper=%s stale=%s hits=%s "
                    "sessions=%s text=%s",
                    uid,
                    keeper.id,
                    len(stale_ids),
                    merged_hits,
                    len(merged_sessions),
                    merged_text,
                )
            conn.commit()
            return MemoryItem(
                id=keeper.id,
                user_id=uid,
                text=merged_text,
                created_at=keeper.created_at if merged_text == keeper.text else now,
                hits=merged_hits,
                last_seen=now if changed else keeper.last_seen,
            )

    def _episode_texts(
        self, conn: Connection, user_id: str, ids: list[str]
    ) -> dict[str, str]:
        rows = conn.execute(
            """
            SELECT id::text AS id, text
            FROM agent_episode
            WHERE user_id = %s AND id = ANY(%s)
            """,
            (user_id, [uuid.UUID(i) for i in ids]),
        ).fetchall()
        return {row["id"]: row["text"] for row in rows}

    def search_episode(
        self,
        query: str,
        k: int | None = None,
        user_id: str | None = None,
        threshold: float | None = None,
        chat_session_id: str = "",
    ) -> list[MemoryItem]:
        uid = user_id or current_user_id()
        vector = Vector(embed_query(query))
        top_k = k or settings.memory_top_k
        cutoff = settings.memory_score_threshold if threshold is None else threshold
        with self._get_pool().connection() as conn:
            hits = self._similar_episodes(conn, vector, uid, top_k, cutoff)
        # 检索命中即登记观察证据、刷新 TTL 基准。只挂在 search_episode 上——
        # write_episode 内部为合并做的 _similar_episodes 不算「被记起」。
        # 失败只记日志，绝不影响召回结果。
        if hits:
            try:
                self.record_episode_hits([hit.id for hit in hits], user_id=uid)
            except Exception:
                logger.exception("情景记忆命中计数失败 user_id=%s", uid)
            # 自动召回发生在某个具体会话里：把这个会话登记进 episode 的去重
            # 会话集合，反复被用到的记忆才能积累「跨会话复现」证据。
            # touch_episode_session 按 session_id 去重，同一会话多轮命中只算
            # 一次；不传 chat_session_id 的调用方行为与改动前逐字一致。
            if chat_session_id:
                try:
                    for hit in hits:
                        self.touch_episode_session(
                            hit.id, chat_session_id, user_id=uid
                        )
                except Exception:
                    logger.exception("情景记忆会话证据登记失败 user_id=%s", uid)
        return hits

    def _similar_episodes(
        self,
        conn: Connection,
        vector: Vector,
        user_id: str,
        k: int,
        threshold: float,
    ) -> list[MemoryItem]:
        ttl = settings.memory_episode_ttl_seconds
        sql = """
            SELECT
                id::text AS id,
                user_id,
                text,
                created_at::text AS created_at,
                COALESCE(last_seen_at, created_at)::text AS last_seen,
                COALESCE(hit_count, 0) AS hit_count,
                1 - (embedding <=> %s) AS score
            FROM agent_episode
            WHERE user_id = %s
              AND (%s <= 0 OR created_at >= now() - (%s * interval '1 second'))
              AND 1 - (embedding <=> %s) >= %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """
        # 命中续期写的是 last_seen_at，检索若仍只看 created_at，续期过的老记忆
        # 会留在库里却永远召不回，TTL 续期就成了空操作。
        sql = sql.replace(
            "OR created_at >=", "OR COALESCE(last_seen_at, created_at) >="
        )
        rows = conn.execute(
            sql, (vector, user_id, ttl, ttl, vector, threshold, vector, k)
        ).fetchall()
        return [
            MemoryItem(
                id=row["id"],
                user_id=row["user_id"],
                text=row["text"],
                created_at=row["created_at"],
                score=float(row["score"]),
                last_seen=row.get("last_seen"),
                hits=int(row.get("hit_count") or 0),
            )
            for row in rows
        ]

    def cleanup_expired_episodes(
        self, ttl_seconds: int | None = None, user_id: str | None = None
    ) -> int:
        """删掉超过 TTL 的情景记忆。0 / 负数 = 关闭，不删。"""
        ttl = (
            settings.memory_episode_ttl_seconds if ttl_seconds is None else ttl_seconds
        )
        if ttl <= 0:
            return 0
        # 命中续期 + 高热度保留 + 长期未命中降权；失败返回 0（宁可漏删，
        # 不误删刚被反复用到的记忆）。
        try:
            return self.decay_and_cleanup_episodes(
                ttl_seconds=ttl,
                decay_seconds=settings.memory_promotion_decay_seconds,
                min_hits=settings.memory_promotion_min_hits,
                user_id=user_id,
            )
        except Exception:
            logger.exception("渐进式情景记忆清理失败，本轮不删除")
            return 0

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

    # ---- L2 管理：分页查看 / 单条删除 / 清空 ----
    def list_episodes_page(
        self,
        user_id: str | None = None,
        *,
        limit: int = 50,
        offset: int = 0,
        include_expired: bool = False,
    ) -> tuple[list[dict], bool]:
        """分页列出情景记忆。默认只列 TTL 内仍会被召回的条目。"""
        uid = user_id or current_user_id()
        page = max(1, min(int(limit), 200))
        start = max(0, int(offset))
        ttl = max(0, int(settings.memory_episode_ttl_seconds))
        where = "user_id = %s"
        params: list[object] = [uid]
        if not include_expired and ttl > 0:
            where += (
                " AND COALESCE(last_seen_at, created_at)"
                " >= now() - (%s * interval '1 second')"
            )
            params.append(ttl)
        params.extend([page + 1, start])
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id::text AS id, user_id, text,
                       created_at::text AS created_at,
                       COALESCE(last_seen_at, created_at)::text AS last_seen,
                       COALESCE(hit_count, 0)::int AS hit_count,
                       cardinality(COALESCE(sessions_seen, '{{}}'::text[]))
                           AS session_seen
                FROM agent_episode
                WHERE {where}
                ORDER BY COALESCE(last_seen_at, created_at) DESC, id
                LIMIT %s OFFSET %s
                """,
                tuple(params),
            ).fetchall()
        items = [dict(row) for row in rows[:page]]
        return items, len(rows) > page

    def delete_episode(
        self, episode_id: str, user_id: str | None = None
    ) -> bool:
        """删除一条情景记忆及其派生副本/待晋升候选。

        promoted 候选对应的是已经进 L1 的结论，不随 L2 单条删除一起抹掉；
        只清 pending 候选，避免它之后又把同一条内容晋升回画像。
        """
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT text
                FROM agent_episode
                WHERE user_id = %s AND id::text = %s
                """,
                (uid, episode_id),
            ).fetchone()
            if row is None:
                return False
            text = str(row["text"])
            conn.execute(
                "DELETE FROM agent_episode WHERE user_id = %s AND id::text = %s",
                (uid, episode_id),
            )
            conn.execute(
                """
                DELETE FROM agent_memory_archive
                WHERE user_id = %s AND kind = 'episode' AND origin_id = %s
                """,
                (uid, episode_id),
            )
            conn.execute(
                """
                DELETE FROM agent_memory_candidate
                WHERE user_id = %s AND status = 'pending' AND text = %s
                """,
                (uid, text),
            )
            conn.commit()
        return True

    def clear_episodes(self, user_id: str | None = None) -> int:
        """清空 L2 情景记忆（含归档 episode 副本与 pending 候选）。"""
        uid = user_id or current_user_id()
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                "DELETE FROM agent_episode WHERE user_id = %s RETURNING id",
                (uid,),
            ).fetchall()
            conn.execute(
                "DELETE FROM agent_memory_archive "
                "WHERE user_id = %s AND kind = 'episode'",
                (uid,),
            )
            conn.execute(
                "DELETE FROM agent_memory_candidate "
                "WHERE user_id = %s AND status = 'pending'",
                (uid,),
            )
            conn.commit()
        return len(rows)
