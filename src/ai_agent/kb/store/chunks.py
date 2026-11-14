"""切片存储：写入、向量回填、正文更新、按页定位。（KbStore 的 mixin 之一，见 kb/store.py）。"""

from __future__ import annotations

import uuid
from psycopg.types.json import Jsonb
from pgvector import Vector

from ai_agent.kb.ingest.chunking import Chunk
from ai_agent.kb.store._helpers import (
    _chunk_indicators,
)

# 方法体只需 self._get_pool()（来自 Database），不依赖其他 mixin。


class KbChunkStore:

    def clear_kb_chunks(self, file_id: uuid.UUID) -> int:
        """删掉某文件已有切片，供重新解析前腾出 (file_id, seq) 主键。"""
        with self._get_pool().connection() as conn:
            cur = conn.execute(
                "DELETE FROM agent_kb_chunk WHERE file_id = %s",
                (file_id,),
            )
            conn.commit()
        return int(cur.rowcount or 0)


    def insert_kb_chunks(
        self,
        file_id: uuid.UUID,
        user_id: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        seq_start: int,
        *,
        subject: str = "",
        period: str = "",
        subjects: list[str] | None = None,
    ) -> int:
        """写入切片，并把文件的主体/报告期冗余到每一行。

        冗余是检索分桶的前提：``ROW_NUMBER() OVER (PARTITION BY subject)``
        要在单表上做，JOIN 文件表既走不成索引、也让窗口函数变慢。空主体
        （文件名里抽不出公司名）留空串，检索侧对空主体退回不分桶的原行为。
        """
        rows = [
            (
                uuid.uuid4(),
                file_id,
                user_id,
                seq_start + i,
                chunk.page,
                chunk.page_end,
                chunk.section,
                chunk.kind,
                chunk.scope,
                chunk.text,
                Vector(vector),
                subjects[i] if subjects is not None else subject,
                period,
                Jsonb(_chunk_indicators(chunk)),
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO agent_kb_chunk
                        (id, file_id, user_id, seq, page, page_end, section,
                         kind, scope, text, embedding, subject, period, meta)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
            conn.commit()
        return len(rows)


    def list_kb_chunks(
        self, file_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        """文件的切片明细（分页，按 seq 排序）。不取 embedding（大而无用）。

        返回 (chunks, total)：total 是该文件切片总数，供前端判断是否还有下一页。
        """
        with self._get_pool().connection() as conn:
            chunks = conn.execute(
                """
                SELECT seq, page, page_end, section, kind, text
                FROM agent_kb_chunk
                WHERE file_id = %s
                ORDER BY seq
                LIMIT %s OFFSET %s
                """,
                (file_id, limit, offset),
            ).fetchall()
            row = conn.execute(
                "SELECT count(*) AS total FROM agent_kb_chunk WHERE file_id = %s",
                (file_id,),
            ).fetchone()
        return chunks, int(row["total"])


    def locate_kb_chunk_page(
        self, file_id: uuid.UUID, page: int
    ) -> dict | None:
        """按页码找切片位置：该页（或其后最近一页）第一条切片。

        前端切片列表是虚拟滚动、按 seq 顺序累积加载的，所以「跳到第 N 页」
        真正需要的是两个数：目标切片的 seq（用于高亮）和它在文件内的序号
        index（用于算滚动位置）。没有切片命中的页（图片页/空白页）顺延到
        后一页，page 字段回传实际命中的页，前端据此提示。

        page 为 NULL 的切片（md/txt 这类不分页的文件）不参与定位。
        """
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT seq, page
                FROM agent_kb_chunk
                WHERE file_id = %s AND page IS NOT NULL AND page >= %s
                ORDER BY seq
                LIMIT 1
                """,
                (file_id, page),
            ).fetchone()
            if row is None:
                return None
            counted = conn.execute(
                """
                SELECT count(*) AS n
                FROM agent_kb_chunk
                WHERE file_id = %s AND seq < %s
                """,
                (file_id, row["seq"]),
            ).fetchone()
        return {
            "seq": int(row["seq"]),
            "page": int(row["page"]),
            "index": int(counted["n"]),
        }


    def kb_chunk_page_count(self, file_id: uuid.UUID) -> int:
        """文件的最大页码（跨页切片取 page_end）。无页码的文件返回 0。"""
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT coalesce(max(coalesce(page_end, page)), 0) AS n
                FROM agent_kb_chunk
                WHERE file_id = %s
                """,
                (file_id,),
            ).fetchone()
        return int(row["n"]) if row else 0


    def list_kb_chunk_inputs(self, file_id: uuid.UUID) -> list[dict]:
        """改名重嵌用：全量切片的 (id, seq, text, page, kind, section)，按 seq 排序。

        与 list_kb_chunks 的区别：带 id（要回写向量）、不分页（重嵌必须全量）、
        不取 embedding（大而无用）。section 必须取：重嵌的语境前缀要跟入库
        时一致，否则同一份文件里新旧向量不同源，召回会偏。seq 也必须取：
        建图时要靠它写 mention，不能拿列表下标顶替（见 Chunk.seq）。
        """
        with self._get_pool().connection() as conn:
            return conn.execute(
                """
                SELECT id, seq, text, page, page_end, kind, section
                FROM agent_kb_chunk
                WHERE file_id = %s
                ORDER BY seq
                """,
                (file_id,),
            ).fetchall()


    def update_kb_chunk_vectors(
        self, items: list[tuple[uuid.UUID, list[float]]]
    ) -> int:
        """批量回写切片向量（一个事务），返回更新条数。文本不动。"""
        if not items:
            return 0
        with self._get_pool().connection() as conn:
            conn.executemany(
                "UPDATE agent_kb_chunk SET embedding = %s WHERE id = %s",
                [(Vector(vector), chunk_id) for chunk_id, vector in items],
            )
            conn.commit()
        return len(items)


    def get_kb_chunk(self, file_id: uuid.UUID, seq: int) -> dict | None:
        """取单个切片，带上文件状态：编辑前校验、改完标图谱 stale 都要用。"""
        with self._get_pool().connection() as conn:
            return conn.execute(
                """
                SELECT c.id, c.file_id, c.seq, c.page, c.page_end, c.section,
                       c.kind, c.text,
                       f.status AS file_status, f.graph_status, f.filename
                FROM agent_kb_chunk c
                JOIN agent_kb_file f ON f.id = c.file_id
                WHERE c.file_id = %s AND c.seq = %s
                """,
                (file_id, seq),
            ).fetchone()


    def update_kb_chunk_text(
        self, chunk_id: uuid.UUID, text: str, vector: list[float]
    ) -> None:
        """人工修订切片：文本与向量一起更新，下一次检索即用新值。

        只动这一行——切片向量彼此独立，没有跨片聚合，重嵌一整份文件纯属浪费。
        """
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                UPDATE agent_kb_chunk
                SET text = %s, embedding = %s
                WHERE id = %s
                """,
                (text, Vector(vector), chunk_id),
            )
            conn.commit()
