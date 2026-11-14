"""邻接与整段取用：按位置取片、按 seq 取片、前后邻域。（KbStore 的 mixin 之一，见 kb/store.py）。"""

from __future__ import annotations

import uuid

# 方法体只需 self._get_pool()（来自 Database），不依赖其他 mixin。


class KbNeighborStore:

    def fetch_kb_chunks_by_location(
        self, name: str, page: int | None = None, *, limit: int = 3
    ) -> list[dict]:
        """按「文件名片段 + 页码」取切片：``kb_read`` 用，位置就是回答里的出处。

        命中的位置是模型本来就要交代的东西（文件名 · 第N页），不需要另造编号。
        """
        like = f"%{name.strip()}%"
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT c.file_id::text AS file_id, c.seq, c.page, c.page_end,
                       c.kind, c.section, c.scope, c.period, c.text, f.filename
                FROM agent_kb_chunk c JOIN agent_kb_file f ON f.id = c.file_id
                WHERE f.filename LIKE %(like)s
                  AND (
                    %(page)s::int IS NULL
                    OR c.page = %(page)s::int
                    OR (c.page <= %(page)s::int
                        AND COALESCE(c.page_end, c.page) >= %(page)s::int)
                  )
                ORDER BY c.seq
                LIMIT %(limit)s
                """,
                {"like": like, "page": page, "limit": limit},
            ).fetchall()
        return [dict(row) for row in rows]


    def fetch_kb_chunks(
        self, pairs: list[tuple[uuid.UUID, int]]
    ) -> dict[tuple[str, int], dict]:
        """按 (file_id, seq) 取切片原文：{(file_id, seq): {filename, page, ...}}。

        给 ``kb_read`` 用：检索结果只带引用（file:seq），原文始终从库里现读，
        所以读回来的一定是**当前**内容——知识库重跑后不会读到旧文本。
        """
        if not pairs:
            return {}
        wanted = {(str(fid), int(seq)) for fid, seq in pairs}
        file_ids = sorted({fid for fid, _ in wanted})
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT c.file_id::text AS file_id, c.seq, c.page, c.page_end,
                       c.kind, c.section, c.scope, c.period, c.text, f.filename
                FROM agent_kb_chunk c JOIN agent_kb_file f ON f.id = c.file_id
                WHERE c.file_id = ANY(%(ids)s::uuid[])
                """,
                {"ids": file_ids},
            ).fetchall()
        out: dict[tuple[str, int], dict] = {}
        for row in rows:
            key = (str(row["file_id"]), int(row["seq"]))
            if key not in wanted:
                continue
            out[key] = dict(row)
        return out


    def kb_chunk_neighbors(
        self,
        pairs: list[tuple[uuid.UUID, int]],
        *,
        radius: int = 1,
    ) -> dict[tuple[str, int], list[dict]]:
        """命中切片的相邻片段：{(file_id, seq): [{seq, page, kind, text}...]}。

        切片按字符长度切分，跨段落的内容（"原因有三：…" 接下一页）会被
        切断，只喂命中片给模型就会答半截。这里按 seq 前后各取 radius 段，
        检索层拼成一份连续上下文。
        """
        if not pairs or radius <= 0:
            return {}
        # 键统一用 str：查询结果里的 file_id 是 UUID 对象，两侧类型不一致
        # 会导致一条都匹配不上（表现为邻接永远为空）
        wanted = {
            (str(file_id), seq + delta)
            for file_id, seq in pairs
            for delta in range(-radius, radius + 1)
            if seq + delta >= 0
        }
        file_id_list = sorted({str(file_id) for file_id, _ in wanted})
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT file_id, seq, page, kind, text
                FROM agent_kb_chunk
                WHERE file_id = ANY(%(ids)s::uuid[])
                """,
                {"ids": file_id_list},
            ).fetchall()
        grouped: dict[tuple[str, int], list[dict]] = {}
        for row in rows:
            key = (str(row["file_id"]), int(row["seq"]))
            if key not in wanted:
                continue
            grouped.setdefault(key, []).append(
                {
                    "seq": int(row["seq"]),
                    "page": row["page"],
                    "kind": row["kind"],
                    "text": row["text"],
                }
            )
        return grouped
