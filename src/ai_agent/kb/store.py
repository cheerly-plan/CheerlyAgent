"""知识库文件与切片的 DB 存取。

与 CheckpointStore/EpisodeStore 同为 mixin，挂到 MemoryStore 上
共享连接池。

权限模型（P1）：知识库是**全站共享**的文档库——管理（列表/上传/删除）
只对管理员开放（路由层 require_admin 把关），普通用户只能在对话里通过
kb_search 工具检索内容。因此这里的读写方法都不按 user_id 过滤；
chunk.user_id 只作为「谁上传的」溯源信息保留。update_kb_status 供后台
管道用（不在任何 user_scope 里，按主键更新）。
"""

import uuid

from ai_agent.kb.chunking import Chunk
from pgvector import Vector


class KbStore:
    def create_kb_file(
        self,
        user_id: str,
        filename: str,
        size_bytes: int,
        mime: str,
        *,
        tags: list[str] | None = None,
    ) -> uuid.UUID:
        file_id = uuid.uuid4()
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_kb_file (id, user_id, filename, size_bytes, mime, tags)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (file_id, user_id, filename, size_bytes, mime, tags or []),
            )
            conn.commit()
        return file_id

    def list_kb_files(
        self, *, q: str = "", tag: str = ""
    ) -> list[dict]:
        """共享库：返回全部文件（管理员上传、全站可见）。

        q 按文件名模糊过滤，tag 精确匹配标签成员——都是管理页的
        轻量筛选，文件量级小，一个 WHERE 搞定不引入全文索引。
        """
        sql = """
                -- 对外字段名统一叫 file_id（与 POST /v1/kb/files 的返回值一致），
                -- 直接 SELECT id 会让前端拿到 undefined 拼出 /files/undefined
                SELECT id AS file_id, filename, size_bytes, mime, status, error,
                       chunk_count, text_count, table_count,
                       graph_status, graph_entity_count,
                       graph_batch_done, graph_batch_total, align_status, tags,
                       doc_type, extract_level, description, created_at
                FROM agent_kb_file
                WHERE (%(q)s::text = '' OR filename ILIKE %(q_like)s)
                  AND (%(tag)s::text = '' OR %(tag)s = ANY(tags))
                ORDER BY created_at DESC
                """
        params = {"q": q or "", "q_like": f"%{q}%", "tag": tag or ""}
        with self._get_pool().connection() as conn:
            return conn.execute(sql, params).fetchall()

    def set_kb_file_tags(self, file_id: uuid.UUID, tags: list[str]) -> bool:
        """全量覆盖标签；文件不存在返回 False（调用方转 404）。"""
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                UPDATE agent_kb_file SET tags = %s, updated_at = now()
                WHERE id = %s
                RETURNING id
                """,
                (tags, file_id),
            ).fetchone()
            conn.commit()
        return row is not None

    def rename_kb_file(self, file_id: uuid.UUID, filename: str) -> bool:
        """改文件展示名；文件不存在返回 False（调用方转 404）。

        磁盘文件与展示名同名（complete 时按 filename 落盘），改名必须
        同步重命名磁盘文件，否则下载 404——这在路由层做，这里只管 DB。
        """
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                UPDATE agent_kb_file SET filename = %s, updated_at = now()
                WHERE id = %s
                RETURNING id
                """,
                (filename, file_id),
            ).fetchone()
            conn.commit()
        return row is not None

    def find_kb_file_by_hash(
        self, content_hash: str, *, exclude_file_id: uuid.UUID | None = None
    ) -> dict | None:
        """内容指纹查重：返回任意一个同内容文件的 {file_id, filename}。"""
        if not content_hash:
            return None
        with self._get_pool().connection() as conn:
            return conn.execute(
                """
                SELECT id AS file_id, filename
                FROM agent_kb_file
                WHERE content_hash = %s AND id <> COALESCE(%s::uuid, '00000000-0000-0000-0000-000000000000'::uuid)
                LIMIT 1
                """,
                (content_hash, exclude_file_id),
            ).fetchone()

    def update_kb_file_hash(self, file_id: uuid.UUID, content_hash: str) -> None:
        with self._get_pool().connection() as conn:
            conn.execute(
                "UPDATE agent_kb_file SET content_hash = %s WHERE id = %s",
                (content_hash, file_id),
            )
            conn.commit()

    def has_ready_kb_file(self) -> bool:
        """是否有可检索的文件（检索前的一次轻量探针，避免拉全表）。"""
        with self._get_pool().connection() as conn:
            row = conn.execute(
                "SELECT EXISTS (SELECT 1 FROM agent_kb_file WHERE status = 'ready')"
            ).fetchone()
        return bool(row["exists"])

    def get_kb_file(self, file_id: uuid.UUID) -> dict | None:
        with self._get_pool().connection() as conn:
            return conn.execute(
                """
                SELECT id, filename, size_bytes, mime, status, doc_type, extract_level
                FROM agent_kb_file
                WHERE id = %s
                """,
                (file_id,),
            ).fetchone()

    def set_kb_extract_config(
        self, file_id: uuid.UUID, *, doc_type: str, level: str
    ) -> bool:
        """保存该文件的抽取配置（文档类型 + 粒度）。

        只改配置不动图谱：配置变了旧图就过期了，但要不要重抽是用户的决定
        （一次重抽是几十分钟的模型算力），前端提示「需重建生效」即可。
        返回 False 表示文件不存在。
        """
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                UPDATE agent_kb_file
                SET doc_type = %s, extract_level = %s, updated_at = now()
                WHERE id = %s
                RETURNING id
                """,
                (doc_type, level, file_id),
            ).fetchone()
            conn.commit()
        return row is not None

    def update_kb_status(
        self,
        file_id: uuid.UUID,
        status: str,
        *,
        error: str = "",
        chunk_count: int | None = None,
        text_count: int | None = None,
        table_count: int | None = None,
    ) -> None:
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                UPDATE agent_kb_file
                SET status = %s,
                    error = %s,
                    chunk_count = COALESCE(%s, chunk_count),
                    text_count = COALESCE(%s, text_count),
                    table_count = COALESCE(%s, table_count),
                    updated_at = now()
                WHERE id = %s
                """,
                (status, error, chunk_count, text_count, table_count, file_id),
            )
            conn.commit()

    def delete_kb_file(self, file_id: uuid.UUID) -> bool:
        """chunks 由 agent_kb_chunk 的 ON DELETE CASCADE 级联删除。"""
        with self._get_pool().connection() as conn:
            row = conn.execute(
                "DELETE FROM agent_kb_file WHERE id = %s RETURNING id",
                (file_id,),
            ).fetchone()
            conn.commit()
        return row is not None

    def recover_stuck_kb_files(self) -> int:
        """启动时不存在本进程的在途任务：卡在中间态的都是上次异常
        退出留下的，直接标 failed（与 checkpoint 僵尸 run 同套路）。"""
        with self._get_pool().connection() as conn:
            cur = conn.execute(
                """
                UPDATE agent_kb_file
                SET status = 'failed',
                    error = '服务重启导致处理中断，请删除后重新上传',
                    updated_at = now()
                WHERE status IN ('parsing', 'embedding')
                """
            )
            conn.commit()
        return cur.rowcount

    def insert_kb_chunks(
        self,
        file_id: uuid.UUID,
        user_id: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        seq_start: int,
    ) -> int:
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
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO agent_kb_chunk
                        (id, file_id, user_id, seq, page, page_end, section,
                         kind, scope, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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

    def search_kb_chunks(
        self,
        query_embedding: list[float],
        *,
        file_ids: list[uuid.UUID] | None = None,
        top_k: int = 3,
        max_distance: float | None = None,
    ) -> list[dict]:
        """全库检索（共享语义，不按 user_id 过滤）。

        `max_distance` 是余弦距离上限：超出的直接不返回。向量模型对无关
        内容也会给出 0.55~0.65 的基线距离，不设门槛时 top_k 会硬凑满，
        把"话题沾边但没用"的片段喂给模型，反而稀释了真正相关的内容。
        """
        # 查询向量必须包成 Vector：裸 list 会被 psycopg 适配成
        # double precision[]，与 vector 的 <=> 运算符类型不匹配直接报错
        emb = Vector(query_embedding)
        with self._get_pool().connection() as conn:
            return conn.execute(
                """
                SELECT c.id, c.file_id, c.seq, c.page, c.page_end, c.section,
                       c.kind, c.scope, c.text,
                       f.filename, c.embedding <=> %(emb)s AS distance
                FROM agent_kb_chunk c
                JOIN agent_kb_file f ON f.id = c.file_id
                WHERE (%(file_ids)s::uuid[] IS NULL
                       OR c.file_id = ANY(%(file_ids)s::uuid[]))
                  AND (%(max_distance)s::float8 IS NULL
                       OR c.embedding <=> %(emb)s <= %(max_distance)s)
                ORDER BY c.embedding <=> %(emb)s
                LIMIT %(top_k)s
                """,
                {
                    "emb": emb,
                    # 空列表退化为 NULL（= 不过滤）：否则 ANY('{}') 恒假，
                    # 调用方误传空列表会静默返回 0 条
                    "file_ids": file_ids or None,
                    "max_distance": max_distance,
                    "top_k": top_k,
                },
            ).fetchall()

    def search_kb_chunks_keyword(
        self,
        keywords: list[str],
        *,
        file_ids: list[uuid.UUID] | None = None,
        top_k: int = 12,
    ) -> list[dict]:
        """关键词直查（混合检索的第二路召回）。

        向量对"制度编号、专有名词、精确表述"不敏感（同义改写会把编号
        拉远），这里按关键词子串匹配召回：命中的关键词个数越多排越前，
        同分按出现位置靠前的先出。返回字段与 search_kb_chunks 一致
        （无 distance，用 keyword_score 代替），便于上层统一融合。
        """
        if not keywords:
            return []
        # 每个关键词一列 CASE：命中 +1。列数受限于调用方的关键词上限
        # （默认 6 个），不会拼出超长 SQL；关键词只作参数，无注入面
        score_sql = " + ".join(
            f"(CASE WHEN c.text ILIKE %(kw{i})s THEN 1 ELSE 0 END)"
            for i in range(len(keywords))
        )
        params: dict = {
            "file_ids": file_ids or None,
            "top_k": top_k,
        }
        for i, kw in enumerate(keywords):
            params[f"kw{i}"] = f"%{kw}%"
        with self._get_pool().connection() as conn:
            return conn.execute(
                f"""
                SELECT c.id, c.file_id, c.seq, c.page, c.page_end, c.section,
                       c.kind, c.scope, c.text,
                       f.filename,
                       ({score_sql})::float8 AS keyword_score
                FROM agent_kb_chunk c
                JOIN agent_kb_file f ON f.id = c.file_id
                WHERE (%(file_ids)s::uuid[] IS NULL
                       OR c.file_id = ANY(%(file_ids)s::uuid[]))
                  AND ({score_sql}) > 0
                ORDER BY keyword_score DESC, c.file_id, c.seq
                LIMIT %(top_k)s
                """,
                params,
            ).fetchall()

    def keyword_doc_freq(
        self,
        keywords: list[str],
        *,
        file_ids: list[uuid.UUID] | None = None,
    ) -> tuple[int, dict[str, int]]:
        """返回（切片总数, {关键词: 命中切片数}）。

        ILIKE 子串计数没有 IDF：「营业收入」这种词在财务表里遍地都是，
        命中不代表相关。一次聚合把 df 取出来，调用方据此只保留稀有词
        （编号、专名），关键词路才不会被通用词灌满。
        """
        if not keywords:
            return 0, {}
        cols = ", ".join(
            f"count(*) FILTER (WHERE c.text ILIKE %(kw{i})s) AS df{i}"
            for i in range(len(keywords))
        )
        params: dict = {"file_ids": file_ids or None}
        for i, kw in enumerate(keywords):
            params[f"kw{i}"] = f"%{kw}%"
        with self._get_pool().connection() as conn:
            row = conn.execute(
                f"""
                SELECT count(*) AS total, {cols}
                FROM agent_kb_chunk c
                WHERE (%(file_ids)s::uuid[] IS NULL
                       OR c.file_id = ANY(%(file_ids)s::uuid[]))
                """,
                params,
            ).fetchone()
        total = int(row["total"] or 0)
        return total, {kw: int(row[f"df{i}"] or 0) for i, kw in enumerate(keywords)}

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

