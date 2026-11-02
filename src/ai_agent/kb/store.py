"""知识库文件与切片的 DB 存取。

与 CheckpointStore/EpisodeStore 同为 mixin，挂到 MemoryStore 上
共享连接池。

权限模型（P1）：知识库是**全站共享**的文档库——管理（列表/上传/删除）
只对管理员开放（路由层 require_admin 把关），普通用户只能在对话里通过
kb_search 工具检索内容。因此这里的读写方法都不按 user_id 过滤；
chunk.user_id 只作为「谁上传的」溯源信息保留。update_kb_status 供后台
管道用（不在任何 user_scope 里，按主键更新）。
"""

import re
import uuid

from psycopg.types.json import Jsonb

from ai_agent.core.config import settings
from ai_agent.kb.chunking import Chunk, extract_table_indicators
from ai_agent.kb.meta import COMPANY_SUFFIX, extract_aliases, extract_file_meta
from pgvector import Vector

# 检索分桶键：按主体（公司）分桶，每个主体的头名在相关性带宽内保底一个
# 名额（见 search_kb_chunks）。「A 和 B 的营业收入」这种一次调用跨两个
# 主体时，两边的头名都进得来，不再是一家吃满、一家落空。
# 主体为空（老数据 / 文件名里抽不出公司名）的切片全部落在同一个桶里，
# 行为与分桶前逐字一致——回填 subject 之后才生效，不会突然改掉存量结果。
_SUBJECT_BUCKET = "COALESCE(NULLIF(subject, ''), '')"
# 行项目存在 meta.indicators。关键词命中正文或这一列都算，不拿它做 WHERE。
_INDICATOR_LIST = (
    "CASE WHEN jsonb_typeof(c.meta->'indicators') = 'array' "
    "THEN c.meta->'indicators' ELSE '[]'::jsonb END"
)


def _keyword_hit(i: int) -> str:
    return (
        f"(CASE WHEN c.text ILIKE %(kw{i})s OR EXISTS ("
        f"SELECT 1 FROM jsonb_array_elements_text({_INDICATOR_LIST}) AS item "
        f"WHERE item ILIKE %(kw{i})s) THEN 1 ELSE 0 END)"
    )


def _chunk_indicators(chunk: Chunk) -> dict:
    """只有表格切片记行项目。正文没有这一列，meta 留空对象。"""
    if chunk.kind != "table":
        return {}
    names = extract_table_indicators(chunk.text)
    return {"indicators": names} if names else {}


def _chunk_meta_where(
    *,
    kinds: list[str] | None,
    subjects: list[str] | None,
    periods: list[str] | None,
    scopes: list[str] | None,
    sections: list[str] | None,
    params: dict,
) -> str:
    """生成 chunk 元数据过滤的 SQL 片段（前导 AND 连接），并把参数写入 params。

    列名全部硬编码（kind/subject/period/scope/section），值走参数化，无注入面。
    空列表 / None 退化为不过滤（与 file_ids 现有「空列表退化为 NULL」一致）。语义：
    - kinds / periods / scopes：精确 = ANY（枚举 / 标准格式，安全）
    - subjects：ILIKE 模糊，兼容「中国银行」缩写匹配「中国银行股份有限公司」
    - sections：ILIKE 模糊，章节名写法多（「第十节 管理层讨论」vs「十、管理层讨论」）
    """

    clauses: list[str] = []
    if kinds:
        clauses.append("AND c.kind = ANY(%(kinds)s)")
        params["kinds"] = list(kinds)
    if subjects:
        sub = " OR ".join(
            f"c.subject ILIKE %(subject{i})s" for i in range(len(subjects))
        )
        clauses.append(f"AND ({sub})")
        for i, s in enumerate(subjects):
            params[f"subject{i}"] = f"%{s}%"
    if periods:
        clauses.append("AND c.period = ANY(%(periods)s)")
        params["periods"] = list(periods)
    if scopes:
        clauses.append("AND c.scope = ANY(%(scopes)s)")
        params["scopes"] = list(scopes)
    if sections:
        sub = " OR ".join(
            f"c.section ILIKE %(section{i})s" for i in range(len(sections))
        )
        clauses.append(f"AND ({sub})")
        for i, s in enumerate(sections):
            params[f"section{i}"] = f"%{s}%"
    return " ".join(clauses)


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
                       doc_type, extract_level, description, created_at,
                       subject, period, subject_aliases
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

        改名会连带重算主体：主体是从文件名规则抽出来的，名字改了主体就得
        跟着变，否则检索仍按旧主体分桶。切片上冗余的那份一并刷掉。
        """
        file_meta = extract_file_meta(filename)
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                UPDATE agent_kb_file
                SET filename = %s,
                    subject = %s,
                    subject_aliases = %s,
                    period = %s,
                    meta = %s,
                    updated_at = now()
                WHERE id = %s
                RETURNING id
                """,
                (
                    filename,
                    file_meta["subject"],
                    list(file_meta["subject_aliases"]),
                    file_meta["period"],
                    Jsonb(file_meta["meta"]),
                    file_id,
                ),
            ).fetchone()
            conn.execute(
                "UPDATE agent_kb_chunk SET subject = %s, period = %s WHERE file_id = %s",
                (file_meta["subject"], file_meta["period"], file_id),
            )
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
                SELECT id, filename, size_bytes, mime, status, doc_type, extract_level,
                       subject, subject_aliases, period, description
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

    def set_kb_file_meta(
        self,
        file_id: uuid.UUID,
        *,
        subject: str,
        subject_aliases: list[str],
        period: str,
        meta: dict | None = None,
    ) -> bool:
        """写入文件的检索元数据（主体 / 别名 / 报告期 / 长尾）。

        只动元数据不动切片：存量文件的切片由 backfill 脚本按 file_id 回填，
        新文件则由 insert_kb_chunks 直接继承。返回 False 表示文件不存在。
        """
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                UPDATE agent_kb_file
                SET subject = %s,
                    subject_aliases = %s,
                    period = %s,
                    meta = %s,
                    updated_at = now()
                WHERE id = %s
                RETURNING id
                """,
                (
                    subject,
                    list(subject_aliases),
                    period,
                    Jsonb(meta or {}),
                    file_id,
                ),
            ).fetchone()
            conn.commit()
        return row is not None

    def set_kb_file_summary(
        self,
        file_id: uuid.UUID,
        summary: str,
        embedding: list[float] | None = None,
        *,
        status: str = "",
    ) -> bool:
        """写入文档摘要与其向量；``embedding`` 为 None 表示只更新文本。

        status 跟着传进来的写：'' 未生成 / ready / failed / empty。摘要与
        向量放在同一条 UPDATE 里，不出现「文本更新了向量还是旧的」的中间态
        ——那样路由会按旧主题去召回新文档。
        """
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                UPDATE agent_kb_file
                SET summary = %s,
                    summary_embedding = COALESCE(%s, summary_embedding),
                    summary_status = %s,
                    updated_at = now()
                WHERE id = %s
                RETURNING id
                """,
                (
                    summary,
                    Vector(embedding) if embedding else None,
                    status,
                    file_id,
                ),
            ).fetchone()
            conn.commit()
        return row is not None

    def set_kb_subject_aliases(self, file_id: uuid.UUID, aliases: list[str]) -> None:
        """只更新主体别名，不动报告期和其它元数据。"""
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                UPDATE agent_kb_file
                SET subject_aliases = %s, updated_at = now()
                WHERE id = %s
                """,
                (list(aliases), file_id),
            )
            conn.commit()

    def backfill_file_meta(
        self,
        file_id: uuid.UUID,
        filename: str,
        *,
        subject: str | None = None,
        period: str | None = None,
    ) -> tuple[dict, int]:
        """按当前规则重算一份文件的主体/报告期，并把结果刷到它的切片上。

        存量文件在加元数据之前就入库了，文件行与切片行的 subject 都是空的，
        检索因此退化成「不分桶」。回填只改元数据不动向量，随时可重跑。
        ``subject`` / ``period`` 显式传入时用它覆盖规则抽取的结果——文件名
        不规范（「最终版.pdf」）时规则抽不出来，得让人填。

        返回 (实际写入的元数据, 被刷新的切片行数)。
        """
        meta = extract_file_meta(filename)
        if subject is not None:
            meta["subject"] = subject.strip()
            meta["subject_aliases"] = extract_aliases(meta["subject"])
        if period is not None:
            meta["period"] = period.strip()
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                UPDATE agent_kb_file
                SET subject = %s, subject_aliases = %s, period = %s,
                    meta = %s, updated_at = now()
                WHERE id = %s
                """,
                (
                    meta["subject"],
                    list(meta["subject_aliases"]),
                    meta["period"],
                    Jsonb(meta["meta"]),
                    file_id,
                ),
            )
            cur = conn.execute(
                """
                UPDATE agent_kb_chunk SET subject = %s, period = %s
                WHERE file_id = %s
                """,
                (meta["subject"], meta["period"], file_id),
            )
            chunks = cur.rowcount
            conn.commit()
        return meta, chunks

    def list_kb_subjects(self, user_id: str) -> list[dict]:
        """知识库里已出现的全部主体（含别名），供多主体文档切片归属用。

        只取有主体的文件：主体为空说明文件名里抽不出公司名，它既不能被
        别人引用为候选，也不该让别人的切片归到空主体上。
        """
        with self._get_pool().connection() as conn:
            return conn.execute(
                """
                SELECT DISTINCT subject, subject_aliases
                FROM agent_kb_file
                WHERE user_id = %s AND COALESCE(subject, '') <> ''
                """,
                (user_id,),
            ).fetchall()

    def list_kb_file_entity_names(self, file_id: uuid.UUID, limit: int = 40) -> list[str]:
        """本文件图谱里被提及最多的实体名，供文档摘要补充「讲了哪些主体」。

        不限于公司后缀（list_kb_entity_subjects 只取带公司后缀的）：摘要要
        表达的是「这份文件讲什么」，指标名、项目名称同样有区分度。
        """
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT name FROM agent_kg_entity
                WHERE file_id = %s
                ORDER BY mention_count DESC, name
                LIMIT %s
                """,
                (file_id, max(1, limit)),
            ).fetchall()
        return [r["name"] for r in rows if (r["name"] or "").strip()]

    def list_kb_entity_subjects(self, file_id: uuid.UUID) -> list[str]:
        """本文件图谱里像「公司主体」的实体名，补充文件级主体词表。

        行业研报这类文档自己的文件主体可能是「新能源行业」，但正文讲的是
        宁德时代、比亚迪——这些名字只有在图谱实体里才有。用后缀规则筛，
        不信任模型给的 type（实测 type 会写成 org/company/机构 各种）。
        """
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT name FROM agent_kg_entity WHERE file_id = %s
                """,
                (file_id,),
            ).fetchall()
        return [r["name"] for r in rows if COMPANY_SUFFIX.search(r["name"] or "")]

    def reassign_chunk_subjects(
        self, file_id: uuid.UUID, pairs: list[tuple[int, str]]
    ) -> int:
        """按 (seq, subject) 批量覆盖切片主体，返回受影响行数。

        单文件内 seq 唯一，用 executemany 一次刷完；不 commit 到一半——
        归属错了可以整体重跑，不该留下「一半新一半旧」的中间态。
        """
        if not pairs:
            return 0
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    UPDATE agent_kb_chunk SET subject = %s
                    WHERE file_id = %s AND seq = %s
                    """,
                    [(subject, file_id, seq) for seq, subject in pairs],
                )
                affected = cur.rowcount
            conn.commit()
        return affected

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

    def search_kb_files(
        self,
        query_embedding: list[float],
        *,
        file_ids: list[uuid.UUID] | None = None,
        top_k: int = 6,
        max_distance: float | None = None,
    ) -> list[dict]:
        """文档级检索：在摘要向量上找「哪些文件整体上与这个问题相关」。

        与 search_kb_chunks 的分工：它出片段，这里出文件池。file_ids 非空时
        在这个子集里挑（会话已限定范围时不去摸范围外的文件）；max_distance
        是路由的门槛，超线的文件连候选都不该进。
        """
        emb = Vector(query_embedding)
        with self._get_pool().connection() as conn:
            return conn.execute(
                """
                SELECT id AS file_id, filename, subject, period, summary,
                       summary_embedding <=> %(emb)s AS distance
                FROM agent_kb_file
                WHERE summary_embedding IS NOT NULL
                  AND status = 'ready'
                  AND (%(file_ids)s::uuid[] IS NULL
                       OR id = ANY(%(file_ids)s::uuid[]))
                  AND (%(max_distance)s::float8 IS NULL
                       OR summary_embedding <=> %(emb)s <= %(max_distance)s)
                ORDER BY distance
                LIMIT %(top_k)s
                """,
                {
                    "emb": emb,
                    "file_ids": file_ids or None,
                    "max_distance": max_distance,
                    "top_k": top_k,
                },
            ).fetchall()

    def search_kb_chunks(
        self,
        query_embedding: list[float],
        *,
        file_ids: list[uuid.UUID] | None = None,
        top_k: int = 3,
        max_distance: float | None = None,
        kinds: list[str] | None = None,
        subjects: list[str] | None = None,
        periods: list[str] | None = None,
        scopes: list[str] | None = None,
        sections: list[str] | None = None,
    ) -> list[dict]:
        """全库检索（共享语义，不按 user_id 过滤）。

        `max_distance` 是余弦距离上限：超出的直接不返回。向量模型对无关
        内容也会给出 0.55~0.65 的基线距离，不设门槛时 top_k 会硬凑满，
        把"话题沾边但没用"的片段喂给模型，反而稀释了真正相关的内容。

        名额按主体（``subject``）保底：每个主体的头名只要与全局最佳命中
        的距离差在 ``kb_subject_quota_band`` 内，就先各占一个名额，剩下的
        再按距离竞争。这样一次调用里圈了两家公司的文档时两家都出得来
        （否则它们共用一个 query 向量，离混合向量更近的那家会把名额吃满）；
        而只问一家时，别家头名差得远就不保底，不会白白稀释上下文。
        """
        # 查询向量必须包成 Vector：裸 list 会被 psycopg 适配成
        # double precision[]，与 vector 的 <=> 运算符类型不匹配直接报错
        emb = Vector(query_embedding)
        params = {
            "emb": emb,
            # 空列表退化为 NULL（= 不过滤）：否则 ANY('{}') 恒假，
            # 调用方误传空列表会静默返回 0 条
            "file_ids": file_ids or None,
            "max_distance": max_distance,
            "top_k": top_k,
            "band": settings.kb_subject_quota_band,
        }
        # chunk 元数据过滤：列名硬编码、值参数化，无注入面；空列表/NULL 退化为不过滤
        meta_where = _chunk_meta_where(
            kinds=kinds,
            subjects=subjects,
            periods=periods,
            scopes=scopes,
            sections=sections,
            params=params,
        )
        with self._get_pool().connection() as conn:
            return conn.execute(
                f"""
                WITH hit AS (
                    SELECT c.id, c.file_id, c.seq, c.page, c.page_end, c.section,
                           c.kind, c.scope, c.text, f.filename, c.subject,
                           {_INDICATOR_LIST} AS indicators,
                           c.embedding <=> %(emb)s AS distance
                    FROM agent_kb_chunk c
                    JOIN agent_kb_file f ON f.id = c.file_id
                    WHERE (%(file_ids)s::uuid[] IS NULL
                           OR c.file_id = ANY(%(file_ids)s::uuid[]))
                      AND (%(max_distance)s::float8 IS NULL
                           OR c.embedding <=> %(emb)s <= %(max_distance)s)
                      {meta_where}
                ),
                ranked AS (
                    SELECT hit.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY {_SUBJECT_BUCKET} ORDER BY distance
                           ) AS rn,
                           MIN(distance) OVER (PARTITION BY {_SUBJECT_BUCKET})
                               AS subject_best,
                           MIN(distance) OVER () AS global_best
                    FROM hit
                )
                SELECT id, file_id, seq, page, page_end, section,
                       kind, scope, text, filename, subject, indicators, distance
                FROM ranked
                ORDER BY (rn = 1 AND subject_best <= global_best + %(band)s) DESC,
                         distance
                LIMIT %(top_k)s
                """,
                params,
            ).fetchall()

    def search_kb_chunks_keyword(
        self,
        keywords: list[str],
        *,
        file_ids: list[uuid.UUID] | None = None,
        top_k: int = 12,
        kinds: list[str] | None = None,
        subjects: list[str] | None = None,
        periods: list[str] | None = None,
        scopes: list[str] | None = None,
        sections: list[str] | None = None,
    ) -> list[dict]:
        """关键词直查（混合检索的第二路召回）。

        向量对"制度编号、专有名词、精确表述"不敏感（同义改写会把编号
        拉远），这里按关键词子串匹配召回：命中的关键词个数越多排越前，
        同分按出现位置靠前的先出。返回字段与 search_kb_chunks 一致
        （无 distance，用 keyword_score 代替），便于上层统一融合。

        名额同样按主体保底（与向量路同一套规则，带宽为 1 个关键词）：
        关键词路也会被「营业收入」这种通用词灌满单一主体的切片。
        """
        if not keywords:
            return []
        # 每个关键词一列 CASE：命中 +1。列数受限于调用方的关键词上限
        # （默认 6 个），不会拼出超长 SQL；关键词只作参数，无注入面
        score_sql = " + ".join(_keyword_hit(i) for i in range(len(keywords)))
        params: dict = {
            "file_ids": file_ids or None,
            "top_k": top_k,
            # 关键词分是「命中了几个关键词」的整数，带宽 1 = 只给「少命中
            # 一个词以内」的主体保底；差两个词以上的主体硬塞只会灌水
            "band": 1,
        }
        for i, kw in enumerate(keywords):
            params[f"kw{i}"] = f"%{kw}%"
        # chunk 元数据过滤：列名硬编码、值参数化，无注入面；空列表/NULL 退化为不过滤
        meta_where = _chunk_meta_where(
            kinds=kinds,
            subjects=subjects,
            periods=periods,
            scopes=scopes,
            sections=sections,
            params=params,
        )
        with self._get_pool().connection() as conn:
            return conn.execute(
                f"""
                WITH hit AS (
                    SELECT c.id, c.file_id, c.seq, c.page, c.page_end, c.section,
                           c.kind, c.scope, c.text, f.filename, c.subject,
                           {_INDICATOR_LIST} AS indicators,
                           ({score_sql})::float8 AS keyword_score
                    FROM agent_kb_chunk c
                    JOIN agent_kb_file f ON f.id = c.file_id
                    WHERE (%(file_ids)s::uuid[] IS NULL
                           OR c.file_id = ANY(%(file_ids)s::uuid[]))
                      AND ({score_sql}) > 0
                      {meta_where}
                ),
                ranked AS (
                    SELECT hit.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY {_SUBJECT_BUCKET}
                               ORDER BY keyword_score DESC, file_id, seq
                           ) AS rn,
                           MAX(keyword_score) OVER (PARTITION BY {_SUBJECT_BUCKET})
                               AS subject_best,
                           MAX(keyword_score) OVER () AS global_best
                    FROM hit
                )
                SELECT id, file_id, seq, page, page_end, section,
                       kind, scope, text, filename, subject, indicators, keyword_score
                FROM ranked
                ORDER BY (rn = 1 AND subject_best >= global_best - %(band)s) DESC,
                         keyword_score DESC, file_id, seq
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

    def keyword_boundary_freq(
        self,
        keywords: list[str],
        *,
        file_ids: list[uuid.UUID] | None = None,
    ) -> tuple[int, dict[str, int]]:
        """返回（切片总数, {关键词: 后面不再是汉字的切片数}）。

        「合并利润」在库里全是「合并利润表」的前缀，子串计数会把它当成真词。
        四个字后面还是汉字的不计；标点、括号、数字或片段结束才算。
        """
        if not keywords:
            return 0, {}
        cols = ", ".join(
            f"count(*) FILTER (WHERE c.text ~ %(kw{i})s) AS df{i}"
            for i in range(len(keywords))
        )
        params: dict = {"file_ids": file_ids or None}
        for i, kw in enumerate(keywords):
            params[f"kw{i}"] = re.escape(kw) + "([^\\u4e00-\\u9fff]|$)"
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

