"""知识库文件元数据与状态机：CRUD、标签/主体别名、上传管道领取与恢复。（KbStore 的 mixin 之一，见 kb/store.py）。"""

from __future__ import annotations

import uuid
from psycopg.types.json import Jsonb
from pgvector import Vector
from ai_agent.kb.ingest.meta import (
    COMPANY_SUFFIX,
    extract_aliases,
    extract_file_meta,
)

# 方法体只需 self._get_pool()（来自 Database），不依赖其他 mixin。


class KbFileStore:
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
                       subject, period, subject_aliases, parser_version
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
        parser_version: str = "",
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
                    parser_version = COALESCE(NULLIF(%s, ''), parser_version),
                    updated_at = now()
                WHERE id = %s
                """,
                (
                    status,
                    error,
                    chunk_count,
                    text_count,
                    table_count,
                    parser_version,
                    file_id,
                ),
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


    def list_interrupted_kb_files(self) -> list[dict]:
        """上次进程退出时卡在解析／向量化，或可自动重试的失败文件。

        供启动恢复：磁盘上还有 PDF 的退回 uploaded 队列，不必删了重传。
        含「服务重启中断」与「解析子进程异常退出」（SIGSEGV 等），重启 worker
        应再试；真正业务失败（空文档等）不匹配这些前缀，不会死循环重投。
        """
        with self._get_pool().connection() as conn:
            return conn.execute(
                """
                SELECT id AS file_id, user_id, filename, status, error
                FROM agent_kb_file
                WHERE status IN ('parsing', 'embedding')
                   OR (
                        status = 'failed'
                        AND (
                            error LIKE '服务重启导致处理中断%'
                            OR error LIKE '解析子进程异常退出%'
                        )
                   )
                """
            ).fetchall()


    def list_false_missing_source_kb_files(self) -> list[dict]:
        """误标「源文件已不在」但磁盘目录仍在的失败档（供启动自愈）。"""
        with self._get_pool().connection() as conn:
            return conn.execute(
                """
                SELECT id AS file_id, user_id, filename, status, error
                FROM agent_kb_file
                WHERE status = 'failed'
                  AND error LIKE '服务重启导致处理中断，源文件已不在%'
                """
            ).fetchall()


    def claim_next_uploaded_kb_file(self) -> dict | None:
        """原子领取一份 uploaded 文件并标为 parsing。

        ``FOR UPDATE SKIP LOCKED``：多 ingest-worker 并发时不会抢同一行。
        """
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                WITH next AS (
                    SELECT id
                    FROM agent_kb_file
                    -- 兼容误写成 upload（少 ed）的脏数据
                    WHERE status IN ('uploaded', 'upload')
                    ORDER BY updated_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE agent_kb_file AS f
                SET status = 'parsing',
                    error = '',
                    updated_at = now()
                FROM next
                WHERE f.id = next.id
                RETURNING f.id AS file_id, f.user_id, f.filename
                """
            ).fetchone()
            conn.commit()
        return dict(row) if row else None


    def recover_stuck_kb_files(self, *, exclude_ids: list[uuid.UUID] | None = None) -> int:
        """把仍卡在中间态、且未能重投的文件标 failed。

        能重投的（磁盘 PDF 还在）由 pipeline 先拿走；这里只处理
        无源文件可恢复的残留，避免永远停在 parsing／embedding。
        """
        excluded = list(exclude_ids or [])
        with self._get_pool().connection() as conn:
            if excluded:
                cur = conn.execute(
                    """
                    UPDATE agent_kb_file
                    SET status = 'failed',
                        error = '服务重启导致处理中断，源文件已不在，请重新上传',
                        updated_at = now()
                    WHERE status IN ('parsing', 'embedding')
                      AND NOT (id = ANY(%s))
                    """,
                    (excluded,),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE agent_kb_file
                    SET status = 'failed',
                        error = '服务重启导致处理中断，源文件已不在，请重新上传',
                        updated_at = now()
                    WHERE status IN ('parsing', 'embedding')
                    """
                )
            conn.commit()
        return cur.rowcount
