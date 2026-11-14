"""知识库文件与切片。

由 storage/db.py 的 ensure_schema 按表拆出，SQL 逐字不变。
"""

from __future__ import annotations


def ensure_kb_file(conn, *, dim: int) -> None:
    """知识库文件元数据。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_kb_file (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            size_bytes BIGINT NOT NULL,
            mime TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'uploaded',
            error TEXT NOT NULL DEFAULT '',
            chunk_count INT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kb_file_user_id_idx
        ON agent_kb_file (user_id)
        """
    )


def ensure_kb_chunk(conn, *, dim: int) -> None:
    """知识库切片。"""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS agent_kb_chunk (
            id UUID PRIMARY KEY,
            file_id UUID NOT NULL REFERENCES agent_kb_file(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL,
            seq INT NOT NULL,
            page INT,
            page_end INT,
            section TEXT,
            kind TEXT NOT NULL DEFAULT 'text',
            text TEXT NOT NULL,
            embedding vector({dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (file_id, seq)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kb_chunk_embedding_hnsw
        ON agent_kb_chunk USING hnsw (embedding vector_cosine_ops)
        """
    )
    # 既有库的增量列：CREATE IF NOT EXISTS 不会补列，靠 ALTER 兜底
    conn.execute(
        """
        ALTER TABLE agent_kb_chunk
        ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'text'
        """
    )
    # 跨页切片的末页：page 存起始页，page_end 存结束页（单页片两者
    # 相等）。引用要按「第134–135页」展示，只存起始页会误导溯源
    conn.execute(
        "ALTER TABLE agent_kb_chunk ADD COLUMN IF NOT EXISTS page_end INT"
    )
    # 所属一级节：「第十节 管理层讨论与分析」。检索/引用/图谱抽取
    # 都要用它做上下文与分组
    conn.execute(
        "ALTER TABLE agent_kb_chunk ADD COLUMN IF NOT EXISTS section TEXT"
    )
    # 报表口径（consolidated / parent）：同一指标在合并报表与母公司
    # 报表里各有一份数，检索要能按用户的默认预期（集团口径）区分
    conn.execute(
        "ALTER TABLE agent_kb_chunk ADD COLUMN IF NOT EXISTS scope TEXT"
    )


def ensure_kb_file_more1(conn, *, dim: int) -> None:
    """知识库文件元数据（续：后续补充的列/索引）。"""
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS graph_status TEXT NOT NULL DEFAULT ''
        """
    )
    # 实体对齐进度（P2）：'' 未跑 / aligning / ready / failed
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS align_status TEXT NOT NULL DEFAULT ''
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS alias_count INT NOT NULL DEFAULT 0
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS graph_entity_count INT NOT NULL DEFAULT 0
        """
    )
    # 建图进度：已完成批 / 总批。前端轮询文件列表就能画出百分比，
    # 不必另开一条进度接口。
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS graph_batch_done INT NOT NULL DEFAULT 0
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS graph_batch_total INT NOT NULL DEFAULT 0
        """
    )
    # 抽取配置：文档类型（决定注入哪套配置块）与粒度（完整/轻量）。
    # 存在文件上而不是每次建图现传：重建时不用再问一次，也让
    # 「这张图是按什么 prompt 抽的」有据可查。空串 = 用全局默认（general）
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS doc_type TEXT NOT NULL DEFAULT ''
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS extract_level TEXT NOT NULL DEFAULT 'full'
        """
    )
    # 管理能力：标签（读侧过滤/展示）与内容指纹（上传去重）
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT '{}'
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT ''
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kb_file_hash_idx
        ON agent_kb_file (content_hash)
        """
    )
    # 向量化完成时记下文本/表格片数，列表标题后面直接展示，
    # 不用再扫一遍切片表。存量文件从切片回填一次。
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS text_count INT NOT NULL DEFAULT 0
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS table_count INT NOT NULL DEFAULT 0
        """
    )
    # 文件内容描述：让模型通过 kb_list_files 工具判断要查哪几份文件，
    # 替代原先「目录注入」把文件名塞进对话上下文的做法
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS description TEXT NOT NULL DEFAULT ''
        """
    )
    conn.execute(
        """
        UPDATE agent_kb_file f
        SET text_count = s.text_n,
            table_count = s.table_n
        FROM (
            SELECT file_id,
                   count(*) FILTER (WHERE kind = 'text') AS text_n,
                   count(*) FILTER (WHERE kind = 'table') AS table_n
            FROM agent_kb_chunk
            GROUP BY file_id
        ) s
        WHERE f.id = s.file_id
          AND f.text_count = 0 AND f.table_count = 0
          AND f.chunk_count > 0
        """
    )
    # ——— 检索元数据（主体 / 报告期 / 长尾）———
    # 之前检索只能按 file_id 过滤，再靠单一 query 向量在 top_k 个名额里
    # 全局抢：「A 和 B 的营业收入」这种对比问句里，离混合向量更近的那家
    # 会把名额吃满，另一家一条都回不来。补上 subject 维度后，检索可以按
    # 主体分桶、每桶独立配额（见 kb/store.search_kb_chunks），这类问题
    # 才不需要靠模型「自觉拆成两次调用」来兜。
    # 热维度成列（可直接索引、可 PARTITION BY 分桶），长尾进 meta JSONB。
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS subject TEXT NOT NULL DEFAULT ''
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS subject_aliases TEXT[] NOT NULL DEFAULT '{}'
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS period TEXT NOT NULL DEFAULT ''
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS meta JSONB
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kb_file_subject_idx
        ON agent_kb_file (subject)
        """
    )


def ensure_kb_chunk_more1(conn, *, dim: int) -> None:
    """知识库切片（续：后续补充的列/索引）。"""
    conn.execute(
        """
        ALTER TABLE agent_kb_chunk
        ADD COLUMN IF NOT EXISTS subject TEXT NOT NULL DEFAULT ''
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_kb_chunk
        ADD COLUMN IF NOT EXISTS period TEXT NOT NULL DEFAULT ''
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_kb_chunk
        ADD COLUMN IF NOT EXISTS meta JSONB
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kb_chunk_subject_idx
        ON agent_kb_chunk (subject)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kb_chunk_meta_idx
        ON agent_kb_chunk USING gin (meta)
        """
    )


def ensure_kb_file_more2(conn, *, dim: int) -> None:
    """知识库文件元数据（续：后续补充的列/索引）。"""
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT ''
        """
    )
    conn.execute(
        f"""
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS summary_embedding vector({dim})
        """
    )
    # '' 未生成 / ready / failed / empty（无可用素材）
    conn.execute(
        """
        ALTER TABLE agent_kb_file
        ADD COLUMN IF NOT EXISTS summary_status TEXT NOT NULL DEFAULT ''
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kb_file_summary_hnsw
        ON agent_kb_file USING hnsw (summary_embedding vector_cosine_ops)
        """
    )
