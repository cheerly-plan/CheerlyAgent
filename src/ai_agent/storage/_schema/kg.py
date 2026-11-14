"""知识图谱实体 / 关系 / 提及 / 别名。

由 storage/db.py 的 ensure_schema 按表拆出，SQL 逐字不变。
"""

from __future__ import annotations


def ensure_kg_entity(conn, *, dim: int) -> None:
    """图谱实体。"""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS agent_kg_entity (
            id UUID PRIMARY KEY,
            file_id UUID NOT NULL REFERENCES agent_kb_file(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            norm_name TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'other',
            description TEXT NOT NULL DEFAULT '',
            embedding vector({dim}),
            mention_count INT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (file_id, norm_name, type)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kg_entity_file_idx
        ON agent_kg_entity (file_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kg_entity_norm_idx
        ON agent_kg_entity (norm_name)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kg_entity_embedding_hnsw
        ON agent_kg_entity USING hnsw (embedding vector_cosine_ops)
        """
    )


def ensure_kg_relation(conn, *, dim: int) -> None:
    """图谱关系。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_kg_relation (
            id UUID PRIMARY KEY,
            file_id UUID NOT NULL REFERENCES agent_kb_file(id) ON DELETE CASCADE,
            source_id UUID NOT NULL REFERENCES agent_kg_entity(id) ON DELETE CASCADE,
            target_id UUID NOT NULL REFERENCES agent_kg_entity(id) ON DELETE CASCADE,
            relation TEXT NOT NULL DEFAULT '相关',
            description TEXT NOT NULL DEFAULT '',
            weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            chunk_seq INT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kg_relation_file_idx
        ON agent_kg_relation (file_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kg_relation_source_idx
        ON agent_kg_relation (source_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kg_relation_target_idx
        ON agent_kg_relation (target_id)
        """
    )


def ensure_kg_mention(conn, *, dim: int) -> None:
    """图谱提及。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_kg_mention (
            entity_id UUID NOT NULL REFERENCES agent_kg_entity(id) ON DELETE CASCADE,
            file_id UUID NOT NULL REFERENCES agent_kb_file(id) ON DELETE CASCADE,
            chunk_seq INT NOT NULL,
            hits INT NOT NULL DEFAULT 1,
            PRIMARY KEY (entity_id, chunk_seq)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kg_mention_file_idx
        ON agent_kg_mention (file_id)
        """
    )


def ensure_kg_alias(conn, *, dim: int) -> None:
    """图谱实体别名。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_kg_alias (
            id UUID PRIMARY KEY,
            file_id UUID NOT NULL REFERENCES agent_kb_file(id) ON DELETE CASCADE,
            entity_id UUID NOT NULL REFERENCES agent_kg_entity(id) ON DELETE CASCADE,
            alias_name TEXT NOT NULL,
            alias_norm TEXT NOT NULL,
            method TEXT NOT NULL DEFAULT 'llm',
            score DOUBLE PRECISION NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (file_id, alias_norm)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_kg_alias_file_idx
        ON agent_kg_alias (file_id)
        """
    )
