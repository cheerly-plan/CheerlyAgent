"""initial schema baseline

Revision ID: e49f3a9e1a4c
Revises:
Create Date: 2026-10-31 01:13:58.492042

把 memory/db.py 的 ensure_schema() 全部 DDL 平移为 Alembic baseline。
- 全部用 op.execute 透传原始 SQL，不改语义，保留 IF NOT EXISTS 幂等性
- vector 维度从 settings.embedding_dim 取（与运行时同源）
- 现有库（ensure_schema 已跑过）用 `alembic stamp head` 标记到此版本，不实际执行
- 全新库 `alembic upgrade head` 从零建出完整 schema
- 后续 schema 变更新增迁移版本，不再改本文件
"""

from typing import Sequence, Union

from alembic import op

from ai_agent.core.config import settings

revision: str = "e49f3a9e1a4c"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    dim = int(settings.embedding_dim)
    if dim <= 0 or dim > 20000:
        raise ValueError("EMBEDDING_DIM 非法")

    # ---- pgvector 扩展 ----
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ---- 情景记忆 episode ----
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS agent_episode (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding vector({dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_episode_user_id_idx ON agent_episode (user_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_episode_embedding_hnsw "
        "ON agent_episode USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_episode_created_at_idx ON agent_episode (created_at)"
    )

    # ---- 用户画像 profile ----
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_user_profile (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            text TEXT NOT NULL,
            facet TEXT NOT NULL DEFAULT 'other',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "ALTER TABLE agent_user_profile ADD COLUMN IF NOT EXISTS facet TEXT NOT NULL DEFAULT 'other'"
    )
    # 历史遗留：全 facet 唯一索引会把多条独立偏好挤成一行，只保留互斥型 facet 单行约束
    op.execute("DROP INDEX IF EXISTS agent_user_profile_user_facet_uidx")
    op.execute(
        """
        DELETE FROM agent_user_profile a
        WHERE a.facet IN ('name', 'language', 'locale', 'occupation')
          AND a.ctid NOT IN (
            SELECT DISTINCT ON (user_id, facet) ctid
            FROM agent_user_profile
            WHERE facet IN ('name', 'language', 'locale', 'occupation')
            ORDER BY user_id, facet, created_at DESC
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS agent_user_profile_mutex_uidx
        ON agent_user_profile (user_id, facet)
        WHERE facet IN ('name', 'language', 'locale', 'occupation')
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_user_profile_user_id_idx ON agent_user_profile (user_id)"
    )

    # ---- 聊天历史 ----
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_chat_history (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            chat_session_id TEXT NOT NULL,
            seq INT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            tool_calls JSONB,
            tool_call_id TEXT,
            name TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (user_id, chat_session_id, seq)
        )
        """
    )
    # 消息级附属信息（知识库引用 sources），单独一列不塞进 content
    op.execute("ALTER TABLE agent_chat_history ADD COLUMN IF NOT EXISTS meta JSONB")

    # ---- 会话 session ----
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_session (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '新会话',
            summary TEXT NOT NULL DEFAULT '',
            idempotency_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            deleted_at TIMESTAMPTZ,
            UNIQUE (user_id, idempotency_key)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_session_user_updated_idx
        ON agent_session (user_id, updated_at DESC)
        WHERE deleted_at IS NULL
        """
    )
    op.execute("ALTER TABLE agent_session ALTER COLUMN title SET DEFAULT '新会话'")
    op.execute("ALTER TABLE agent_session ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT ''")

    # ---- 用户与令牌 ----
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_user (
            id UUID PRIMARY KEY,
            username TEXT NOT NULL,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS agent_user_username_uidx ON agent_user (username)")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_auth_token (
            token_hash TEXT PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES agent_user(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS agent_auth_token_user_id_idx ON agent_auth_token (user_id)")

    # ---- 断点续跑 checkpoint ----
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_checkpoint (
            run_id          UUID PRIMARY KEY,
            user_id         TEXT NOT NULL,
            chat_session_id TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'running',
            step            INT NOT NULL DEFAULT 0,
            messages        JSONB NOT NULL,
            use_web_search  BOOLEAN NOT NULL DEFAULT TRUE,
            use_kb          BOOLEAN NOT NULL DEFAULT TRUE,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "ALTER TABLE agent_checkpoint ADD COLUMN IF NOT EXISTS use_web_search BOOLEAN NOT NULL DEFAULT TRUE"
    )
    op.execute(
        "ALTER TABLE agent_checkpoint ADD COLUMN IF NOT EXISTS use_kb BOOLEAN NOT NULL DEFAULT TRUE"
    )
    # 启动时残留 running 一律是上次进程异常退出的僵尸，标 interrupted
    op.execute(
        "UPDATE agent_checkpoint SET status = 'interrupted', updated_at = now() WHERE status = 'running'"
    )
    # DB 级互斥兜底：同会话同时最多一个 running checkpoint
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS agent_checkpoint_running_uidx
        ON agent_checkpoint (user_id, chat_session_id)
        WHERE status = 'running'
        """
    )

    # ---- 知识库文件与切片 ----
    op.execute(
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
    op.execute("CREATE INDEX IF NOT EXISTS agent_kb_file_user_id_idx ON agent_kb_file (user_id)")
    op.execute(
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
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_kb_chunk_embedding_hnsw "
        "ON agent_kb_chunk USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("ALTER TABLE agent_kb_chunk ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'text'")
    op.execute("ALTER TABLE agent_kb_chunk ADD COLUMN IF NOT EXISTS page_end INT")
    op.execute("ALTER TABLE agent_kb_chunk ADD COLUMN IF NOT EXISTS section TEXT")
    op.execute("ALTER TABLE agent_kb_chunk ADD COLUMN IF NOT EXISTS scope TEXT")

    # ---- 知识图谱：实体/关系/提及/别名 ----
    op.execute(
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
    op.execute("CREATE INDEX IF NOT EXISTS agent_kg_entity_file_idx ON agent_kg_entity (file_id)")
    op.execute("CREATE INDEX IF NOT EXISTS agent_kg_entity_norm_idx ON agent_kg_entity (norm_name)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_kg_entity_embedding_hnsw "
        "ON agent_kg_entity USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
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
    op.execute("CREATE INDEX IF NOT EXISTS agent_kg_relation_file_idx ON agent_kg_relation (file_id)")
    op.execute("CREATE INDEX IF NOT EXISTS agent_kg_relation_source_idx ON agent_kg_relation (source_id)")
    op.execute("CREATE INDEX IF NOT EXISTS agent_kg_relation_target_idx ON agent_kg_relation (target_id)")
    op.execute(
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
    op.execute("CREATE INDEX IF NOT EXISTS agent_kg_mention_file_idx ON agent_kg_mention (file_id)")
    op.execute(
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
    op.execute("CREATE INDEX IF NOT EXISTS agent_kg_alias_file_idx ON agent_kg_alias (file_id)")

    # ---- 文件表增量列：图谱进度/对齐进度/抽取配置/标签/指纹/片数 ----
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS graph_status TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS align_status TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS alias_count INT NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS graph_entity_count INT NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS graph_batch_done INT NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS graph_batch_total INT NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS doc_type TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS extract_level TEXT NOT NULL DEFAULT 'full'")
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT '{}'")
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT ''")
    op.execute("CREATE INDEX IF NOT EXISTS agent_kb_file_hash_idx ON agent_kb_file (content_hash)")
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS text_count INT NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS table_count INT NOT NULL DEFAULT 0")
    # 存量文件从切片回填片数
    op.execute(
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


def downgrade() -> None:
    """baseline 不可逆：drop 全表会丢失全部业务数据，生产环境慎用。

    仅用于本地开发重置。按外键依赖反序级联 drop。
    """
    op.execute("DROP TABLE IF EXISTS agent_kg_alias CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_kg_mention CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_kg_relation CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_kg_entity CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_kb_chunk CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_kb_file CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_checkpoint CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_auth_token CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_user CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_session CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_chat_history CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_user_profile CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_episode CASCADE")
    # 不 drop extension：其他库可能也在用
