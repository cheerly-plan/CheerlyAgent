"""chat history embedding index

Revision ID: a8b9c0d1e2f3
Revises: f7a8b9c0d1e2
Create Date: 2026-11-14 22:30:00.000000

历史对话语义检索：search_history 先走向量召回，解决「换个说法就搜不到」
的问题。正文仍在 agent_chat_history，本表只存向量与用于展示的文本副本。
"""

from typing import Sequence, Union

from alembic import op

revision: str = "a8b9c0d1e2f3"
down_revision: Union[str, Sequence[str], None] = "a9b0c1d2e3f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 维度跟 embedding_dim 走；迁移里写死 1024 会在换模型时静默错配，
    # 这里读一次运行配置。
    from ai_agent.core.config import settings

    dim = int(settings.embedding_dim)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS agent_chat_embedding (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            chat_session_id TEXT NOT NULL,
            seq INT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding vector({dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (user_id, chat_session_id, seq)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_chat_embedding_session_idx
        ON agent_chat_embedding (user_id, chat_session_id, seq)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_chat_embedding_hnsw
        ON agent_chat_embedding USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS agent_chat_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS agent_chat_embedding_session_idx")
    op.execute("DROP TABLE IF EXISTS agent_chat_embedding")
