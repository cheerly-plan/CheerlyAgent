"""kb file summary index for document routing

Revision ID: c2d8f5b1a7e3
Revises: b7c3e1a9d2f4
Create Date: 2026-10-31 23:58:00.000000

给每份文件建一条文档级摘要向量，让「该查哪几份文件」在检索层解决：
以前模型得对着注入的【知识库可用文件】清单猜 files（文件一多就撑爆
上下文且选不准），现在按摘要向量做语义路由，files 不再是模型负担。

- summary：摘要文本（可读，管理页与 dry-run 都看它）
- summary_embedding：摘要向量，独立 hnsw 索引
- summary_status：'' 未生成 / ready / failed / empty
"""

from typing import Sequence, Union

from alembic import op

from ai_agent.core.config import settings

revision: str = "c2d8f5b1a7e3"
down_revision: Union[str, Sequence[str], None] = "b7c3e1a9d2f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    dim = int(settings.embedding_dim)
    if dim <= 0 or dim > 20000:
        raise ValueError("EMBEDDING_DIM 非法")
    op.execute(
        "ALTER TABLE agent_kb_file "
        "ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        f"ALTER TABLE agent_kb_file "
        f"ADD COLUMN IF NOT EXISTS summary_embedding vector({dim})"
    )
    op.execute(
        "ALTER TABLE agent_kb_file "
        "ADD COLUMN IF NOT EXISTS summary_status TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_kb_file_summary_hnsw "
        "ON agent_kb_file USING hnsw (summary_embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS agent_kb_file_summary_hnsw")
    op.execute("ALTER TABLE agent_kb_file DROP COLUMN IF EXISTS summary_embedding")
    op.execute("ALTER TABLE agent_kb_file DROP COLUMN IF EXISTS summary_status")
    op.execute("ALTER TABLE agent_kb_file DROP COLUMN IF EXISTS summary")
