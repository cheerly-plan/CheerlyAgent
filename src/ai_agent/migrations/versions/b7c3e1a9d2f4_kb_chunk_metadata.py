"""kb chunk/file 检索元数据（主体 / 报告期 / 长尾）

Revision ID: b7c3e1a9d2f4
Revises: f0c1d2e3a4b5
Create Date: 2026-10-31 21:30:00.000000

之前检索只能按 file_id 过滤，再用单一 query 向量在 top_k 个名额里全局抢：
「A 和 B 的营业收入」这类对比问句里，离混合向量更近的那家会把名额吃满，
另一家一条都回不来。补上 subject 维度后，检索按主体分桶、够近的主体各保底
一个名额（带宽见 settings.kb_subject_quota_band）。

- 热维度（subject / period）成列：可直接索引、可 PARTITION BY 分桶
- 长尾（指标名、币种、表标题等）进 meta JSONB + GIN
- 切片冗余 subject/period：分桶排序要在单表上做，JOIN 文件表走不成索引；
  且「一份研报讲多家公司」需要切片级覆盖
"""

from typing import Sequence, Union

from alembic import op

revision: str = "b7c3e1a9d2f4"
down_revision: Union[str, Sequence[str], None] = "f0c1d2e3a4b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE agent_kb_file "
        "ADD COLUMN IF NOT EXISTS subject TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        "ALTER TABLE agent_kb_file "
        "ADD COLUMN IF NOT EXISTS subject_aliases TEXT[] NOT NULL DEFAULT '{}'"
    )
    op.execute(
        "ALTER TABLE agent_kb_file "
        "ADD COLUMN IF NOT EXISTS period TEXT NOT NULL DEFAULT ''"
    )
    op.execute("ALTER TABLE agent_kb_file ADD COLUMN IF NOT EXISTS meta JSONB")
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_kb_file_subject_idx "
        "ON agent_kb_file (subject)"
    )

    op.execute(
        "ALTER TABLE agent_kb_chunk "
        "ADD COLUMN IF NOT EXISTS subject TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        "ALTER TABLE agent_kb_chunk "
        "ADD COLUMN IF NOT EXISTS period TEXT NOT NULL DEFAULT ''"
    )
    op.execute("ALTER TABLE agent_kb_chunk ADD COLUMN IF NOT EXISTS meta JSONB")
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_kb_chunk_subject_idx "
        "ON agent_kb_chunk (subject)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_kb_chunk_meta_idx "
        "ON agent_kb_chunk USING gin (meta)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS agent_kb_chunk_meta_idx")
    op.execute("DROP INDEX IF EXISTS agent_kb_chunk_subject_idx")
    op.execute("ALTER TABLE agent_kb_chunk DROP COLUMN IF EXISTS meta")
    op.execute("ALTER TABLE agent_kb_chunk DROP COLUMN IF EXISTS period")
    op.execute("ALTER TABLE agent_kb_chunk DROP COLUMN IF EXISTS subject")

    op.execute("DROP INDEX IF EXISTS agent_kb_file_subject_idx")
    op.execute("ALTER TABLE agent_kb_file DROP COLUMN IF EXISTS meta")
    op.execute("ALTER TABLE agent_kb_file DROP COLUMN IF EXISTS period")
    op.execute("ALTER TABLE agent_kb_file DROP COLUMN IF EXISTS subject_aliases")
    op.execute("ALTER TABLE agent_kb_file DROP COLUMN IF EXISTS subject")
