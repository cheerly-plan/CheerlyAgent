"""add parser_version to agent_kb_file

Revision ID: d4e5f6a7b8c9
Revises: c2d8f5b1a7e3
Create Date: 2026-11-10 12:40:00.000000

解析器版本号：切片规则变了以后，存量文件不会自己变。记下每份文件是用哪版
解析器切的，`scripts/kb.py doctor` 才能直接列出「已过期、建议重解析」的清单，
不用再靠人工比对。空串 = 本迁移之前的旧文件。
"""

from typing import Sequence, Union

from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c2d8f5b1a7e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE agent_kb_file "
        "ADD COLUMN IF NOT EXISTS parser_version TEXT NOT NULL DEFAULT ''"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agent_kb_file DROP COLUMN IF EXISTS parser_version")