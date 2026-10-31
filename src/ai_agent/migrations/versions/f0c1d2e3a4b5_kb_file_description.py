"""add description column to agent_kb_file

Revision ID: f0c1d2e3a4b5
Revises: e49f3a9e1a4c
Create Date: 2026-10-31 04:30:00.000000

知识库文件内容描述：让模型通过 kb_list_files 工具查看文件清单与描述，
自行判断要查哪几份文件，替代原先「目录注入」把文件名塞进对话上下文的做法。
"""

from typing import Sequence, Union

from alembic import op

revision: str = "f0c1d2e3a4b5"
down_revision: Union[str, Sequence[str], None] = "e49f3a9e1a4c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE agent_kb_file "
        "ADD COLUMN IF NOT EXISTS description TEXT NOT NULL DEFAULT ''"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agent_kb_file DROP COLUMN IF EXISTS description")
