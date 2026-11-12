"""memory setting: per-user enable/disable switch

Revision ID: c3d4e5f6a7b8
Revises: f1a2b3c4d5e6
Create Date: 2026-11-12 23:50:00.000000

用户级记忆开关。关闭后不写新记忆、不注入旧记忆、不挂 memory 工具，
但已有数据保留，删除走 /v1/memory 的显式接口。
"""

from typing import Sequence, Union

from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_memory_setting (
            user_id TEXT PRIMARY KEY,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_memory_setting")