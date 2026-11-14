"""episode related_ids

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-11-14 23:30:00.000000

情景记忆关联链接（A-MEM 风格）：相似但没达到合并阈值的条目互相记 id，
召回时一跳扩展。与 merge 互补——merge 处理「同一件事重复说」，
link 保留「不同但相关的事」（如「下周去深圳出差」↔「华南业务提醒」）。
"""

from typing import Sequence, Union

from alembic import op

revision: str = "d3e4f5a6b7c8"
down_revision: Union[str, Sequence[str], None] = "c2d3e4f5a6b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE agent_episode
        ADD COLUMN IF NOT EXISTS related_ids TEXT[] NOT NULL DEFAULT '{}'
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE agent_episode DROP COLUMN IF EXISTS related_ids")
