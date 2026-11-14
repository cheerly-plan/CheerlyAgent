"""profile bitemporal columns

Revision ID: b1c2d3e4f5a6
Revises: a8b9c0d1e2f3
Create Date: 2026-11-14 22:50:00.000000

画像事实的双时间轴：``valid_from`` / ``valid_to``。

双时间轴 double time：
- 事实真实生效的时间区间（valid_from / valid_to）
- 写入数据库的时间（created_at / superseded_at，已有）

有了 valid 区间，「2024 年住在深圳、2025 年搬到上海」可以两条并存；
当前实现只标 superseded_at，历史事实被直接覆盖、问旧时间点的居住地
就答不出来。
"""

from typing import Sequence, Union

from alembic import op

revision: str = "b1c2d3e4f5a6"
down_revision: Union[str, Sequence[str], None] = "a8b9c0d1e2f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE agent_user_profile
        ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ
        """
    )
    op.execute(
        """
        ALTER TABLE agent_user_profile
        ADD COLUMN IF NOT EXISTS valid_to TIMESTAMPTZ
        """
    )
    # 存量行：写入时间即默认生效时间（event-time 未知时退化为 ingest-time）
    op.execute(
        """
        UPDATE agent_user_profile
        SET valid_from = created_at
        WHERE valid_from IS NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_user_profile_valid_idx
        ON agent_user_profile (user_id, facet, valid_from, valid_to)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS agent_user_profile_valid_idx")
    op.execute("ALTER TABLE agent_user_profile DROP COLUMN IF EXISTS valid_to")
    op.execute("ALTER TABLE agent_user_profile DROP COLUMN IF EXISTS valid_from")
