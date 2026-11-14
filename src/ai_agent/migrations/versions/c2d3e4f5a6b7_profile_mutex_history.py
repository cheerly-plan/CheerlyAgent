"""allow history rows for mutex facets

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-11-14 22:55:00.000000

互斥 facet（姓名/语言/地区/职业）原来靠「每个 facet 只有一行」的 upsert
保证唯一，代价是历史值被原地覆盖——「2024 年在深圳」永远查不到。

改成：唯一约束只覆盖**当前生效**的行（superseded_at IS NULL），
被覆盖的行保留为历史，带 valid_from / valid_to 区间。
"""

from typing import Sequence, Union

from alembic import op

revision: str = "c2d3e4f5a6b7"
down_revision: Union[str, Sequence[str], None] = "b1c2d3e4f5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS agent_user_profile_mutex_uidx")
    # 存量数据里互斥 facet 只可能有一行（旧约束保证），缩窄成部分唯一索引
    # 不会产生冲突。
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS agent_user_profile_mutex_active_uidx
        ON agent_user_profile (user_id, facet)
        WHERE facet IN ('name', 'language', 'locale', 'occupation')
          AND superseded_at IS NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS agent_user_profile_mutex_active_uidx")
    # 回滚需要重新回到「每 facet 一行」：只保留最新的一行
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
