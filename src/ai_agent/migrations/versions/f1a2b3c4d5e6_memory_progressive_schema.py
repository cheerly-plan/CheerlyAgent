"""memory progressive schema: profile expiry, episode evidence, candidate, archive

Revision ID: f1a2b3c4d5e6
Revises: d4e5f6a7b8c9
Create Date: 2026-11-12 23:30:00.000000

渐进式记忆第 1–5 步的 schema 之前只写在 ``storage/db.py::ensure_schema()``，
但应用启动已经改成只跑 Alembic（``api/app.py`` 的 ``run_migrations()``）。
于是全新库上：

* ``agent_user_profile.expires_at / superseded_at`` 不存在 → 画像写入直接报错；
* ``agent_episode.hit_count / last_seen_at / last_decay_at / sessions_seen``
  不存在 → 情景写入、检索、清理全部静默失败；
* ``agent_memory_candidate / agent_memory_archive`` 不存在 → 候选与归档不可用。

本迁移把 ``ensure_schema()`` 里这些 DDL 原样平移过来，全部使用
``IF NOT EXISTS`` 保持幂等：已经跑过 ensure_schema 的老库可以安全执行，
全新库则从零建齐。以后记忆相关的 schema 变更也必须走新的 Alembic revision。
"""

from typing import Sequence, Union

from alembic import op

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ---- 画像：软过期与覆盖留档 ----
    op.execute(
        "ALTER TABLE agent_user_profile "
        "ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ"
    )
    op.execute(
        "ALTER TABLE agent_user_profile "
        "ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_user_profile_active_idx "
        "ON agent_user_profile (user_id, facet) "
        "WHERE superseded_at IS NULL"
    )

    # ---- 情景：观察态证据列 ----
    for column, definition in (
        ("hit_count", "INT NOT NULL DEFAULT 0"),
        ("last_seen_at", "TIMESTAMPTZ"),
        ("last_decay_at", "TIMESTAMPTZ"),
        ("sessions_seen", "TEXT[] NOT NULL DEFAULT '{}'"),
    ):
        op.execute(
            f"ALTER TABLE agent_episode "
            f"ADD COLUMN IF NOT EXISTS {column} {definition}"
        )

    # ---- L3 候选层 ----
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_memory_candidate (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            text TEXT NOT NULL,
            facet TEXT NOT NULL DEFAULT 'other',
            source TEXT NOT NULL DEFAULT '',
            hit_count INT NOT NULL DEFAULT 0,
            session_seen INT NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    # 第 5 步人工审核轨迹；表可能更早建过，所以逐列 ADD COLUMN IF NOT EXISTS
    for column, definition in (
        ("reviewed_by", "TEXT NOT NULL DEFAULT ''"),
        ("review_note", "TEXT NOT NULL DEFAULT ''"),
        ("reviewed_at", "TIMESTAMPTZ"),
    ):
        op.execute(
            f"ALTER TABLE agent_memory_candidate "
            f"ADD COLUMN IF NOT EXISTS {column} {definition}"
        )
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_memory_candidate_user_status_idx "
        "ON agent_memory_candidate (user_id, status)"
    )

    # ---- L4 蒸馏归档 ----
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_memory_archive (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            origin_id TEXT NOT NULL,
            text TEXT NOT NULL,
            facet TEXT NOT NULL DEFAULT 'other',
            source TEXT NOT NULL DEFAULT '',
            hit_count INT NOT NULL DEFAULT 0,
            session_seen INT NOT NULL DEFAULT 0,
            reviewed_by TEXT NOT NULL DEFAULT '',
            archived_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (user_id, kind, origin_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS agent_memory_archive_user_idx "
        "ON agent_memory_archive (user_id, archived_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS agent_memory_archive_user_idx")
    op.execute("DROP TABLE IF EXISTS agent_memory_archive CASCADE")
    op.execute("DROP INDEX IF EXISTS agent_memory_candidate_user_status_idx")
    op.execute("DROP TABLE IF EXISTS agent_memory_candidate CASCADE")
    for column in ("sessions_seen", "last_decay_at", "last_seen_at", "hit_count"):
        op.execute(
            f"ALTER TABLE agent_episode DROP COLUMN IF EXISTS {column}"
        )
    op.execute("DROP INDEX IF EXISTS agent_user_profile_active_idx")
    op.execute(
        "ALTER TABLE agent_user_profile DROP COLUMN IF EXISTS superseded_at"
    )
    op.execute(
        "ALTER TABLE agent_user_profile DROP COLUMN IF EXISTS expires_at"
    )