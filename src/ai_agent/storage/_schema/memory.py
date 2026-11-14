"""记忆相关表：episode / 画像 / 候选 / 归档 / 开关。

由 storage/db.py 的 ensure_schema 按表拆出，SQL 逐字不变。
"""

from __future__ import annotations


def ensure_episode(conn, *, dim: int) -> None:
    """情景记忆 episode。"""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS agent_episode (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding vector({dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_episode_user_id_idx
        ON agent_episode (user_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_episode_embedding_hnsw
        ON agent_episode USING hnsw (embedding vector_cosine_ops)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_episode_created_at_idx
        ON agent_episode (created_at)
        """
    )


def ensure_user_profile(conn, *, dim: int) -> None:
    """用户画像。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_user_profile (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            text TEXT NOT NULL,
            facet TEXT NOT NULL DEFAULT 'other',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_user_profile
        ADD COLUMN IF NOT EXISTS facet TEXT NOT NULL DEFAULT 'other'
        """
    )
    # 渐进式记忆：画像条目的时效与覆盖
    #   expires_at    易变事实（职业/居住地）的软过期；NULL = 永不过期
    #   superseded_at 被新事实覆盖的时间；非空即失效，只在库里留档
    for column, definition in (
        ("expires_at", "TIMESTAMPTZ"),
        ("superseded_at", "TIMESTAMPTZ"),
    ):
        conn.execute(
            f"ALTER TABLE agent_user_profile "
            f"ADD COLUMN IF NOT EXISTS {column} {definition}"
        )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_user_profile_active_idx
        ON agent_user_profile (user_id, facet)
        WHERE superseded_at IS NULL
        """
    )
    # 历史遗留：全 facet 唯一索引会把多条独立偏好挤成一行，
    # 只保留互斥型 facet 的单行约束，其余 facet 允许多条共存
    conn.execute(
        "DROP INDEX IF EXISTS agent_user_profile_user_facet_uidx"
    )
    conn.execute(
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
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS agent_user_profile_mutex_uidx
        ON agent_user_profile (user_id, facet)
        WHERE facet IN ('name', 'language', 'locale', 'occupation')
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_user_profile_user_id_idx
        ON agent_user_profile (user_id)
        """
    )


def ensure_episode_more1(conn, *, dim: int) -> None:
    """情景记忆 episode（续：后续补充的列/索引）。"""
    if True:  # 渐进式记忆恒定开启：证据列与候选表始终建好
        for column, definition in (
            ("hit_count", "INT NOT NULL DEFAULT 0"),
            ("last_seen_at", "TIMESTAMPTZ"),
            ("last_decay_at", "TIMESTAMPTZ"),
            ("sessions_seen", "TEXT[] NOT NULL DEFAULT '{}'"),
        ):
            conn.execute(
                f"ALTER TABLE agent_episode "
                f"ADD COLUMN IF NOT EXISTS {column} {definition}"
            )
        conn.execute(
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
        # 第 5 步：人工审核轨迹。表可能在第 4 步已建过，所以用
        # ADD COLUMN IF NOT EXISTS 平滑升级，老库零迁移。
        for column, definition in (
            ("reviewed_by", "TEXT NOT NULL DEFAULT ''"),
            ("review_note", "TEXT NOT NULL DEFAULT ''"),
            ("reviewed_at", "TIMESTAMPTZ"),
        ):
            conn.execute(
                f"ALTER TABLE agent_memory_candidate "
                f"ADD COLUMN IF NOT EXISTS {column} {definition}"
            )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS agent_memory_candidate_user_status_idx
            ON agent_memory_candidate (user_id, status)
            """
        )


def ensure_memory_archive(conn, *, dim: int) -> None:
    """L4 蒸馏归档。"""
    if True:  # 恒定开启，没有总开关
        conn.execute(
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
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS agent_memory_archive_user_idx
            ON agent_memory_archive (user_id, archived_at DESC)
            """
        )


def ensure_memory_setting(conn, *, dim: int) -> None:
    """记忆开关。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_memory_setting (
            user_id TEXT PRIMARY KEY,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
