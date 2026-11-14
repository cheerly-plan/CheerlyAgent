"""用户与认证令牌。

由 storage/db.py 的 ensure_schema 按表拆出，SQL 逐字不变。
"""

from __future__ import annotations


def ensure_user(conn, *, dim: int) -> None:
    """用户。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_user (
            id UUID PRIMARY KEY,
            username TEXT NOT NULL,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS agent_user_username_uidx
        ON agent_user (username)
        """
    )


def ensure_auth_token(conn, *, dim: int) -> None:
    """认证令牌。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_auth_token (
            token_hash TEXT PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES agent_user(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_auth_token_user_id_idx
        ON agent_auth_token (user_id)
        """
    )
