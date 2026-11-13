"""raw tool result store

Revision ID: f7a8b9c0d1e2
Revises: e6f7a8b9c0d1
Create Date: 2026-11-13 14:40:00.000000

agent_chat_history.content 是模型当时看到的（已被单工具输出上限裁剪的）
版本；进程内 tool_result_pool 淘汰/重启后，read_tool_result 只能拿到它。
这里把裁剪前全文单独持久化，池子不在时仍可读到原始工具输出。
"""

from typing import Sequence, Union

from alembic import op

revision: str = "f7a8b9c0d1e2"
down_revision: Union[str, Sequence[str], None] = "e6f7a8b9c0d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_tool_result (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            chat_session_id TEXT NOT NULL,
            tool_call_id TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT 'tool',
            content TEXT NOT NULL DEFAULT '',
            meta JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at TIMESTAMPTZ,
            UNIQUE (user_id, chat_session_id, tool_call_id)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_tool_result_user_session_idx
        ON agent_tool_result (user_id, chat_session_id, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_tool_result_expires_idx
        ON agent_tool_result (expires_at)
        WHERE expires_at IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS agent_tool_result_expires_idx")
    op.execute("DROP INDEX IF EXISTS agent_tool_result_user_session_idx")
    op.execute("DROP TABLE IF EXISTS agent_tool_result")
