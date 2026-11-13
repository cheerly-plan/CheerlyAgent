"""summary cursor + tool result lookup index

Revision ID: e6f7a8b9c0d1
Revises: c3d4e5f6a7b8
Create Date: 2026-11-13 13:40:00.000000

1. agent_session.summary_until_seq：摘要已覆盖到的 chat seq。冷启动只加载
   seq 之后的消息，避免 summary + 全部原文重复进入上下文。
2. agent_chat_history 的 tool_call_id 索引：结果池淘汰/进程重启后，
   read_tool_result 可以按 call_id 回落到 DB 原文。
"""

from typing import Sequence, Union

from alembic import op

revision: str = "e6f7a8b9c0d1"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE agent_session
        ADD COLUMN IF NOT EXISTS summary_until_seq BIGINT NOT NULL DEFAULT 0
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_chat_history_tool_result_idx
        ON agent_chat_history (user_id, chat_session_id, tool_call_id)
        WHERE tool_call_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS agent_chat_history_tool_result_idx")
    op.execute(
        "ALTER TABLE agent_session DROP COLUMN IF EXISTS summary_until_seq"
    )
