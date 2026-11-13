"""token usage on chat messages

Revision ID: a9b0c1d2e3f4
Revises: f7a8b9c0d1e2
Create Date: 2026-11-13 19:20:00.000000

主对话走流式，网关的 usage 只落在最后一帧（choices 为空），此前整条链路
都没有采集，管理面板没法回答「这个会话烧了多少 token」。

这里在 agent_chat_history 上按消息记 prompt/completion 两个数：
- assistant 消息记该次请求的 prompt_tokens（含历史 + 工具 schema）与
  本次产出的 completion_tokens；
- 会话总量由 SUM 聚合，不再单独维护计数列，避免与消息级数据不一致。

历史消息一律为 NULL（不回填）：真实值只有网关知道，用本地 tokenizer
估算会与计费对不上，宁可不显示也不要显示错的。
"""

from typing import Sequence, Union

from alembic import op

revision: str = "a9b0c1d2e3f4"
down_revision: Union[str, Sequence[str], None] = "f7a8b9c0d1e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE agent_chat_history "
        "ADD COLUMN IF NOT EXISTS prompt_tokens INT"
    )
    op.execute(
        "ALTER TABLE agent_chat_history "
        "ADD COLUMN IF NOT EXISTS completion_tokens INT"
    )
    # 会话级聚合只看 assistant 行，且只关心 session 维度：部分索引让
    # SUM 不必扫全表（历史消息的 token 全为 NULL）。
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_chat_history_token_idx
        ON agent_chat_history (user_id, chat_session_id)
        WHERE prompt_tokens IS NOT NULL OR completion_tokens IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS agent_chat_history_token_idx")
    op.execute(
        "ALTER TABLE agent_chat_history DROP COLUMN IF EXISTS completion_tokens"
    )
    op.execute("ALTER TABLE agent_chat_history DROP COLUMN IF EXISTS prompt_tokens")