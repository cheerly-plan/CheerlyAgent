"""对话相关表：历史、工具结果池、会话、checkpoint。

由 storage/db.py 的 ensure_schema 按表拆出，SQL 逐字不变。
"""

from __future__ import annotations


def ensure_chat_history(conn, *, dim: int) -> None:
    """对话历史。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_chat_history (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            chat_session_id TEXT NOT NULL,
            seq INT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            tool_calls JSONB,
            tool_call_id TEXT,
            name TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (user_id, chat_session_id, seq)
        )
        """
    )
    # 存量库迁移：消息级附属信息（目前是知识库引用 sources）。
    # 单独一列而不是塞进 content：正文要原样回放给模型，附件信息
    # 只用于前端渲染，混在一起会让历史消息和模型输入失真
    conn.execute(
        """
        ALTER TABLE agent_chat_history
        ADD COLUMN IF NOT EXISTS meta JSONB
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_chat_history_tool_result_idx
        ON agent_chat_history (user_id, chat_session_id, tool_call_id)
        WHERE tool_call_id IS NOT NULL
        """
    )


def ensure_tool_result(conn, *, dim: int) -> None:
    """工具结果原文池。"""
    conn.execute(
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
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_tool_result_user_session_idx
        ON agent_tool_result (user_id, chat_session_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_tool_result_expires_idx
        ON agent_tool_result (expires_at)
        WHERE expires_at IS NOT NULL
        """
    )


def ensure_session(conn, *, dim: int) -> None:
    """会话。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_session (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '新会话',
            summary TEXT NOT NULL DEFAULT '',
            idempotency_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            deleted_at TIMESTAMPTZ,
            UNIQUE (user_id, idempotency_key)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_session_user_updated_idx
        ON agent_session (user_id, updated_at DESC)
        WHERE deleted_at IS NULL
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_session
        ALTER COLUMN title SET DEFAULT '新会话'
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_session
        ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT ''
        """
    )
    conn.execute(
        """
        ALTER TABLE agent_session
        ADD COLUMN IF NOT EXISTS summary_until_seq BIGINT NOT NULL DEFAULT 0
        """
    )


def ensure_checkpoint(conn, *, dim: int) -> None:
    """断点续跑 checkpoint。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_checkpoint (
            run_id          UUID PRIMARY KEY,
            user_id         TEXT NOT NULL,
            chat_session_id TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'running',
            step            INT NOT NULL DEFAULT 0,
            messages        JSONB NOT NULL,
            use_web_search  BOOLEAN NOT NULL DEFAULT TRUE,
            use_kb          BOOLEAN NOT NULL DEFAULT TRUE,
            use_report      BOOLEAN NOT NULL DEFAULT FALSE,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    # 存量库迁移：补建 use_web_search 列（断点续跑时沿用原 run 的开关状态）
    conn.execute(
        """
        ALTER TABLE agent_checkpoint
        ADD COLUMN IF NOT EXISTS use_web_search BOOLEAN NOT NULL DEFAULT TRUE
        """
    )
    # 同上：知识库工具开关（工具是否挂载要跟原 run 保持一致）
    conn.execute(
        """
        ALTER TABLE agent_checkpoint
        ADD COLUMN IF NOT EXISTS use_kb BOOLEAN NOT NULL DEFAULT TRUE
        """
    )
    # 同上：研报工具开关（内置技能），断点续跑时沿用原 run 的开关状态。
    # 默认 FALSE：只有前端明确传 use_report=true 的 run 才有研报工具。
    conn.execute(
        """
        ALTER TABLE agent_checkpoint
        ADD COLUMN IF NOT EXISTS use_report BOOLEAN NOT NULL DEFAULT FALSE
        """
    )
    # 启动时不存在本进程的进行中 run（单进程部署）：残留的 running
    # 一律是上次进程异常退出留下的僵尸，标为 interrupted（进程被杀
    # 属于中断而非失败，两者均可被 /chat/resume 认领续跑），
    # 否则既无法恢复又会触发 session 互斥把会话锁死
    conn.execute(
        """
        UPDATE agent_checkpoint
        SET status = 'interrupted', updated_at = now()
        WHERE status = 'running'
        """
    )
    # DB 级互斥兜底：同会话同时最多一个 running checkpoint，
    # 路由层预检（has_running_checkpoint）之后的并发竞态由它裁决
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS agent_checkpoint_running_uidx
        ON agent_checkpoint (user_id, chat_session_id)
        WHERE status = 'running'
        """
    )
