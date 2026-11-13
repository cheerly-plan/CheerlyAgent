"""通用工具结果读取：按 tool_call_id 从当前上下文取回任意工具的完整输出。

与 ``kb_read`` 的分工：
- ``read_tool_result`` 读的是**已有工具结果**，不重新检索，适用于天气、联网
  搜索、记忆查询、研报状态等所有工具；
- ``kb_read`` 读的是**知识库当前内容**，按「文件名 · 第N页」取回原文并补
  上下文/引用，两者互不替代。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai_agent.prompts.tools import (
    READ_TOOL_RESULT_DESC,
    READ_TOOL_RESULT_ID_DESC,
    READ_TOOL_RESULT_MAX_CHARS_DESC,
    READ_TOOL_RESULT_OFFSET_DESC,
)
from ai_agent.core.logging import get_logger
from ai_agent.llm import metrics
from ai_agent.storage.db import current_user_id
from ai_agent.tools.base import Tool
from ai_agent.tools.context import current_session_id, get_tool_result
from ai_agent.tools.result_pool import tool_result_pool

logger = get_logger(__name__)


class ReadToolResultArgs(BaseModel):
    tool_call_id: str = Field(description=READ_TOOL_RESULT_ID_DESC)
    offset: int = Field(default=0, ge=0, description=READ_TOOL_RESULT_OFFSET_DESC)
    max_chars: int = Field(
        default=4000,
        ge=200,
        le=6000,
        description=READ_TOOL_RESULT_MAX_CHARS_DESC,
    )


def read_tool_result(
    tool_call_id: str,
    offset: int = 0,
    max_chars: int = 4000,
) -> str:
    call_id = (tool_call_id or "").strip()
    if not call_id:
        return "请提供 tool_call_id，例如 read_tool_result(tool_call_id=\"call_abc\")。"

    # 查找顺序：进程池 → raw 工具结果表 → 当前 messages → 历史消息。
    # raw 表保存裁剪前全文；历史消息里的 content 可能已被单工具上限裁剪。
    pooled = tool_result_pool.get(
        current_user_id(), current_session_id(), call_id
    )
    from_pool = pooled is not None
    from_raw = False
    if pooled is not None:
        metrics.incr(metrics.TOOL_RESULT_POOL_HIT)
        ref = {
            "name": pooled.get("name") or "tool",
            "content": pooled.get("content") or "",
            "truncated": bool(pooled.get("truncated")),
        }
    else:
        try:
            from ai_agent.storage.store import store

            raw_row = store.load_raw_tool_result(
                current_user_id(), current_session_id(), call_id
            )
        except Exception:
            logger.exception("工具原文表查询失败 call_id=%s", call_id)
            raw_row = None
        if raw_row is not None:
            metrics.incr(metrics.TOOL_RESULT_RAW_HIT)
            from_raw = True
            ref = {
                "name": raw_row.get("name") or "tool",
                "content": raw_row.get("content") or "",
                "truncated": bool(raw_row.get("truncated")),
            }
        else:
            ref = get_tool_result(call_id)
            if ref is None:
                # 最后才回落到历史消息（模型当时看到的裁剪版）
                try:
                    from ai_agent.storage.store import store

                    db_row = store.load_tool_result(
                        current_user_id(), current_session_id(), call_id
                    )
                except Exception:
                    logger.exception(
                        "工具结果历史兜底查询失败 call_id=%s", call_id
                    )
                    db_row = None
                if db_row is None:
                    metrics.incr(metrics.TOOL_RESULT_POOL_MISS)
                    return (
                        f"没有找到 tool_call_id={call_id} 的工具结果；"
                        "它可能已被整轮裁剪，请重新调用原工具。"
                    )
                metrics.incr(metrics.TOOL_RESULT_DB_HIT)
                ref = {
                    "name": db_row.get("name") or "tool",
                    "content": db_row.get("content") or "",
                }

    content = ref.get("content") or ""
    name = ref.get("name") or "tool"
    total = len(content)
    if total == 0:
        return f"[{name} · {call_id}] 该工具结果为空。"

    start = max(0, offset)
    limit = max(200, min(int(max_chars), 6000))
    if start >= total:
        return (
            f"[{name} · {call_id}] 已到末尾；原文共 {total} 字符，"
            f"offset={start} 超出范围。"
        )

    chunk = content[start : start + limit]
    end = start + len(chunk)
    body = f"[{name} · {call_id}]\n{chunk}"
    if end < total:
        body += (
            f"\n\n[已读 {end}/{total} 字符；继续读用 "
            f"read_tool_result(tool_call_id=\"{call_id}\", offset={end})]"
        )
    if (from_pool or from_raw) and ref.get("truncated"):
        body += "\n[快照已超过单条上限并截断]"
    return body


read_tool_result_tool = Tool(
    name="read_tool_result",
    description=READ_TOOL_RESULT_DESC,
    func=read_tool_result,
    args_model=ReadToolResultArgs,
    max_output_chars=6000,
)
