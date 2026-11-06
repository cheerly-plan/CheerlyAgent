"""记忆链路专用的轻量 LLM 调用封装。

与 kb / 图谱共用的 JSON 稳健解析已上移到 `llm/json_utils.py`；这里只剩
记忆链路自己的调用参数（轻模型、关思考、短超时——抽取与合并对延迟敏感，
失败即静默降级，不值得重试）。
"""

from __future__ import annotations

from typing import Any

from ai_agent.core.config import settings
from ai_agent.llm.client import complete_chat, lightweight_llm
from ai_agent.llm.types import ChatMessage

_MEMORY_EXTRA_BODY: dict[str, Any] = {
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "think": False,
}


def complete_memory_chat(
    messages: list[ChatMessage],
    *,
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
) -> str:
    model, client = lightweight_llm(settings.memory_model)
    return complete_chat(
        messages,
        model=model,
        timeout=settings.memory_merge_timeout_seconds,
        max_retries=0,
        max_tokens=max_tokens,
        client=client,
        extra_body=_MEMORY_EXTRA_BODY,
        response_format=response_format,
    )
