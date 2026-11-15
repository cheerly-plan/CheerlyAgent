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

# 关思考的厂商差异：
# - DashScope 兼容口认顶层 ``enable_thinking``；
# - 部分网关（如 auto 路由）把该字段限制为 True，传 False 直接 400；
# - ``chat_template_kwargs`` 与 ``think`` 在两类网关上都可用。
# 因此只下发后两者：关思考的效果一样，且不会被网关拒绝。
_MEMORY_EXTRA_BODY: dict[str, Any] = {
    "chat_template_kwargs": {"enable_thinking": False},
    "think": False,
}


def complete_memory_chat(
    messages: list[ChatMessage],
    *,
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
    force_json: bool = False,
) -> str:
    """记忆链路的补全。

    ``force_json``：调用方要的是结构化输出，但网关不一定接受
    ``response_format``。置 True 时对本地 Ollama 仍走
    ``format="json" + think=False``，避免回退成「开思考的自由文本」——
    那种模式推理会把 num_predict 烧光，正文返回空串。
    """
    model, client = lightweight_llm(settings.memory_model)
    return complete_chat(
        messages,
        model=model,
        timeout=settings.memory_merge_timeout_seconds,
        max_retries=0,
        max_tokens=max_tokens,
        client=client,
        # 0 温：抽取是结构化任务，措辞抖动会让「该记的记没记住」的评测
        # 拿不到可复现的数字（与 kb_graph_temperature 同一理由）
        temperature=settings.memory_temperature,
        extra_body=_MEMORY_EXTRA_BODY,
        response_format=response_format,
        force_json=force_json,
    )
