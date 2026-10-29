import json
from collections.abc import Iterator
from typing import Any, Literal

from openai import APIStatusError, OpenAI

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.types import ChatMessage, ChatResult, ToolCall

logger = get_logger(__name__)

_client = OpenAI(
    base_url=settings.base_url,
    api_key=settings.api_key,
)

# 流式对话专用客户端：显式超时 + 关闭自动重试。
# 默认客户端 timeout=600s：LLM 挂死时一个并发槽 + SSE 线程被占最长 10 分钟，
# 并发槽全满后聊天接口整体 429。流式重试还会从断点重复 yield 已发内容，
# 必须关掉；彻底失败由上层走断点续跑兜底。
_stream_client = _client.with_options(
    timeout=settings.llm_stream_timeout_seconds,
    max_retries=0,
)

StreamItem = tuple[Literal["text"], str] | tuple[Literal["result"], ChatResult]

# 主对话也必须关思考链：Qwen3 等模型开着会把正文写进 reasoning，
# delta.content 一直是空，前端就只剩「已调用 xx」没有回答。
_THINKING_OFF = {
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
    # Ollama / Qwen3 原生字段；vLLM 会忽略
    "think": False,
}


def _status_of(exc: Exception) -> int | None:
    """取上游 HTTP 状态码，用于上层区分鉴权失败和临时故障。"""
    return exc.status_code if isinstance(exc, APIStatusError) else None


def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for msg in messages:
        item: dict[str, Any] = {"role": msg["role"]}
        if "content" in msg:
            item["content"] = msg["content"]
        if msg.get("tool_calls"):
            item["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    },
                }
                for call in msg["tool_calls"]
            ]
        if "tool_call_id" in msg:
            item["tool_call_id"] = msg["tool_call_id"]
        if "name" in msg:
            item["name"] = msg["name"]
        if msg.get("partial"):
            # 消息级续写标记：告诉服务端这条 assistant 是半截，
            # 从断点接着生成（仅出现在断点续跑的种子消息上）
            item["partial"] = True
        payload.append(item)
    return payload


def stream_chat(messages: list[ChatMessage], tools: list[dict]) -> Iterator[StreamItem]:
    try:
        stream = _stream_client.chat.completions.create(
            model=settings.model,
            messages=_to_openai_messages(messages),
            tools=tools,
            tool_choice="auto",
            stream=True,
            extra_body=_THINKING_OFF,
        )
    except Exception as e:
        raise LLMError(str(e), status_code=_status_of(e)) from e

    content_parts: list[str] = []
    tool_acc: dict[int, dict[str, str]] = {}
    leftover = ""
    reasoning_parts: list[str] = []

    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta and delta.content:
            content_parts.append(delta.content)
            yield ("text", delta.content)
        # 部分网关把整段正文只放在最后一帧的 message 上，delta.content 为空
        message = getattr(choice, "message", None)
        piece = getattr(message, "content", None) if message is not None else None
        if piece:
            leftover = piece
        reasoning = _delta_reasoning(delta)
        if reasoning:
            reasoning_parts.append(reasoning)
        for call in (delta.tool_calls if delta else None) or []:
            index = call.index if call.index is not None else 0
            item = tool_acc.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if call.id:
                item["id"] = call.id
            if call.function:
                if call.function.name:
                    item["name"] += call.function.name
                if call.function.arguments:
                    item["arguments"] += call.function.arguments

    tool_calls: list[ToolCall] = []
    for item in tool_acc.values():
        try:
            arguments = json.loads(item["arguments"] or "{}")
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        tool_calls.append(
            {
                "id": item["id"],
                "name": item["name"],
                "arguments": arguments,
            }
        )
    text = "".join(content_parts)
    if not text and leftover:
        text = leftover
        yield ("text", leftover)
    if not text and reasoning_parts:
        # 思考链没关干净时，至少把推理正文交给前端，避免气泡全空
        text = "".join(reasoning_parts)
        yield ("text", text)
        logger.warning("stream_chat 正文为空，回退 reasoning 长度=%d", len(text))
    if not text and not tool_calls:
        logger.warning("stream_chat 空正文且无工具调用")
    yield (
        "result",
        {"content": text, "tool_calls": tool_calls},
    )


def _delta_reasoning(delta: Any) -> str:
    """取出思考链字段。Qwen3 / 部分网关不走 delta.content。"""
    if delta is None:
        return ""
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(delta, attr, None)
        if value:
            return str(value)
    extra = getattr(delta, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = extra.get(key)
            if value:
                return str(value)
    return ""


def complete_chat(
    messages: list[ChatMessage],
    model: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
) -> str:
    options: dict[str, float | int] = {}
    if timeout is not None:
        options["timeout"] = timeout
    if max_retries is not None:
        options["max_retries"] = max_retries
    client = _client.with_options(**options) if options else _client
    payload: dict[str, Any] = {
        "model": model or settings.model,
        "messages": _to_openai_messages(messages),
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if extra_body is not None:
        payload["extra_body"] = extra_body
    try:
        response = client.chat.completions.create(**payload)
    except Exception as e:
        raise LLMError(str(e), status_code=_status_of(e)) from e
    if not response.choices:
        return ""
    return (response.choices[0].message.content or "").strip()
