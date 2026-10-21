import json
from collections.abc import Iterator
from typing import Any, Literal

from openai import OpenAI

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.llm.types import ChatMessage, ChatResult, ToolCall

_client = OpenAI(
    base_url=settings.base_url,
    api_key=settings.api_key,
)

StreamItem = tuple[Literal["text"], str] | tuple[Literal["result"], ChatResult]


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
        payload.append(item)
    return payload


def stream_chat(messages: list[ChatMessage], tools: list[dict]) -> Iterator[StreamItem]:
    try:
        stream = _client.chat.completions.create(
            model=settings.model,
            messages=_to_openai_messages(messages),
            tools=tools,
            tool_choice="auto",
            stream=True,
        )
    except Exception as e:
        raise LLMError(str(e)) from e

    content_parts: list[str] = []
    tool_acc: dict[int, dict[str, str]] = {}

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content_parts.append(delta.content)
            yield ("text", delta.content)
        for call in delta.tool_calls or []:
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
    yield (
        "result",
        {"content": "".join(content_parts), "tool_calls": tool_calls},
    )


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
        raise LLMError(str(e)) from e
    if not response.choices:
        return ""
    return (response.choices[0].message.content or "").strip()
