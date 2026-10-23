from __future__ import annotations

import json
from typing import Any

from ai_agent.core.config import settings
from ai_agent.llm.client import complete_chat
from ai_agent.llm.types import ChatMessage

_MEMORY_EXTRA_BODY: dict[str, Any] = {
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
}


def complete_memory_chat(
    messages: list[ChatMessage],
    *,
    max_tokens: int,
) -> str:
    return complete_chat(
        messages,
        model=settings.llm_for_memory(),
        timeout=settings.memory_merge_timeout_seconds,
        max_retries=0,
        max_tokens=max_tokens,
        extra_body=_MEMORY_EXTRA_BODY,
    )


def load_llm_json(raw: str) -> object | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        blob = text[start : end + 1]
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return None
        blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None
