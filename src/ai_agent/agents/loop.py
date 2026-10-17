from collections.abc import Iterator

from ai_agent.agents.prompts import SYSTEM_PROMPT
from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import stream_chat
from ai_agent.llm.types import ChatEvent, ChatMessage, ChatResult
from ai_agent.tools.registry import execute, schemas

logger = get_logger(__name__)


def run_stream(user_input: str) -> Iterator[ChatEvent]:
    messages: list[ChatMessage] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    for step in range(settings.max_turns):
        result: ChatResult | None = None
        try:
            for kind, payload in stream_chat(messages, schemas()):
                if kind == "text":
                    yield {"type": "text", "content": payload}
                else:
                    result = payload
        except LLMError as e:
            yield {"type": "error", "message": str(e)}
            return

        if result is None:
            yield {"type": "error", "message": "模型没有返回结果"}
            return

        logger.info("step %s tool_calls: %s", step, result["tool_calls"])
        if not result["tool_calls"]:
            yield {"type": "done"}
            return

        messages.append(
            {
                "role": "assistant",
                "content": result["content"],
                "tool_calls": result["tool_calls"],
            }
        )
        for call in result["tool_calls"]:
            yield {"type": "tool_start", "name": call["name"]}
            output = execute(call["name"], call["arguments"])
            yield {"type": "tool_end", "name": call["name"]}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": output,
                }
            )
    yield {"type": "error", "message": f"已超过最大轮数 {settings.max_turns}。"}


def run(user_input: str) -> str:
    parts: list[str] = []
    for event in run_stream(user_input):
        if event["type"] == "text":
            parts.append(event.get("content") or "")
        elif event["type"] == "error":
            return event.get("message") or "ERROR"
    return "".join(parts)
