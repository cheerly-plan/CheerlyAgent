from collections.abc import Iterator

from ai_agent.agents.prompts import SYSTEM_PROMPT
from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import stream_chat
from ai_agent.llm.types import ChatEvent, ChatMessage, ChatResult
from ai_agent.memory.short_term import short_term_memory
from ai_agent.memory.store import store, user_scope
from ai_agent.tools.registry import execute, schemas

logger = get_logger(__name__)


def _profile_prompt(user_id: str) -> str:
    try:
        items = store.list_profile(user_id)
        logger.info("list_profile user_id=%s count=%s", user_id, len(items))
    except Exception:
        logger.exception("用户画像读取失败")
        return ""
    if not items:
        return ""
    lines = "\n".join(f"- {item.text}" for item in items)
    return f"用户画像（稳定偏好/约束）：\n{lines}"


def _system_prompt(user_id: str) -> str:
    block = _profile_prompt(user_id)
    if not block:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n{block}"


def run_stream(
    user_input: str, user_id: str = "default", session_id: str = "default"
) -> Iterator[ChatEvent]:
    messages = short_term_memory.load(session_id)
    system = {"role": "system", "content": _system_prompt(user_id)}
    if messages and messages[0]["role"] == "system":
        messages[0] = system
    else:
        messages = [system, *messages]
    messages.append({"role": "user", "content": user_input})
    try:
        yield from run_loop(messages, user_id)
    finally:
        short_term_memory.save(session_id, messages)


def run_loop(
    messages: list[ChatMessage], user_id: str = "default"
) -> Iterator[ChatEvent]:
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
        assistant: ChatMessage = {"role": "assistant", "content": result["content"]}
        if result["tool_calls"]:
            assistant["tool_calls"] = result["tool_calls"]
        messages.append(assistant)
        if not result["tool_calls"]:
            yield {"type": "done"}
            return
        for call in result["tool_calls"]:
            yield {"type": "tool_start", "name": call["name"]}
            with user_scope(user_id):
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


def run(user_input: str, user_id: str = "default", session_id: str = "default") -> str:
    parts: list[str] = []
    for event in run_stream(user_input, user_id, session_id):
        if event["type"] == "text":
            parts.append(event.get("content") or "")
        elif event["type"] == "error":
            return event.get("message") or "ERROR"
    return "".join(parts)
