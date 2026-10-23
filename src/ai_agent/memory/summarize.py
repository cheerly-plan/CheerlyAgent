from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import complete_chat
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.short_term import crude_summary

logger = get_logger(__name__)

SUMMARY_SYSTEM = """
你是会话摘要器。把对话压缩成后续回复仍需要的上下文摘要。
规则：
1. 只输出摘要正文，不要标题、前缀或解释；
2. 保留关键事实、用户偏好、已做约定、未完成事项和重要结论；
3. 去掉寒暄、重复内容和工具调用细节；
4. 不要编造对话中没有的信息；
5. 使用简洁中文陈述句，控制在 400 字以内。
"""


def summarize_session(previous: str, overflow: list[ChatMessage]) -> str:
    source = _format_overflow(overflow)
    if not source.strip():
        return previous.strip()
    if previous.strip():
        user_content = f"已有摘要：\n{previous.strip()}\n\n新增对话：\n{source}"
    else:
        user_content = f"对话内容：\n{source}"
    model = settings.resolve_llm(settings.short_term_summary_model)
    try:
        raw = complete_chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            model=model,
            timeout=settings.short_term_summary_timeout_seconds,
            max_retries=0,
            max_tokens=512,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    except LLMError as e:
        logger.warning("生成会话摘要失败，回退为摘录: %s", e)
        return crude_summary(previous, overflow)
    text = " ".join((raw or "").split()).strip()
    if not text:
        return crude_summary(previous, overflow)
    logger.info("会话摘要已更新 chars=%s", len(text))
    return text


def _format_overflow(overflow: list[ChatMessage]) -> str:
    budget = max(int(settings.short_term_memory_max_tokens), 256)
    lines: list[str] = []
    used = 0
    for msg in overflow:
        line = _line(msg)
        if not line:
            continue
        remain = budget - used
        if remain <= 0:
            lines.append("…")
            break
        if len(line) > remain:
            lines.append(line[:remain] + "…")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def _line(msg: ChatMessage) -> str:
    role = msg.get("role")
    content = (msg.get("content") or "").strip()
    if role == "user":
        return f"用户：{content}" if content else ""
    if role == "assistant":
        calls = msg.get("tool_calls") or []
        names = ", ".join(call.get("name") or "tool" for call in calls)
        if names and content:
            return f"助手：{content}（调用工具 {names}）"
        if names:
            return f"助手：调用工具 {names}"
        return f"助手：{content}" if content else ""
    if role == "tool":
        name = msg.get("name") or "tool"
        snippet = content[:200]
        return f"工具[{name}]：{snippet}" if snippet else ""
    return ""
