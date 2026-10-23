from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import complete_chat

logger = get_logger(__name__)

MERGE_SYSTEM = """
你是情景记忆合并器。把同一主题的多条近况合成一段完整、可检索的情景记忆。
规则：
1. 只输出合并后的正文，不要标题、前缀、解释、列表或引号；
2. 保留任务、技术栈、目标、进展等不重复关键事实，不要编造；
3. 去掉重复表述；
4. 写成 40～120 字的一两句话，不要压成口号。
"""


def _episode_norm(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", " \t\r\n。；;，,、."))


def merge_episode_text(old: str, new: str) -> str:
    old_s = old.strip()
    new_s = new.strip()
    if not old_s:
        return new_s
    if not new_s:
        return old_s
    old_n = _episode_norm(old_s)
    new_n = _episode_norm(new_s)
    if new_n == old_n:
        return old_s
    if len(new_n) >= 6 and new_n in old_n:
        return old_s
    if len(old_n) >= 6 and old_n in new_n:
        return new_s
    return f"{old_s.rstrip('。；;，,')}；{new_s}"


def merge_episode_texts(parts: list[str]) -> str:
    merged = ""
    for part in parts:
        merged = merge_episode_text(merged, part)
    return merged


def unique_episode_texts(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        text = part.strip()
        if not text:
            continue
        key = _episode_norm(text)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique


def compact_episode_texts(parts: list[str]) -> str:
    unique = unique_episode_texts(parts)
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    fallback = merge_episode_texts(unique)
    if fallback in unique:
        return fallback
    model = settings.llm_for_memory()
    numbered = "\n".join(f"- {item}" for item in unique)
    try:
        raw = complete_chat(
            [
                {"role": "system", "content": MERGE_SYSTEM},
                {"role": "user", "content": f"请合并这些情景记忆：\n{numbered}"},
            ],
            model=model,
            timeout=settings.memory_merge_timeout_seconds,
            max_retries=0,
            max_tokens=256,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    except LLMError as e:
        logger.warning("合并情景记忆失败，回退为拼接: %s", e)
        return fallback
    text = _clean_merged(raw)
    if not text or len(text) > 200:
        logger.warning("合并情景记忆结果不可用，回退为拼接")
        return fallback
    logger.info("情景记忆已合并 chars=%s", len(text))
    return text


def _clean_merged(raw: str) -> str:
    text = " ".join((raw or "").split()).strip()
    text = text.strip("\"'“”‘’")
    for prefix in ("合并后：", "合并结果：", "输出："):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text
