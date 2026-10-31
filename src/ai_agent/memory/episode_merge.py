from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.memory.parse import complete_memory_chat
from ai_agent.prompts.memory import MERGE_SYSTEM, build_merge_user

logger = get_logger(__name__)


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
    numbered = "\n".join(f"- {item}" for item in unique)
    try:
        raw = complete_memory_chat(
            [
                {"role": "system", "content": MERGE_SYSTEM},
                {"role": "user", "content": build_merge_user(numbered)},
            ],
            max_tokens=256,
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
