from __future__ import annotations

import atexit
import re
from concurrent.futures import Future, ThreadPoolExecutor

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import complete_chat
from ai_agent.llm.types import ChatEvent
from ai_agent.memory.store import DEFAULT_SESSION_TITLE, store

logger = get_logger(__name__)

_TITLE_MAX_LEN = 16
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="session-title")
_pool_closed = False

TITLE_SYSTEM = """
你是会话标题摘要器。根据用户问题和助手回答，提取这次对话的核心主题生成标题。
规则：
1. 只输出标题本身，不要输出“标题：”等前缀；
2. 长度8至14个汉字，Node、SSR、CSR这类英文缩写视为一个整体词，不计入汉字数量；
3. 结合问题和回答的实际内容，简洁关键词化，禁止标点、引号、书名号；
4. 不能是半截句子，不要直接复述用户整句话；
5. 禁止回答问题、禁止附加任何解释；
6. 问题和回答都没有有效主题时，固定输出：新会话

示例：
用户：clinic.js 和 --inspect 有什么区别
助手：Clinic.js 做性能诊断，--inspect 做断点调试。
标题：Node性能诊断与调试对比

用户：帮我写一个登录接口
助手：可以用 JWT 校验账号密码，返回 token。
标题：登录接口实现需求

用户：SSR 和 CSR 怎么选
助手：SEO 和首屏用 SSR，强交互后台用 CSR。
标题：SSR与CSR渲染模式选型对比

仅输出标题，不要任何额外文字。
"""


def normalize_title(raw: str, fallback: str = "", question: str = "") -> str:
    text = _one_line(raw)
    if _copied_question(text, question):
        text = ""
    if not text:
        text = _one_line(fallback)
    if not text or _copied_question(text, question):
        text = _compress_question(question)
    if len(text) > _TITLE_MAX_LEN:
        text = text[:_TITLE_MAX_LEN]
    return text or DEFAULT_SESSION_TITLE


def generate_session_title(question: str, answer: str = "") -> str:
    model = settings.title_model or settings.short_term_summary_model or settings.model
    try:
        raw = complete_chat(
            [
                {"role": "system", "content": TITLE_SYSTEM},
                {"role": "user", "content": _title_source(question, answer)},
            ],
            model=model,
            timeout=settings.title_timeout_seconds,
            max_retries=0,
            max_tokens=48,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    except LLMError as e:
        logger.warning("生成会话标题失败，回退为压缩主题: %s", e)
        return normalize_title(_compress_question(question), question=question)
    title = normalize_title(raw, question=question)
    logger.info("生成会话标题 question=%s title=%s", question[:40], title)
    return title


def start_title_job(
    user_id: str, session_id: str, question: str, answer: str = ""
) -> Future[str | None] | None:
    try:
        session = store.get_session(user_id, session_id)
    except Exception:
        logger.exception("读取会话失败，跳过标题生成")
        return None
    if session is None or session.title != DEFAULT_SESSION_TITLE:
        return None
    return _pool.submit(_run_title_job, user_id, session_id, question, answer)


def take_title_event(
    future: Future[str | None] | None, session_id: str
) -> ChatEvent | None:
    if future is None or not future.done():
        return None
    return _title_event_from_future(future, session_id)


def wait_title_event(
    future: Future[str | None] | None,
    session_id: str,
    timeout: float,
) -> ChatEvent | None:
    if future is None:
        return None
    try:
        future.result(timeout=timeout)
    except TimeoutError:
        logger.warning("标题生成超时 session_id=%s", session_id)
        return None
    except Exception:
        logger.exception("等待标题生成失败")
        return None
    return _title_event_from_future(future, session_id)


def shutdown_title_jobs() -> None:
    global _pool_closed
    if _pool_closed:
        return
    _pool_closed = True
    _pool.shutdown(wait=False, cancel_futures=True)


atexit.register(shutdown_title_jobs)


def _run_title_job(
    user_id: str, session_id: str, question: str, answer: str
) -> str | None:
    title = generate_session_title(question, answer)
    try:
        return store.update_session_title(user_id, session_id, title)
    except Exception:
        logger.exception("更新会话标题失败")
        return None


def _title_source(question: str, answer: str) -> str:
    parts = [f"用户：{(question or '').strip()[:400]}"]
    reply = (answer or "").strip()
    if reply:
        parts.append(f"助手：{reply[:800]}")
    return "\n".join(parts)


def _title_event_from_future(
    future: Future[str | None], session_id: str
) -> ChatEvent | None:
    try:
        title = future.result()
    except Exception:
        logger.exception("标题任务异常")
        return None
    if not title:
        return None
    return {"type": "title_updated", "session_id": session_id, "title": title}


def _one_line(text: str) -> str:
    cleaned = (text or "").replace("\n", " ").strip()
    cleaned = cleaned.strip(" \t\"'`“”‘’《》【】[]()（）")
    return re.sub(r"\s+", " ", cleaned)


def _plain(text: str) -> str:
    return re.sub(r"\W+", "", text, flags=re.UNICODE)


def _copied_question(title: str, question: str) -> bool:
    a = _plain(title)
    b = _plain(question)
    if not a or not b:
        return False
    if a == b:
        return True
    longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
    return shorter in longer and len(shorter) / len(longer) >= 0.7


def _compress_question(question: str) -> str:
    text = _one_line(question)
    text = re.sub(
        r"^(你能|能不能|能否|可以帮我|帮我|请问|请你|请|怎么|如何)",
        "",
        text,
    ).strip()
    text = re.sub(r"[吗嘛呢啊呀么？?！!。.…]+$", "", text).strip()
    return text
