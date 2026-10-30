"""检索改写。

从 loop.py 拆出(S5)。本地小模型常「觉得自己知道」而不调 kb_search，这里负责
多轮里省略主体时把问句补全成自包含的检索词；知识库检索的触发与否完全交给模型
（或 kb_list_files 工具）决定，本模块不再注入目录、也不再强制首轮检索。
"""

from __future__ import annotations

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import complete_chat
from ai_agent.prompts.tasks import REWRITE_SYSTEM, build_rewrite_user

logger = get_logger(__name__)



def _rewrite_query(user_input: str, prev_turn: list[str]) -> str:
    """对话改写（补全 + 扩写）：结合上一轮完整对话，把当前追问改写成自包含的检索词。

    既允许补全省略的主体（指代消解），也允许扩写同义词/相关要点——检索词
    不追求还原用户原话，而是追求「能独立完成检索」。唯一保留的约束是：补全
    的主体要来自上一轮对话，不凭空编造无关专有名词。

    关键点：喂上一轮「用户问 + 助手答」两段，而不是只喂上一句用户原话——
    主体（公司名）常出现在最早那一问里，中间隔了若干句不带主体的追问；而
    助手上一轮的回答里通常已经带出主体全称，据此能更稳地把「本季度」这类
    指代补回正确的公司。

    失败/超时/无上文一律退回原话：检索增强是非阻塞、可降级的，不能让一次
    小模型调用挡住回答。改写后结果等于原句也退回，省掉无效变更。
    """
    if not settings.kb_rewrite_enabled:
        return user_input
    prev = [s.strip() for s in prev_turn if s and s.strip()]
    if not prev:
        return user_input
    model = settings.kb_rewrite_model or settings.resolve_llm()
    context = "\n".join(f"- {s}" for s in prev)
    try:
        out = complete_chat(
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {
                    "role": "user",
                    "content": build_rewrite_user(context, user_input),
                },
            ],
            model=model,
            timeout=settings.kb_rewrite_timeout_seconds,
            max_retries=0,
            max_tokens=256,
            temperature=settings.kb_rewrite_temperature,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    except LLMError as e:
        logger.warning("kb 对话改写失败，退回原话：%s", e)
        return user_input
    except Exception:
        logger.exception("kb 对话改写异常，退回原话")
        return user_input
    rewritten = (out or "").strip().strip("\"'“”‘’")
    if not _usable_rewrite(user_input, rewritten):
        if rewritten and rewritten != user_input.strip():
            logger.info("kb 对话改写结果不可用，退回原话：%r", rewritten[:80])
        return user_input
    logger.info("kb 对话改写：%r -> %r", user_input, rewritten)
    return rewritten


_REWRITE_ANSWER_MARKERS = (
    "根据知识库",
    "根据文档",
    "未明确提及",
    "建议访问",
    "请稍后",
    "作为AI",
    "我无法",
)


def _usable_rewrite(original: str, rewritten: str) -> bool:
    """改写模型偶发直接作答，那种段落不能当检索词。"""
    text = (rewritten or "").strip()
    orig = (original or "").strip()
    if not text or text == orig:
        return False
    if len(text) > 160:
        return False
    if text.count("。") >= 2:
        return False
    return not any(marker in text for marker in _REWRITE_ANSWER_MARKERS)
