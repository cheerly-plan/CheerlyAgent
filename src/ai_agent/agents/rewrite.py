"""检索改写与知识库目录注入。

从 loop.py 拆出(S5)。本地小模型常「觉得自己知道」而不调 kb_search，所以这里
做两件事：把就绪文件名/标签/类型注入上下文让模型自决，以及多轮里省略主体时
把问句补全成自包含的检索词。首轮若模型仍不检索，由服务端代调一次。
"""

from __future__ import annotations

import re
import uuid

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.kb.doc_profiles import resolve_profile
from ai_agent.kb.retrieval import query_looks_on_topic
from ai_agent.llm.client import complete_chat
from ai_agent.llm.types import ChatMessage, ChatResult, ToolCall
from ai_agent.memory.store import store
from ai_agent.prompts.kb import (
    CATALOG_EMPTY,
    CATALOG_HEADER,
    CATALOG_REWRITE_HINT,
    CATALOG_RULES,
    build_catalog_rewrite_line,
)
from ai_agent.prompts.tasks import REWRITE_SYSTEM, build_rewrite_user

logger = get_logger(__name__)



def _is_kb_catalog(msg: ChatMessage) -> bool:
    meta = msg.get("meta")
    return isinstance(meta, dict) and bool(meta.get("kb_catalog"))


def _drop_old_kb_catalog(messages: list[ChatMessage]) -> None:
    """只保留本轮目录，避免多轮对话里叠多份文件列表。

    目录夹在历史中间。若已经记下落库起点 `persisted` 再删，后面的下标会
    前移，`messages[persisted:]` 会把本轮 user 问句切掉，刷新后会话记录丢失。
    """
    kept = [msg for msg in messages if not _is_kb_catalog(msg)]
    if len(kept) != len(messages):
        messages[:] = kept


def _format_kb_catalog(
    files: list[dict], *, original: str, rewritten: str
) -> str:
    lines = [CATALOG_RULES, ""]
    if rewritten and original and rewritten.strip() != original.strip():
        lines.append(build_catalog_rewrite_line(rewritten.strip()))
        lines.append(CATALOG_REWRITE_HINT)
        lines.append("")
    if not files:
        lines.append(CATALOG_EMPTY)
        return "\n".join(lines)
    lines.append(CATALOG_HEADER)
    for item in files:
        name = str(item.get("filename") or "").strip() or "(未命名)"
        tags = [
            str(tag).strip()
            for tag in (item.get("tags") or [])
            if str(tag).strip()
        ]
        extras: list[str] = []
        if tags:
            extras.append("标签：" + "、".join(tags))
        doc_type = str(item.get("doc_type") or "").strip()
        if doc_type:
            label = resolve_profile(doc_type).label
            if label:
                extras.append("类型：" + label)
        suffix = ("｜" + "｜".join(extras)) if extras else ""
        lines.append(f"- {name}{suffix}")
    return "\n".join(lines)


def _kb_catalog_message(original: str, rewritten: str) -> ChatMessage | None:
    """use_kb 时把就绪文件目录注入成一条 system，供模型决定查哪些文件。"""
    try:
        rows = store.list_kb_files()
    except Exception:
        logger.exception("kb 文件目录读取失败，本轮不注入目录")
        return None
    ready = [row for row in rows if row.get("status") == "ready"]
    ready = ready[: settings.kb_catalog_max_files]
    return {
        "role": "system",
        "content": _format_kb_catalog(ready, original=original, rewritten=rewritten),
        "meta": {"kb_catalog": True, "kb_ready_files": len(ready)},
    }


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


_RETRIEVAL_TOOLS = frozenset({"kb_search", "kg_search"})


def _has_retrieval_call(result: ChatResult | None) -> bool:
    if not result:
        return False
    return any(call.get("name") in _RETRIEVAL_TOOLS for call in result.get("tool_calls") or [])


def _forced_kb_search_call(query: str) -> ToolCall:
    return {
        "id": f"kb_force_{uuid.uuid4().hex[:12]}",
        "name": "kb_search",
        "arguments": {"query": query},
    }


_KB_OFFTOPIC = re.compile(
    r"天气|气象|气温|下雨|落雨|下雪|台风|暴雨|"
    r"带伞|雨伞|口罩|穿什么|穿啥|多少度|几度|冷不冷|热不热"
)
_KB_CHITCHAT = re.compile(
    r"^(你好|嗨|哈喽|在吗|早上好|中午好|晚上好|谢谢|再见)[！!。.?？\s]*$"
)


def _skip_forced_kb_search(query: str) -> bool:
    """天气/闲聊不要强行检索：关键词会误撞「深圳市顺丰投资」这类公司名。"""
    text = (query or "").strip()
    if not text:
        return True
    if _KB_CHITCHAT.match(text):
        return True
    if _KB_OFFTOPIC.search(text) and not query_looks_on_topic(text):
        return True
    return False


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
