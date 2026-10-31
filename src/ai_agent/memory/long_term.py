from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import json_response_format
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.facets import (
    coalesce_by_facet,
    format_profile_line,
    normalize_facet,
)
from ai_agent.memory.gate import jev_noul
from ai_agent.memory.parse import complete_memory_chat, load_llm_json
from ai_agent.memory.profile import ProfileAction, apply_profile_action
from ai_agent.memory.store import store
from ai_agent.prompts.memory import (
    EXTRACT_SYSTEM,
    GATE_INSTRUCTIONS,
    LABEL_CURRENT_TURN,
    LABEL_EXISTING_EPISODES,
    LABEL_EXISTING_EPISODES_EMPTY,
    LABEL_EXISTING_PROFILE,
    LABEL_EXISTING_PROFILE_EMPTY,
    LABEL_SESSION_SUMMARY,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExtractedMemory:
    kind: Literal["profile", "episode"]
    content: str
    op: Literal["add", "update", "delete"] = "add"
    facet: str = "other"


def capture_long_term(
    user_id: str,
    turn: list[ChatMessage],
    summary: str,
    user_input: str,
) -> None:
    if not any(msg.get("role") == "assistant" for msg in turn):
        return
    # Jev 门控：命中阈值以下直接跳过，省掉向量检索 + 小模型抽取；
    # 未配 key 或门控失败时放行，行为与旧版一致
    if _gate_blocks(turn, summary):
        return
    existing_episodes: list[str] = []
    try:
        existing_episodes = [
            item.text for item in store.search_episode(user_input, k=3, user_id=user_id)
        ]
    except Exception:
        logger.exception("相近情景记忆检索失败")
    existing_profile: list[str] = []
    try:
        existing_profile = [
            format_profile_line(item) for item in store.list_profile(user_id)
        ]
    except Exception:
        logger.exception("用户画像读取失败")
    items = extract_turn_memories(
        turn,
        summary,
        existing_episodes=existing_episodes,
        existing_profile=existing_profile,
    )
    for item in items:
        _write_extracted(user_id, item)


def extract_turn_memories(
    turn: list[ChatMessage],
    summary: str = "",
    existing_episodes: list[str] | None = None,
    existing_profile: list[str] | None = None,
) -> list[ExtractedMemory]:
    source = _format_turn(turn)
    if not source.strip():
        return []
    if not any(msg.get("role") == "assistant" for msg in turn):
        return []
    user_content = _build_user_content(
        source, summary, existing_episodes, existing_profile
    )
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    # JSON 模式（MEMORY_EXTRACT_JSON_MODE=json_object）：网关不支持会直接 400，
    # 这时退回「prompt 约束 + load_llm_json 兜底」再跑一次，不让开关整条链路
    # 挂掉。默认 off，行为与开启前一致。
    fmt = json_response_format(settings.memory_extract_json_mode)
    try:
        raw = complete_memory_chat(messages, max_tokens=384, response_format=fmt)
    except LLMError as e:
        if fmt is None:
            logger.warning("抽取长期记忆失败: %s", e)
            return []
        logger.warning("JSON 模式抽取失败（网关可能不支持），回退 prompt 约束: %s", e)
        try:
            raw = complete_memory_chat(messages, max_tokens=384)
        except LLMError as e2:
            logger.warning("抽取长期记忆失败: %s", e2)
            return []
    items = _parse_memories(raw)
    if items:
        logger.info("长期记忆已抽取 count=%s", len(items))
    return items


def _write_extracted(user_id: str, item: ExtractedMemory) -> None:
    try:
        if item.kind == "profile":
            apply_profile_action(
                ProfileAction(op=item.op, facet=item.facet, content=item.content),
                user_id=user_id,
            )
        else:
            store.write_episode(item.content, user_id=user_id)
    except Exception:
        logger.exception("长期记忆写入失败 kind=%s", item.kind)


def _gate_blocks(turn: list[ChatMessage], summary: str) -> bool:
    """Jev 门控：True = 本轮不值得抽取，直接跳过。

    fail-open——门控不可用（未配 key / 调用失败）时返回 False 放行，
    行为退化为旧的「每轮都调小模型抽取」。阈值刻意偏低：
    门控漏判 = 静默丢记忆，比多花一次抽取调用严重。
    """
    state = _build_gate_state(turn, summary)
    if not state:
        return False
    probability = jev_noul(GATE_INSTRUCTIONS, state)
    if probability is None:
        logger.info("Jev 门控不可用（未配 key 或调用失败），放行抽取")
        return False
    threshold = settings.memory_gate_threshold
    passed = probability >= threshold
    logger.info(
        "Jev 门控判定 prob=%.2f threshold=%.2f → %s",
        probability,
        threshold,
        "放行，继续抽取" if passed else "拦截，跳过本轮",
    )
    return not passed


def _build_gate_state(turn: list[ChatMessage], summary: str) -> str:
    """门控输入只送「摘要 + 本轮对话」，已有画像/相近记忆不喂给 Jev——
    去重交给抽取阶段看完整上下文处理，门控只做语义门槛判断。"""
    parts: list[str] = []
    text = summary.strip()
    if text:
        parts.append(f"{LABEL_SESSION_SUMMARY}{text[:600]}")
    turn_text = _format_turn(turn)[:6000]
    if turn_text:
        parts.append(f"{LABEL_CURRENT_TURN}\n{turn_text}")
    return "\n\n".join(parts)


def _build_user_content(
    source: str,
    summary: str,
    existing_episodes: list[str] | None,
    existing_profile: list[str] | None,
) -> str:
    parts: list[str] = []
    text = summary.strip()
    if text:
        parts.append(f"{LABEL_SESSION_SUMMARY}\n{text}")
    if existing_profile:
        parts.append(LABEL_EXISTING_PROFILE + "\n".join(existing_profile))
    else:
        parts.append(LABEL_EXISTING_PROFILE_EMPTY)
    if existing_episodes:
        lines = "\n".join(f"- {item}" for item in existing_episodes)
        parts.append(f"{LABEL_EXISTING_EPISODES}\n{lines}")
    else:
        parts.append(LABEL_EXISTING_EPISODES_EMPTY)
    parts.append(f"{LABEL_CURRENT_TURN}\n{source}")
    return "\n\n".join(parts)


def _format_turn(turn: list[ChatMessage]) -> str:
    lines: list[str] = []
    for msg in turn:
        line = _line(msg)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _line(msg: ChatMessage) -> str:
    role = msg.get("role")
    content = (msg.get("content") or "").strip()
    if role == "user":
        return f"用户：{content}" if content else ""
    if role == "assistant":
        calls = msg.get("tool_calls") or []
        names = ", ".join(call.get("name") or "tool" for call in calls)
        body = content[:800]
        if names and body:
            return f"助手：{body}（调用工具 {names}）"
        if names:
            return f"助手：调用工具 {names}"
        return f"助手：{body}" if body else ""
    if role == "tool":
        name = msg.get("name") or "tool"
        snippet = content[:400]
        return f"工具[{name}]：{snippet}" if snippet else ""
    return ""


# 抽取模型常把本轮问答写成记忆。这些开头几乎一定是对话纪要，不是用户近况。
_RECAP_HEAD = re.compile(
    r"^(用户(询问|问了|问的是|讨论|要求|请求|查询|查了|了解到|分析|"
    r"正在分析|正在查询|确认了|提到|被[告知]|希望获取)"
    r"|助手(建议|回答|决定|通过|告知)"
    r"|讨论了|近期讨论)"
)
_COMPANY_MARK = re.compile(r"股份有限公司|有限责任公司")
_DOC_QA = ("kb_search", "kg_search", "招股", "年报披露", "助手")
_TOPIC_PREF = ("关注公司", "财务数据", "股东信息", "重大工程")


def _keep_extracted(item: ExtractedMemory) -> bool:
    """小模型常把招股书问答写成记忆，这里挡一层确定性噪声。"""
    text = item.content.strip()
    if item.kind == "episode":
        if _RECAP_HEAD.match(text):
            return False
        if any(mark in text for mark in _DOC_QA):
            return False
        return True
    if item.op == "delete":
        return True
    if _RECAP_HEAD.match(text):
        return False
    if item.facet in {"preference", "other", "constraint"}:
        if any(mark in text for mark in _TOPIC_PREF):
            return False
        if any(mark in text for mark in ("招股", "kb_search", "kg_search")):
            return False
    if item.facet == "occupation" and len(_COMPANY_MARK.findall(text)) >= 2:
        return False
    return True


def _parse_memories(raw: str) -> list[ExtractedMemory]:
    data = load_llm_json(raw)
    if data is None:
        logger.warning("抽取长期记忆 JSON 无法解析")
        return []
    if isinstance(data, dict):
        rows = data.get("memories")
    else:
        rows = data
    if not isinstance(rows, list):
        return []
    items: list[ExtractedMemory] = []
    for row in rows[:2]:
        item = _memory_from_row(row)
        if item is None:
            continue
        if not _keep_extracted(item):
            logger.info(
                "丢弃非跨会话记忆 kind=%s facet=%s text=%s",
                item.kind,
                item.facet,
                item.content[:80],
            )
            continue
        items.append(item)
    return _coalesce_memories(items)


def _coalesce_memories(items: list[ExtractedMemory]) -> list[ExtractedMemory]:
    profiles = [item for item in items if item.kind == "profile"]
    others = [item for item in items if item.kind != "profile"]
    return (
        coalesce_by_facet(
            profiles,
            facet_of=lambda item: item.facet,
            op_of=lambda item: item.op,
        )
        + others
    )


def _memory_from_row(row: object) -> ExtractedMemory | None:
    if not isinstance(row, dict):
        return None
    kind = row.get("kind")
    content = " ".join(str(row.get("content") or "").split()).strip()
    if kind == "episode":
        if len(content) < 4 or len(content) > 200:
            return None
        return ExtractedMemory(kind="episode", content=content)
    if kind != "profile":
        return None
    op = str(row.get("op") or "add").strip()
    if op not in ("add", "update", "delete"):
        return None
    facet = normalize_facet(row.get("facet"))
    if op == "delete":
        return ExtractedMemory(kind="profile", content=content, op=op, facet=facet)
    if len(content) < 2 or len(content) > 200:
        return None
    return ExtractedMemory(kind="profile", content=content, op=op, facet=facet)
