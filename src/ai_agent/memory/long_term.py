from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.facets import (
    FACET_WRITE_RULES,
    coalesce_by_facet,
    format_profile_line,
    normalize_facet,
)
from ai_agent.memory.gate import jev_noul
from ai_agent.memory.parse import complete_memory_chat, load_llm_json
from ai_agent.memory.profile import ProfileAction, apply_profile_action
from ai_agent.memory.store import store

logger = get_logger(__name__)

EXTRACT_SYSTEM = f"""
你是用户长期记忆抽取器。只记「关于用户本人、且下一场对话还用得上」的事实。
只输出 JSON，不要解释。格式：
{{"memories":[{{"kind":"episode","content":"..."}},{{"kind":"profile","op":"add"|"update"|"delete","facet":"name"|"occupation"|"locale"|"language"|"preference"|"constraint"|"other","content":"..."}}]}}
没有可记内容时必须输出 {{"memories":[]}}。拿不准就空数组。

kind:
- profile: 用户用第一人称说清的姓名、职业、居住地、语言、长期偏好、约束
- episode: 用户自己正在做、会跨会话延续的任务（项目、已确认的技术方案、明确要求记住的进展）

严禁写入：
- 本轮问答摘要（「用户询问了…」「讨论了某公司…」「助手建议…」）
- 查知识库/招股书/年报/公告得到的公司事实、股东、财务数字
- 把用户问过的公司写成用户的职业或身份（问董事长 ≠ 用户是董事长）
- 一次性查询：天气、股价、某家公司概况、某页表格
- 助手建议、工具调用过程
- 已有相同事实、只是换种说法、闲聊

episode 的 content 约 40～120 字，写用户本人近况，不要写成对话纪要。
profile 必须带 op 和 facet。name / occupation / locale / language 同一侧面只输出一条：
- add: 新的侧面
- update: 同一侧面事实变化或更完整，content 写成该侧面的完整当前描述
- delete: 用户明确否定旧事实
preference / constraint 是累积型：每条独立事实单独输出一个 add，可多条并存，不要 update 合并已有条目；否定某一条时 delete 并在 content 写该条原文。
{FACET_WRITE_RULES}
若已有 other 把姓名/职业混在一起，要拆开写入正确 facet，并 delete other。

规则：
1. 只记用户明确陈述的自身事实，或双方已确认、属于用户本人的技术选择
2. 身份和近况拆开
3. 最多 2 条
"""


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
    try:
        raw = complete_memory_chat(
            [
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            max_tokens=384,
        )
    except LLMError as e:
        logger.warning("抽取长期记忆失败: %s", e)
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


GATE_INSTRUCTIONS = (
    "这段对话是否包含值得写入跨会话长期记忆的信息？"
    "值得记忆：用户用第一人称明确陈述的姓名、职业、身份、居住地、语言、"
    "长期偏好或约束；用户自己正在做且会跨会话延续的任务或技术方案。"
    "不值得记忆：查文档或问某家公司、天气、新闻、一次性事实查询、"
    "本轮问答摘要、助手建议、把检索结果当成用户身份。"
)


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
        parts.append(f"会话摘要：{text[:600]}")
    turn_text = _format_turn(turn)[:6000]
    if turn_text:
        parts.append(f"本轮对话：\n{turn_text}")
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
        parts.append(f"会话摘要：\n{text}")
    if existing_profile:
        parts.append("已有用户画像：\n" + "\n".join(existing_profile))
    else:
        parts.append("已有用户画像：无")
    if existing_episodes:
        lines = "\n".join(f"- {item}" for item in existing_episodes)
        parts.append(f"已有相近情景记忆：\n{lines}")
    else:
        parts.append("已有相近情景记忆：无")
    parts.append(f"本轮对话：\n{source}")
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
