from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.facets import (
    FACET_WRITE_RULES,
    coalesce_by_facet,
    format_profile_line,
    normalize_facet,
)
from ai_agent.memory.parse import complete_memory_chat, load_llm_json
from ai_agent.memory.profile import ProfileAction, apply_profile_action
from ai_agent.memory.store import store

logger = get_logger(__name__)

EXTRACT_SYSTEM = f"""
你是用户长期记忆抽取器。根据本轮对话，判断是否需要写入跨会话记忆。
只输出 JSON，不要解释。格式：
{{"memories":[{{"kind":"episode","content":"..."}},{{"kind":"profile","op":"add"|"update"|"delete","facet":"name"|"occupation"|"locale"|"language"|"preference"|"constraint"|"other","content":"..."}}]}}
没有可记内容时输出 {{"memories":[]}}

kind:
- profile: 长期成立的姓名、职业、身份、语言、偏好、约束
- episode: 当前阶段的任务快照

episode 的 content 必须是可检索的完整近况（约 40～120 字），尽量包含：正在做什么、技术栈或约束、目标、本轮已确认的进展。不要写成「最近在学 X」这种标签。

profile 必须带 op 和 facet。同一侧面只输出一条：
- add: 新的侧面
- update: 同一侧面事实变化或更完整，content 写成该侧面的完整当前描述
- delete: 用户明确否定旧事实
{FACET_WRITE_RULES}
若已有 other 把姓名/职业混在一起，要拆开写入正确 facet，并 delete other。
已有相同事实、只是换种说法、闲聊、助手单方面建议：不要输出该条。

规则：
1. 只记用户明确说的，或双方已经确认的技术选择
2. 身份和近况拆开
3. 最多 4 条
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
    for row in rows[:4]:
        item = _memory_from_row(row)
        if item is None:
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
