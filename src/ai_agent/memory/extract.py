from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import complete_chat
from ai_agent.llm.types import ChatMessage

logger = get_logger(__name__)

EXTRACT_SYSTEM = """
你是用户长期记忆抽取器。根据本轮对话，判断是否需要写入跨会话记忆。
只输出 JSON，不要解释。格式：
{"memories":[{"kind":"episode"|"profile","content":"..."}]}
没有可记内容时输出 {"memories":[]}

kind:
- profile: 长期成立的姓名、职业、身份、语言、偏好、约束
- episode: 当前阶段的任务快照

episode 的 content 必须是可检索的完整近况（约 40～120 字），尽量包含：正在做什么、技术栈或约束、目标、本轮已确认的进展。不要写成「最近在学 X」这种标签。

规则：
1. 只记用户明确说的，或双方已经确认的技术选择；不要把助手单方面建议记成用户决定
2. 闲聊、一次性查资料、与已有记忆重复且无新事实时，不要写
3. 一条记忆只写一件事，身份和近况拆开
4. 最多 2 条
"""


@dataclass(frozen=True)
class ExtractedMemory:
    kind: Literal["profile", "episode"]
    content: str


def extract_turn_memories(
    turn: list[ChatMessage],
    summary: str = "",
    existing: list[str] | None = None,
) -> list[ExtractedMemory]:
    source = _format_turn(turn)
    if not source.strip():
        return []
    if not any(msg.get("role") == "assistant" for msg in turn):
        return []
    user_content = _build_user_content(source, summary, existing)
    model = settings.llm_for_memory()
    try:
        raw = complete_chat(
            [
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": user_content},
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
        logger.warning("抽取长期记忆失败: %s", e)
        return []
    items = _parse_memories(raw)
    if items:
        logger.info("长期记忆已抽取 count=%s", len(items))
    return items


def _build_user_content(
    source: str, summary: str, existing: list[str] | None
) -> str:
    parts: list[str] = []
    text = summary.strip()
    if text:
        parts.append(f"会话摘要：\n{text}")
    if existing:
        lines = "\n".join(f"- {item}" for item in existing)
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
    data = _load_json(raw)
    if data is None:
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
        items.append(item)
    return items


def _memory_from_row(row: object) -> ExtractedMemory | None:
    if not isinstance(row, dict):
        return None
    kind = row.get("kind")
    content = " ".join(str(row.get("content") or "").split()).strip()
    if kind not in ("profile", "episode"):
        return None
    if len(content) < 4 or len(content) > 200:
        return None
    return ExtractedMemory(kind=kind, content=content)


def _load_json(raw: str) -> object | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("抽取长期记忆 JSON 无法解析")
        return None
