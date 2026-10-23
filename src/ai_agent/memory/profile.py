from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.memory.facets import (
    FACET_LABELS,
    FACET_WRITE_RULES,
    coalesce_by_facet,
    format_profile_line,
    normalize_facet,
)
from ai_agent.memory.parse import complete_memory_chat, load_llm_json
from ai_agent.memory.db import MemoryItem, current_user_id
from ai_agent.memory.store import store

logger = get_logger(__name__)

_OCCUPATION_JOB_HINTS = (
    "工程师",
    "修理工",
    "汽修",
    "水电",
    "前端",
    "后端",
    "画家",
    "开发",
)

RESOLVE_SYSTEM = f"""
你是用户画像写入器。根据已有画像和新事实，决定如何更新。
只输出 JSON，不要解释。格式：
{{"actions":[{{"op":"add"|"update"|"delete"|"skip","facet":"name"|"occupation"|"locale"|"language"|"preference"|"constraint"|"other","content":"..."}}]}}

facet:
- name: 姓名
- occupation: 职业或长期身份
- locale: 地区或居住地
- language: 语言
- preference: 长期偏好
- constraint: 长期约束
- other: 其他稳定事实

{FACET_WRITE_RULES}

规则：
1. 一条输入可拆成多个 actions：姓名用 name，职业用 occupation
2. 已有 other 把姓名/职业混在一起时，要拆开写入正确 facet，并 delete other
3. 已有相同事实或只是换种说法：skip
4. 用户明确否定旧事实：delete
5. 不要把当前任务写成画像
"""


@dataclass(frozen=True)
class ProfileAction:
    op: Literal["add", "update", "delete", "skip"]
    facet: str = "other"
    content: str = ""


def _profile_norm(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", " \t\r\n。；;，,、.（）()（）"))


def is_redundant_profile(content: str, facet: str, existing: list[MemoryItem]) -> bool:
    new_norm = _profile_norm(content)
    if not new_norm:
        return True
    for item in existing:
        if (item.facet or "other") != facet:
            continue
        old_norm = _profile_norm(item.text)
        if not old_norm:
            continue
        if new_norm == old_norm or new_norm in old_norm:
            return True
        if old_norm in new_norm and len(new_norm) > len(old_norm) + 4:
            return False
        new_jobs = {hint for hint in _OCCUPATION_JOB_HINTS if hint in new_norm}
        old_jobs = {hint for hint in _OCCUPATION_JOB_HINTS if hint in old_norm}
        if new_jobs and new_jobs <= old_jobs:
            return True
        overlap = len(set(new_norm) & set(old_norm)) / len(set(new_norm))
        if overlap >= 0.75 and len(new_norm) <= len(old_norm) + 8:
            return True
    return False


def sanitize_profile_action(
    action: ProfileAction, existing: list[MemoryItem]
) -> ProfileAction:
    facet = normalize_facet(action.facet)
    cleaned = ProfileAction(op=action.op, facet=facet, content=action.content)
    if cleaned.op in ("add", "update") and is_redundant_profile(
        cleaned.content, cleaned.facet, existing
    ):
        logger.warning(
            "profile action looks redundant op=%s facet=%s text=%s",
            cleaned.op,
            cleaned.facet,
            cleaned.content,
        )
    return cleaned


def apply_profile_action(
    action: ProfileAction, user_id: str | None = None
) -> MemoryItem | None:
    uid = user_id or current_user_id()
    existing = store.list_profile(uid)
    cleaned = sanitize_profile_action(action, existing)
    if cleaned.op == "skip":
        return None
    facet = normalize_facet(cleaned.facet)
    if cleaned.op == "delete":
        store.delete_profile_facet(facet, user_id=uid)
        logger.info("delete_profile user_id=%s facet=%s", uid, facet)
        return None
    return store.write_profile(cleaned.content, user_id=uid, facet=facet)


def apply_profile_text(content: str, user_id: str | None = None) -> str:
    uid = user_id or current_user_id()
    existing = store.list_profile(uid)
    actions = resolve_profile_write(content, existing)
    messages: list[str] = []
    for action in actions:
        if action.op == "skip":
            continue
        if action.op == "delete":
            apply_profile_action(action, user_id=uid)
            messages.append(
                f"已删除用户画像: {FACET_LABELS.get(action.facet, action.facet)}"
            )
            continue
        item = apply_profile_action(action, user_id=uid)
        if item is None:
            continue
        messages.append(f"已写入用户画像: {item.text}")
    return "；".join(messages) if messages else "用户画像无需更新"


def resolve_profile_write(
    content: str, existing: list[MemoryItem]
) -> list[ProfileAction]:
    text = content.strip()
    if not text:
        return [ProfileAction(op="skip")]
    numbered = "\n".join(format_profile_line(item) for item in existing) or "无"
    try:
        raw = complete_memory_chat(
            [
                {"role": "system", "content": RESOLVE_SYSTEM},
                {
                    "role": "user",
                    "content": f"已有用户画像：\n{numbered}\n\n新事实：\n{text}",
                },
            ],
            max_tokens=256,
        )
    except LLMError as e:
        logger.warning("解析用户画像写入失败，回退为追加: %s", e)
        return [ProfileAction(op="add", facet="other", content=text)]
    actions = _parse_actions(raw, text)
    if actions is None:
        return [ProfileAction(op="add", facet="other", content=text)]
    return actions or [ProfileAction(op="skip")]


def _parse_actions(raw: str, fallback: str) -> list[ProfileAction] | None:
    data = load_llm_json(raw)
    if data is None:
        return None
    rows: list[object]
    if isinstance(data, dict) and isinstance(data.get("actions"), list):
        rows = data["actions"]
    elif isinstance(data, dict):
        rows = [data]
    elif isinstance(data, list):
        rows = data
    else:
        return None
    items: list[ProfileAction] = []
    for row in rows[:4]:
        action = _action_from_row(row, fallback)
        if action is not None:
            items.append(action)
    return coalesce_by_facet(
        items,
        facet_of=lambda item: item.facet,
        op_of=lambda item: item.op,
        skip=lambda item: item.op == "skip",
    )


def _action_from_row(row: object, fallback: str) -> ProfileAction | None:
    if not isinstance(row, dict):
        return None
    op = str(row.get("op") or "").strip()
    if op not in ("add", "update", "delete", "skip"):
        return None
    facet = normalize_facet(row.get("facet"))
    content = " ".join(str(row.get("content") or fallback).split()).strip()
    if op in ("add", "update") and (len(content) < 2 or len(content) > 200):
        return None
    return ProfileAction(op=op, facet=facet, content=content)
