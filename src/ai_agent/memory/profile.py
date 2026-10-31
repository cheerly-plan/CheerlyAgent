from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.memory.db import MemoryItem, current_user_id
from ai_agent.memory.facets import (
    FACET_LABELS,
    coalesce_by_facet,
    format_profile_line,
    is_mutex_facet,
    normalize_facet,
)
from ai_agent.memory.gate import jev_noul_batch
from ai_agent.memory.parse import complete_memory_chat, load_llm_json
from ai_agent.memory.store import store
from ai_agent.prompts.memory import (
    DUP_INSTRUCTIONS,
    LABEL_EXISTING_PROFILE,
    LABEL_NEW_FACT,
    RESOLVE_SYSTEM,
)

logger = get_logger(__name__)

# Jev 语义去重阈值：新条目与某条已有条目「同一偏好」概率达到该值即跳过写入。
# 刻意偏高（宁可漏去重不可错杀新偏好），跑一段时间看日志再调
_JEV_DUP_THRESHOLD = 0.7

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
        if is_mutex_facet(cleaned.facet):
            # 互斥 facet 交给 update 覆盖语义（如职业补充完善），只告警不拦截
            logger.warning(
                "profile action looks redundant op=%s facet=%s text=%s",
                cleaned.op,
                cleaned.facet,
                cleaned.content,
            )
        else:
            # 累积 facet（preference/constraint/other）：与已有条目高度相似
            # 就不写，防止同义偏好无限堆积
            logger.info(
                "profile 条目与已有条目高度相似，跳过 facet=%s text=%s",
                cleaned.facet,
                cleaned.content,
            )
            return ProfileAction(op="skip", facet=facet, content=cleaned.content)
    return cleaned


def _jev_duplicate(content: str, facet: str, existing: list[MemoryItem]) -> bool:
    """Jev 语义去重：新条目是否与同 facet 已有条目表达同一件事。

    一次请求对每条已有条目并行发一个 Noul；任一命中阈值即视为重复。
    返回 False 含「Jev 不可用」——fail-open，放行写入。
    """
    candidates = [
        item.text
        for item in existing
        if (item.facet or "other") == facet and item.text and item.text != content
    ]
    if not candidates:
        return False
    state = (
        f"新条目：{content}\n\n已有条目：\n"
        + "\n".join(f"{i}. {text}" for i, text in enumerate(candidates))
    )
    answers = jev_noul_batch(
        {f"dup_{i}": DUP_INSTRUCTIONS for i in range(len(candidates))},
        state,
    )
    if answers is None:
        return False
    for i, text in enumerate(candidates):
        if answers[f"dup_{i}"] >= _JEV_DUP_THRESHOLD:
            logger.info(
                "Jev 判定条目重复 facet=%s new=%s old=%s prob=%.2f",
                facet,
                content,
                text,
                answers[f"dup_{i}"],
            )
            return True
    return False


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
        # 互斥 facet 删整行；累积 facet 只删 content 精确匹配的那条，
        # 防止否定一条偏好时把同 facet 的其他条目全清掉
        text = None if is_mutex_facet(facet) else (cleaned.content or None)
        store.delete_profile_facet(facet, user_id=uid, text=text)
        logger.info(
            "delete_profile user_id=%s facet=%s text=%s", uid, facet, text or "*"
        )
        return None
    # 累积 facet 写入前过 Jev 语义去重（模型转述的近义条目拦在这里）
    if not is_mutex_facet(facet) and _jev_duplicate(cleaned.content, facet, existing):
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
                    "content": f"{LABEL_EXISTING_PROFILE}\n{numbered}\n\n{LABEL_NEW_FACT}\n{text}",
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
    # 互斥 facet 同名冲突取最后一条；累积 facet（preference/constraint/other）
    # 的多条独立事实各自保留，不能合并
    mutex = coalesce_by_facet(
        [item for item in items if is_mutex_facet(item.facet)],
        facet_of=lambda item: item.facet,
        op_of=lambda item: item.op,
        skip=lambda item: item.op == "skip",
    )
    acc = [item for item in items if not is_mutex_facet(item.facet)]
    return mutex + acc


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
