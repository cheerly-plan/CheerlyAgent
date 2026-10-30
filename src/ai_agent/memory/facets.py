from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from ai_agent.core.logging import get_logger
from ai_agent.memory.db import MemoryItem
from ai_agent.prompts.memory import PROFILE_BLOCK_HEADER

logger = get_logger(__name__)

T = TypeVar("T")

FACETS = (
    "name",
    "occupation",
    "locale",
    "language",
    "preference",
    "constraint",
    "other",
)

# 互斥型 facet：天然单值（姓名、当前职业、居住地、语言），新事实覆盖旧值，
# DB 层对 (user_id, facet) 建部分唯一索引兜底
MUTEX_FACETS = ("name", "language", "locale", "occupation")

# 累积型 facet（preference/constraint/other）：可以同时成立多条独立事实，
# 新事实追加为新行，只有用户明确否定某条时才删除那一行


def is_mutex_facet(facet: str) -> bool:
    return facet in MUTEX_FACETS
FACET_LABELS = {
    "name": "姓名",
    "occupation": "职业",
    "locale": "地区",
    "language": "语言",
    "preference": "偏好",
    "constraint": "约束",
    "other": "其他",
}
_FACET_ALIASES = {
    "name": "name",
    "姓名": "name",
    "occupation": "occupation",
    "职业": "occupation",
    "身份": "occupation",
    "locale": "locale",
    "地区": "locale",
    "居住地": "locale",
    "城市": "locale",
    "language": "language",
    "语言": "language",
    "preference": "preference",
    "偏好": "preference",
    "喜好": "preference",
    "constraint": "constraint",
    "约束": "constraint",
    "限制": "constraint",
    "other": "other",
    "其他": "other",
}

# FACET_WRITE_RULES 已迁到 ai_agent.prompts.memory —— 它是写给模型的指令，
# 不是 facet 的元数据，放在这里会让「抽取器」和「写入器」各自 import 后
# 用不同的方式内插。现在两处统一从 prompts.memory 取渲染好的版本。


def normalize_facet(value: object) -> str:
    key = str(value or "").strip()
    return _FACET_ALIASES.get(key) or _FACET_ALIASES.get(key.lower()) or "other"


def format_profile_line(item: MemoryItem) -> str:
    label = FACET_LABELS.get(item.facet or "other", "其他")
    return f"- {label}：{item.text}"


def format_profile_block(items: list[MemoryItem]) -> str:
    if not items:
        return ""
    lines = "\n".join(format_profile_line(item) for item in items)
    return f"{PROFILE_BLOCK_HEADER}\n{lines}"


def coalesce_by_facet(
    items: Iterable[T],
    *,
    facet_of: Callable[[T], str],
    op_of: Callable[[T], str],
    skip: Callable[[T], bool] | None = None,
) -> list[T]:
    last: dict[str, T] = {}
    order: list[str] = []
    for item in items:
        if skip is not None and skip(item):
            continue
        facet = facet_of(item)
        prev = last.get(facet)
        if prev is None:
            order.append(facet)
        elif op_of(prev) != op_of(item):
            logger.warning(
                "profile actions conflict facet=%s first_op=%s last_op=%s",
                facet,
                op_of(prev),
                op_of(item),
            )
        last[facet] = item
    return [last[facet] for facet in order]
