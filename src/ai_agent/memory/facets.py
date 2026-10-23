from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from ai_agent.core.logging import get_logger
from ai_agent.memory.db import MemoryItem

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

FACET_WRITE_RULES = """
facet 行为：
- name / language / locale：互斥，新事实直接替换旧内容，不要拼接历史；旧值不写回当前画像
- locale：只写当前地区/居住地。已有「北京」、新事实「搬到上海」→ locale 写成「上海」，不要写成「曾住北京，现住上海」
- occupation：可保留过往转行信息，写成现状 + 必要历史。已有「画家」、新事实「转行前端」→ occupation 写成「曾是画家，现为前端开发工程师」
- preference / constraint：长期稳定描述，新内容写成该侧面的完整当前描述并覆盖旧描述
- other：其他稳定事实；姓名、职业、地区、语言不要写进 other
转行、工种、职业身份一律 occupation。「想学后端 / 准备考证」这类学习或意向不是 occupation。
已有 occupation 已覆盖该转行/工种时 skip，不要再写 other。
本轮如果只改了姓名，不要把已有职业再写一遍。
""".strip()


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
    return f"用户画像（稳定偏好/约束）：\n{lines}"


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
