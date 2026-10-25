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

FACET_WRITE_RULES = """
facet 行为：
- name / language / locale：互斥，新事实直接替换旧内容，不要拼接历史；旧值不写回当前画像
- locale：只写当前地区/居住地。已有「北京」、新事实「搬到上海」→ locale 写成「上海」，不要写成「曾住北京，现住上海」
- occupation：只写当前实际从事的职业，可附必要的过往历史。必须区分「已完成转行」和「打算转行」：
  * 已完成（已经转行 / 现在做 X / 刚换到 X 岗位）→ 更新现状。已有「画家」、新事实「已经转行做前端」→ occupation 写成「曾是画家，现为前端开发工程师」
  * 只是意向（打算 / 计划 / 准备 / 想 / 考虑 转行做 X）→ 这不是事实，occupation 保持当前职业不变，严禁写成「现为 X」；意向可写入 other，如「计划转行做 Agent 开发」
- preference / constraint：可同时成立多条独立偏好/约束，每条单独输出一个 add，不要合并成一段、也不要覆盖已有条目；只有用户明确否定某条（「我不再…」「其实我不喜欢…」）才 delete 那一条，delete 时 content 写被否定条目的原文
- other：其他稳定事实；姓名、职业、地区、语言不要写进 other
转行（已发生的）、工种、职业身份一律 occupation。「想学后端 / 准备考证」这类学习或意向不是 occupation。
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
