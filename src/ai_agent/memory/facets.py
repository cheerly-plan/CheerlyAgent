from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from ai_agent.core.logging import get_logger
from ai_agent.storage.db import MemoryItem
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


# 候选 facet 的确定性推断。候选来自 episode（用户正在做的事/近况），
# 原先一律落 ``other``，晋升进画像后丢掉「偏好/约束/职业」语义，排序和
# 注入都按「其他」降权。这里不调模型，只用保守的线索词把「本质是画像事实」
# 的 episode 归到正确 facet；判断不了就老实留 ``other``。
_FACET_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "constraint",
        (
            "必须",
            "不要",
            "禁止",
            "只能",
            "不能",
            "以后都",
            "每次都",
            "一律",
            "务必",
            "再也别",
        ),
    ),
    ("name", ("我叫", "名叫", "名字是", "姓名是", "称呼我", "叫我")),
    (
        "language",
        (
            "用中文",
            "用英文",
            "用英语",
            "说中文",
            "说英文",
            "简体中文",
            "繁体中文",
            "语言偏好",
        ),
    ),
    (
        "locale",
        ("住在", "搬到", "居住在", "居住地", "所在城市", "老家在", "定居"),
    ),
    (
        "occupation",
        (
            "职业",
            "工作",
            "岗位",
            "任职",
            "转行",
            "工程师",
            "程序员",
            "设计师",
            "研究员",
            "分析师",
            "经理",
            "律师",
            "医生",
            "老师",
            "做前端",
            "做后端",
        ),
    ),
    (
        "preference",
        ("喜欢", "偏好", "更喜欢", "不喜欢", "习惯用", "常用", "倾向于", "爱用"),
    ),
)


def infer_facet(text: str, default: str = "other") -> str:
    """从一段用户近况里推断画像 facet（0~1 个最可能的分类）。

    只认显式线索词，宁可返回 ``default`` 也不猜。用于 L2 episode 晋升到
    L3 候选时保留语义；不参与画像写入本身的 facet 决策。
    """
    normalized = " ".join((text or "").split())
    if not normalized:
        return default
    for facet, hints in _FACET_HINTS:
        if any(hint in normalized for hint in hints):
            return facet
    return default
