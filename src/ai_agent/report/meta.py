"""口径结构化元数据：从财报**模板结构**推定，不靠正文里找「合并」二字。

四类口径：
1. 范围（合并／母公司）——「主要会计数据和财务指标」等主表默认合并；
2. 期间（单季／累计／时点）——由列名映射，已在 basis.py；
3. 单位——金额归一为元，保留原单位；
4. 基期缺失——真缺数据，不与前三类混写。

``inferred=True`` 的项在报告里写「按季报模板推定」，不写「范围未标明」。
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from ai_agent.kb.ingest.tables import SCOPE_CONSOLIDATED, SCOPE_PARENT

# 这些章节／表位在 A 股季报模板里固定是合并口径（除非表题显式写母公司）。
_CONSOLIDATED_SECTIONS = (
    "主要会计数据和财务指标",
    "主要财务数据",
    "合并资产负债表",
    "合并利润表",
    "合并现金流量表",
    "合并所有者权益变动表",
    "季度财务报表",
)
# 列名已给出期间；出现在主表即推定合并。
_MAIN_TABLE_COLUMNS = (
    "本报告期",
    "年初至报告期末",
    "本报告期末",
    "上年度末",
)

_UNIT_SCALE = {
    "元": Decimal(1),
    "人民币元": Decimal(1),
    "千元": Decimal(1_000),
    "人民币千元": Decimal(1_000),
    "万元": Decimal(10_000),
    "人民币万元": Decimal(10_000),
    "百万元": Decimal(1_000_000),
    "人民币百万元": Decimal(1_000_000),
    "亿元": Decimal(100_000_000),
    "人民币亿元": Decimal(100_000_000),
}

_NUMBER = re.compile(r"^-?\d[\d,]*\.?\d*$")


def infer_scope(
    *,
    scope: str = "",
    section: str = "",
    column: str = "",
    quote: str = "",
) -> tuple[str, bool]:
    """返回 (scope, inferred)。已有明确口径不覆盖；否则按模板推定。"""
    existing = (scope or "").strip()
    if existing in {SCOPE_CONSOLIDATED, SCOPE_PARENT}:
        return existing, False
    blob = f"{section}\n{column}\n{quote}"
    if re.search(r"母公司|个别报表|单体报表", blob):
        return SCOPE_PARENT, True
    if re.search(r"合并", blob):
        return SCOPE_CONSOLIDATED, True
    section_text = section or ""
    if any(token in section_text for token in _CONSOLIDATED_SECTIONS):
        return SCOPE_CONSOLIDATED, True
    # 主要财务数据表常见列：即使 section 被切碎，列名在也按合并推定。
    col = (column or "").strip()
    if any(col == name or name in col for name in _MAIN_TABLE_COLUMNS):
        return SCOPE_CONSOLIDATED, True
    return "", False


def normalize_amount(
    value: str, unit: str
) -> tuple[str, str, str, bool]:
    """金额归一为元。

    返回 (展示值, 展示单位, 原始单位, 是否折算)。
    比率／空单位／无法解析时原样返回，converted=False。
    """
    raw_unit = (unit or "").strip()
    text = (value or "").strip().replace(",", "").replace("，", "")
    # 半年报摘要用空格做千分位（「356 903」）。不去掉空格，Decimal 解析失败，
    # 这一期就不折算，累计链里混进「元」和「百万元」两种单位，环比直接算不出来。
    text = re.sub(r"(?<=\d)[ \t\u00a0](?=\d)", "", text)
    if not text or raw_unit in {"", "%", "％", "百分点", "次"}:
        return value, raw_unit, raw_unit, False
    scale = _UNIT_SCALE.get(raw_unit)
    if scale is None or scale == 1:
        # 已是元或未知单位：未知单位不擅自折算
        if raw_unit in {"元", "人民币元"}:
            return value, "元", raw_unit, False
        return value, raw_unit, raw_unit, False
    if not _NUMBER.match(text):
        return value, raw_unit, raw_unit, False
    try:
        amount = Decimal(text) * scale
    except (InvalidOperation, ValueError):
        return value, raw_unit, raw_unit, False
    # 整数去小数，否则保留两位（与原文金额风格一致）
    if amount == amount.to_integral_value():
        display = f"{int(amount):,}"
    else:
        display = f"{amount:,.2f}"
    return display, "元", raw_unit, True


def enrich_quote_for_conversion(
    quote: str, *, raw_value: str, raw_unit: str, value: str, unit: str
) -> str:
    """折算后的展示值必须能在证据里核对：把采用值写进 quote。"""
    base = (quote or "").strip()
    note = f"原值={raw_value}{raw_unit} 采用={value}{unit}（由{raw_unit}折算）"
    if note in base:
        return base
    if not base:
        return note
    return f"{base}；{note}"


def scope_label(scope: str, *, inferred: bool = False) -> str:
    """正文／表格用的范围文案。"""
    if scope == SCOPE_CONSOLIDATED:
        return "合并（按季报模板推定）" if inferred else "合并"
    if scope == SCOPE_PARENT:
        return "母公司（按季报模板推定）" if inferred else "母公司"
    return "—"
