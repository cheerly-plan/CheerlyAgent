"""同比、环比和多期增减。只做四则运算，不调用模型。

比率类指标（毛利率，或数值带 %）的变化用百分点，不用「除以基期」再报一个增速。
金额单位能换算的先换成元再减；换算不了或同一期有两个数，这一对不算。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

from ai_agent.kb.meta import period_rank

_PERIOD = re.compile(r"^((?:19|20)\d{2})(Q([1-4])|H1)?$")
_SCALE = {
    "元": Decimal(1),
    "万元": Decimal(10_000),
    "百万元": Decimal(1_000_000),
    "亿元": Decimal(100_000_000),
}
_RATIO_NAME = re.compile(r"(率|比率)$")


@dataclass
class Measure:
    subject: str
    name: str
    period: str
    value: str
    unit: str = ""
    scope: str = ""


@dataclass
class Change:
    base_period: str
    base_value: str
    delta: str
    rate: str
    note: str = ""


@dataclass
class TrendStep:
    earlier: str
    later: str
    delta: str
    rate: str
    adjacent: bool
    note: str = ""


@dataclass
class SeriesCompare:
    subject: str
    name: str
    scope: str
    unit: str
    points: list[tuple[str, str]] = field(default_factory=list)
    yoy: dict[str, Change] = field(default_factory=dict)
    qoq: dict[str, Change] = field(default_factory=dict)
    trend: list[TrendStep] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


def compare_measures(rows: list[Measure]) -> list[SeriesCompare]:
    """按公司、指标、口径分组。同一期出现两个不同数字就整期跳过。"""
    groups: dict[tuple[str, str, str], list[Measure]] = {}
    for row in rows:
        key = (row.subject.strip(), row.name.strip(), (row.scope or "").strip())
        if not key[0] or not key[1] or not row.period.strip():
            continue
        groups.setdefault(key, []).append(row)
    return [_compare_group(key, items) for key, items in groups.items()]


def parse_number(value: str) -> tuple[Decimal, bool] | None:
    """读出数字。返回 (数值, 是否百分数)。括号表示负数。读不懂返回 None。"""
    text = (value or "").strip().replace(",", "").replace("，", "").replace(" ", "")
    text = text.replace("％", "%").replace("－", "-")
    if not text or text in {"—", "-", "不适用", "--"}:
        return None
    negative = False
    if text[0] in "（(" and text[-1] in "）)":
        negative = True
        text = text[1:-1]
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return None
    number = Decimal(text)
    if negative:
        number = -number
    return number, percent


def yoy_period(period: str) -> str | None:
    match = _PERIOD.fullmatch(period or "")
    if not match:
        return None
    return f"{int(match.group(1)) - 1}{match.group(2) or ''}"


def qoq_period(period: str) -> str | None:
    """上一季。一季报的上一期是去年四季报。半年报和年报没有环比基期。"""
    match = _PERIOD.fullmatch(period or "")
    if not match or not match.group(3):
        return None
    year = int(match.group(1))
    quarter = int(match.group(3))
    if quarter == 1:
        return f"{year - 1}Q4"
    return f"{year}Q{quarter - 1}"


def _compare_group(
    key: tuple[str, str, str], items: list[Measure]
) -> SeriesCompare:
    subject, name, scope = key
    series = SeriesCompare(subject=subject, name=name, scope=scope, unit="")
    by_period: dict[str, Measure] = {}
    blocked: set[str] = set()
    for item in items:
        period = item.period.strip()
        if period in blocked:
            continue
        previous = by_period.get(period)
        if previous is not None and (
            previous.value != item.value or (previous.unit or "") != (item.unit or "")
        ):
            series.gaps.append(f"{period} 有两个不同数字，这一期不参与计算")
            by_period.pop(period, None)
            blocked.add(period)
            continue
        if period not in by_period:
            by_period[period] = item
    ordered = sorted(by_period, key=period_rank)
    if not ordered:
        return series
    series.unit = by_period[ordered[-1]].unit
    series.points = [(period, by_period[period].value) for period in ordered]
    ratio = _is_ratio(name, by_period[ordered[-1]])
    for period in ordered:
        current = by_period[period]
        base_period = yoy_period(period)
        if base_period and base_period in by_period:
            change = _change(current, by_period[base_period], ratio=ratio)
            if change is None:
                series.gaps.append(f"{period} 与 {base_period} 单位不一致，不算同比")
            else:
                series.yoy[period] = change
        prior = qoq_period(period)
        if prior and prior in by_period:
            change = _change(current, by_period[prior], ratio=ratio)
            if change is None:
                series.gaps.append(f"{period} 与 {prior} 单位不一致，不算环比")
            else:
                series.qoq[period] = change
    for earlier, later in zip(ordered, ordered[1:]):
        change = _change(by_period[later], by_period[earlier], ratio=ratio)
        if change is None:
            series.gaps.append(f"{earlier} → {later} 单位不一致，不算趋势")
            continue
        series.trend.append(
            TrendStep(
                earlier=earlier,
                later=later,
                delta=change.delta,
                rate=change.rate,
                adjacent=qoq_period(later) == earlier,
                note=change.note,
            )
        )
    return series


def _is_ratio(name: str, row: Measure) -> bool:
    if _RATIO_NAME.search(name):
        return True
    parsed = parse_number(row.value)
    if parsed and parsed[1]:
        return True
    return (row.unit or "").strip() in {"%", "％", "百分点"}


def _change(current: Measure, base: Measure, *, ratio: bool) -> Change | None:
    left = parse_number(current.value)
    right = parse_number(base.value)
    if left is None or right is None:
        return None
    current_number, current_percent = left
    base_number, base_percent = right
    if ratio or current_percent or base_percent:
        if current_percent != base_percent:
            return None
        delta = current_number - base_number
        return Change(
            base_period=base.period,
            base_value=base.value,
            delta=f"{_format_number(delta)}个百分点",
            rate="",
            note="",
        )
    scaled = _to_same_unit(current_number, current.unit, base_number, base.unit)
    if scaled is None:
        return None
    current_scaled, base_scaled, delta_unit = scaled
    delta = current_scaled - base_scaled
    rate = ""
    note = ""
    if base_scaled == 0:
        note = "基期为 0，不算增幅"
    else:
        percent = (delta / base_scaled) * Decimal(100)
        rate = f"{_format_number(percent)}%"
        if base_scaled < 0:
            note = "基期为负"
    return Change(
        base_period=base.period,
        base_value=base.value,
        delta=f"{_format_number(delta)}{delta_unit}",
        rate=rate,
        note=note,
    )


def _to_same_unit(
    current: Decimal, current_unit: str, base: Decimal, base_unit: str
) -> tuple[Decimal, Decimal, str] | None:
    left = (current_unit or "").strip()
    right = (base_unit or "").strip()
    if left == right:
        return current, base, left
    if left not in _SCALE or right not in _SCALE:
        return None
    current_yuan = current * _SCALE[left]
    base_yuan = base * _SCALE[right]
    # 差额用本期单位表达，避免把「万元」减完写成一长串「元」
    return current_yuan / _SCALE[left], base_yuan / _SCALE[left], left


def _format_number(value: Decimal) -> str:
    quantum = Decimal("0.01") if value != value.to_integral() else Decimal("1")
    rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    if quantum == Decimal("1"):
        return f"{int(rounded):,}"
    return f"{rounded:,.2f}"


def format_compare(series_list: list[SeriesCompare]) -> str:
    """给后续撰写步骤看的纯文本。没有算出来的对比不写数字。"""
    if not series_list:
        return "没有可计算的指标。"
    blocks: list[str] = []
    for series in series_list:
        title = f"{series.subject} {series.name}"
        if series.scope:
            title = f"{title}（{series.scope}）"
        lines = [title]
        for period, value in series.points:
            unit = series.unit
            lines.append(f"{period} {value}{unit}")
            yoy = series.yoy.get(period)
            if yoy:
                lines.append(_change_line("同比", yoy))
            qoq = series.qoq.get(period)
            if qoq:
                lines.append(_change_line("环比", qoq))
        if series.trend:
            lines.append("趋势")
            for step in series.trend:
                kind = "环比" if step.adjacent else "跨期"
                extra = f"，{step.note}" if step.note else ""
                rate = f"，幅度 {step.rate}" if step.rate else ""
                lines.append(
                    f"  {step.earlier} → {step.later}（{kind}）增减 {step.delta}{rate}{extra}"
                )
        lines.extend(series.gaps)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _change_line(label: str, change: Change) -> str:
    rate = f"，幅度 {change.rate}" if change.rate else ""
    note = f"，{change.note}" if change.note else ""
    return (
        f"  {label} {change.base_period} {change.base_value}："
        f"增减 {change.delta}{rate}{note}"
    )
