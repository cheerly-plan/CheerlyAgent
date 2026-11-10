"""同比、环比和多期增减。只做四则运算，不调用模型。

比率类指标（毛利率，或数值带 %）的变化用百分点，不用「除以基期」再报一个增速。
金额单位能换算的先换成元再减；换算不了或同一期有两个数，这一对不算。

**期间可比性是硬约束**：累计 1-9 月的值不能和累计 1-3 月的值相减后报增长率
（差额只是期数不同，不是业绩变化）。相减前先比口径与覆盖月数，不一致就
不产出任何增减额或幅度，只留一句「不可比」的缺口。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

from ai_agent.kb.ingest.meta import period_rank
from ai_agent.report.basis import POINT, STOCK

_PERIOD = re.compile(r"^((?:19|20)\d{2})(Q([1-4])|H1)?$")
_SCALE = {
    "元": Decimal(1),
    "万元": Decimal(10_000),
    "百万元": Decimal(1_000_000),
    "亿元": Decimal(100_000_000),
}
_RATIO_NAME = re.compile(r"(率|比率)$")
# 表格里的「单位：百万元人民币（另有说明者除外）」和「单位：百万元人民币
# （百分比除外）」是同一个单位，括号说明不同不该让跨期换算失败。
_UNIT_NOISE = re.compile(r"[（(].*$|人民币|币种")

# A 股累计报告的期末月份。没有独立二季报，Q3 的单季环比只能靠
# Q1（1-3月）、H1（1-6月）、Q3（1-9月）三份累计值相减还原单季。
_CUMULATIVE_CHAIN: dict[str, list[str]] = {
    "H1": ["Q1"],
    "Q3": ["Q1", "H1"],
    "Q4": ["Q1", "H1", "Q3"],
}
# 累计报告覆盖的月份数。Q3 是 1-9 月（9 个月），Q1 是 1-3 月（3 个月）。
_CUMULATIVE_MONTHS = {"Q1": 3, "H1": 6, "Q3": 9, "Q4": 12}
NOT_COMPARABLE_NOTE = "相邻报告期口径或覆盖期数不同、或期间不连续，不做跨期增减计算"
_NOT_COMPARABLE_NOTE = NOT_COMPARABLE_NOTE
# 期末时点余额（贷款/存款/资产总计）在本期季报里只披露「比上年末」变动。
# 自行拿去年同期的时点值算同比，会和原文那个维度并列出现、读者分不清，
# 也容易复现「不同比较基准混在一起」的老问题。这一维度交给视图层呈现。
STOCK_NOTE = "期末时点余额只呈现原文较上年末变动，不做跨期增减计算"


def cumulative_base_periods(period: str) -> list[str]:
    """本期是累计口径时，推导单季环比需要哪些更早的累计报告。

    A 股没有独立二季报，所以 ``2026Q3`` 返回 ``['2026Q1', '2026H1']``：
    Q2 单季 = H1 − Q1，Q3 单季 = Q3累计 − H1，两者相比才是真环比。
    """
    match = _PERIOD.fullmatch(period or "")
    if not match:
        return []
    year = int(match.group(1))
    token = match.group(2)  # "Q1" / "Q3" / "H1" / None
    if token in _CUMULATIVE_CHAIN:
        return [f"{year}{item}" for item in _CUMULATIVE_CHAIN[token]]
    return []


def _unit_key(unit: str) -> str:
    """把表格里的单位说明收敛到 _SCALE 的键，比较与换算都用它。"""
    text = _UNIT_NOISE.sub("", (unit or "").strip()).strip()
    for key in sorted(_SCALE, key=len, reverse=True):
        if text == key or text.startswith(key):
            return key
    return text


@dataclass
class Measure:
    subject: str
    name: str
    period: str
    value: str
    unit: str = ""
    scope: str = ""
    # 季报原文披露的「比上年同期增减(%)」。只在没有跨期绝对值可用时兜底。
    reported_yoy: str = ""
    # 数值取自累计列（年初至报告期末 / 1-6月）。只有三期都为累计，
    # 才允许用相邻累计值相减还原单季，避免把单季数当累计去减。
    cumulative: bool = False
    # 出处标识（file_id 优先）。推导值必须凑齐整条链才允许写进正文。
    source_key: str = ""
    # 口径："" 未标明（沿用旧行为） / cumulative 累计 / single 单季 / point 时点比率。
    # 只有两端口径与覆盖月数都一致，才允许相减并报增减幅度。
    basis: str = ""
    coverage_months: int = 0


@dataclass
class Change:
    base_period: str
    base_value: str
    delta: str
    rate: str
    note: str = ""
    # "" = 由跨期绝对值算出来；"reported" = 季报原文直接披露的增减幅度；
    # "derived" = 由多份累计报告相减还原单季后再算出来的环比。
    source: str = ""
    # derived 必须带完整的出处链（各累计报告的 file_id / 文件名），
    # validate 只对链完整的推导值放行，防止"凭空多出一个数"。
    evidence: tuple[str, ...] = ()


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
    # 期间 → 口径指纹（"cumulative:9" / "single:3" / "point:0"）。
    # 图表只画同口径的点，避免把 1-3月/1-6月/1-9月累计连成一条「增长曲线」。
    bases: dict[str, str] = field(default_factory=dict)


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


def _period_token(period: str) -> str:
    match = _PERIOD.fullmatch(period or "")
    return match.group(2) if match else ""


def _basis(row: Measure) -> tuple[str, int] | None:
    """这一期的口径指纹。返回 None 表示调用方没标口径（沿用历史行为）。"""
    if row.basis:
        months = row.coverage_months
        # 时点比率与期末余额的覆盖期数都是 0；只有期间类才按季报推断。
        if not months and row.basis not in (POINT, STOCK):
            months = _CUMULATIVE_MONTHS.get(_period_token(row.period), 3)
        return row.basis, months
    if row.cumulative:
        return "cumulative", _CUMULATIVE_MONTHS.get(_period_token(row.period), 0)
    return None


def _is_stock(row: Measure) -> bool:
    basis = _basis(row)
    return bool(basis and basis[0] == STOCK)


def _comparable(current: Measure, base: Measure) -> bool:
    """两期能不能相减。口径与覆盖月数必须一致；一端标明一端没标也拒绝。

    都未标明时返回 True 是兼容旧的纯计算调用（工具/单测直接给几期单季值），
    真实流水线里 extract 会明确标出累计口径，不会走到这个分支。
    """
    left, right = _basis(current), _basis(base)
    if left is None or right is None:
        return left is None and right is None
    return left == right


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
    for period in ordered:
        basis = _basis(by_period[period])
        if basis:
            series.bases[period] = f"{basis[0]}:{basis[1]}"
    if any(_is_stock(by_period[period]) for period in ordered):
        series.gaps.append(STOCK_NOTE)
        return series
    ratio = _is_ratio(name, by_period[ordered[-1]])
    for period in ordered:
        current = by_period[period]
        base_period = yoy_period(period)
        if base_period and base_period in by_period:
            base = by_period[base_period]
            if not _comparable(current, base):
                series.gaps.append(f"{period} 与 {base_period} 期间口径不同，不算同比")
            else:
                change = _change(current, base, ratio=ratio)
                if change is None:
                    series.gaps.append(f"{period} 与 {base_period} 单位不一致，不算同比")
                else:
                    series.yoy[period] = change
        prior = qoq_period(period)
        if prior and prior in by_period:
            base = by_period[prior]
            if not _comparable(current, base):
                series.gaps.append(f"{period} 与 {prior} 期间口径不同，不算环比")
            else:
                change = _change(current, base, ratio=ratio)
                if change is None:
                    series.gaps.append(f"{period} 与 {prior} 单位不一致，不算环比")
                else:
                    series.qoq[period] = change
        # 库里没有上一年的季报时，跨期绝对值算不出同比；但季报表格本身
        # 往往已经披露了「比上年同期增减(%)」。直接用原文值，不瞎算、也不
        # 把「没有基期」误报成「没有同比」。跨期计算优先于原文披露值。
        if period not in series.yoy and current.reported_yoy:
            series.yoy[period] = Change(
                base_period="上年同期",
                base_value="",
                delta="",
                rate=current.reported_yoy,
                note="未用跨期绝对值重算",
                source="reported",
            )
    for earlier, later in zip(ordered, ordered[1:]):
        left, right = by_period[earlier], by_period[later]
        # 趋势只写「相邻且同口径」的一对。跨年不连续的两期（2025Q3→2026Q1）
        # 即使同为时点比率也不写：那不是趋势，是两段独立事实。
        if not _comparable(left, right) or qoq_period(later) != earlier:
            # 不产出 delta/rate，也不把期间号写进缺口（避免正文多出未核对数字）。
            if _NOT_COMPARABLE_NOTE not in series.gaps:
                series.gaps.append(_NOT_COMPARABLE_NOTE)
            continue
        change = _change(right, left, ratio=ratio)
        if change is None:
            series.gaps.append(f"{earlier} → {later} 单位不一致，不算趋势")
            continue
        note = change.note
        if by_period[later].cumulative and by_period[earlier].cumulative:
            # 两个累计值相减是「区间增量」，不是单季环比，别让读者误读
            note = "累计口径差额为区间增量，不是单季环比" + (f"；{note}" if note else "")
        series.trend.append(
            TrendStep(
                earlier=earlier,
                later=later,
                delta=change.delta,
                rate=change.rate,
                adjacent=qoq_period(later) == earlier,
                note=note,
            )
        )
    _derive_single_quarter_qoq(series, by_period, ratio)
    return series


def _derive_single_quarter_qoq(
    series: SeriesCompare,
    by_period: dict[str, Measure],
    ratio: bool,
) -> None:
    """用相邻累计值相减还原单季，再算真正的季度环比。

    A 股没有独立二季报（qoq_period("2026Q3") 指向的 "2026Q2" 永远不存在）。
    有了 Q1（1-3月累计）、H1（1-6月累计）、Q3（1-9月累计）才能算：
    Q2 单季 = H1 − Q1，Q3 单季 = Q3累计 − H1，两者相比才是标准环比。

    以下条件缺一就写进 gaps，绝不硬算：三期都在库里且都标记为累计口径、
    单位归一化后一致、数值可读且非比率、每份报告都有出处标识。

    只给最新一期（研报本期）推导：同比基期等历史期间缺累计报告是正常的，
    不该给它们刷出一堆与本期无关的缺口。
    """
    if ratio or not by_period:
        return
    focus = max(by_period, key=period_rank)
    for period in sorted(by_period, key=period_rank):
        if period != focus:
            continue
        bases = cumulative_base_periods(period)
        if len(bases) < 2:
            continue
        if period in series.qoq:
            # 有基于真实跨期绝对值的环比时以它为准，不用推导值覆盖
            continue
        chain_periods = [*bases, period]
        rows = [by_period.get(item) for item in chain_periods]
        if any(row is None for row in rows):
            # 缺累计报告由 parse 层按实际库藏写进 plan.gaps（它知道有哪些文件）。
            # 纯计算层不重复报缺口，否则只传两期做同比的调用也会刷出无关噪音。
            return
        present = [row for row in rows if row is not None]
        if not all(row.cumulative for row in present):
            series.gaps.append(
                f"{period} 的累计报告不齐（{'、'.join(bases)} 需同为累计口径），不推导单季环比"
            )
            continue
        sources = tuple((row.source_key or "").strip() for row in present)
        if not all(sources):
            series.gaps.append(f"{period} 缺累计报告出处，不推导单季环比")
            continue
        units = {_unit_key(row.unit) for row in present}
        if len(units) != 1 or not units.issubset(_SCALE):
            series.gaps.append(f"{period} 累计报告单位不一致或无法换算，不推导单季环比")
            continue
        unit = next(iter(units))
        values: list[Decimal] = []
        for row in present:
            parsed = parse_number(row.value)
            if parsed is None or parsed[1]:
                break
            values.append(parsed[0])
        if len(values) != len(present):
            series.gaps.append(f"{period} 累计值读不出来，不推导单季环比")
            continue
        prior_single = values[-2] - values[-3]
        current_single = values[-1] - values[-2]
        if prior_single <= 0 or current_single <= 0:
            series.gaps.append(f"{period} 推导出的单季值非正，不写环比")
            continue
        delta = current_single - prior_single
        rate = (delta / prior_single) * Decimal(100)
        series.qoq[period] = Change(
            base_period=qoq_period(period) or "",
            base_value=f"{_format_number(prior_single)}{unit}",
            delta=f"{_format_number(delta)}{unit}",
            rate=f"{_format_number(rate)}%",
            note="单季环比由相邻累计值相减还原，非原文披露，出处见溯源附件",
            source="derived",
            evidence=sources,
        )


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
    left = _unit_key(current_unit)
    right = _unit_key(base_unit)
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
    if change.source == "reported":
        return f"  {label} {change.rate}（季报原文披露，{change.note}）"
    if change.source == "derived":
        return (
            f"  {label} {change.base_period}（累计值推导的基期）{change.base_value}："
            f"增减 {change.delta}，幅度 {change.rate}（{change.note}）"
        )
    rate = f"，幅度 {change.rate}" if change.rate else ""
    note = f"，{change.note}" if change.note else ""
    return (
        f"  {label} {change.base_period} {change.base_value}："
        f"增减 {change.delta}{rate}{note}"
    )
