"""缺失说明与缺口归并。"""

from __future__ import annotations

import re

from ai_agent.kb.ingest.meta import name_in_text
from ai_agent.report.compare import (
    NOT_COMPARABLE_NOTE,
    STOCK_NOTE,
    SeriesCompare,
)
from ai_agent.report.extract import IndicatorFact
from ai_agent.report.industry import industry_of, is_excluded
from ai_agent.report.parse import ReportPlan
from ai_agent.report.compose_common import period_label, sole_subject

def gap_lines(plan: ReportPlan, current: list[IndicatorFact], gaps: list[str]) -> list[str]:
    # 范围／期间能推定的在阅读说明与指标表已交代；这里只列例外与真缺口。
    lines: list[str] = []
    parents = [fact.name for fact in current if fact.scope == "parent"]
    if parents:
        lines.append(f"母公司口径（例外）：{'、'.join(sorted(set(parents)))}。")
    inferred = [fact for fact in current if getattr(fact, "scope_inferred", False)]
    if inferred and not parents:
        lines.append("范围：本期主表指标按季报模板推定为合并（详见阅读说明）。")
    bases = sorted({period_label(fact) for fact in current if fact.basis})
    if bases:
        lines.append(f"期间口径：{'、'.join(bases)}（同比只用同口径同期）。")
    # 真缺口：缺指标／缺基期文件。把「口径不齐」类与「待入库」分开列。
    data_gaps = [gap for gap in gaps if is_data_gap(gap)]
    other_gaps = [gap for gap in gaps if not is_data_gap(gap)]
    if data_gaps:
        lines.append("数据缺口（待入库或未抽到）：")
        lines.extend(data_gaps)
    if other_gaps:
        lines.append("其他说明：")
        lines.extend(other_gaps)
    return lines or ["没有缺失项。"]


def is_data_gap(gap: str) -> bool:
    text = gap or ""
    return any(
        token in text
        for token in (
            "没有抽到",
            "没有找到",
            "未入库",
            "待",
            "A 股无独立",
            "累计报告不齐",
            "知识库",
        )
    )


def has_indicator(
    facts: list[IndicatorFact], subject: str, name: str
) -> bool:
    """需求简称与事实主体全称可互相包含即视为已抽到。"""
    for fact in facts:
        if fact.name != name:
            continue
        if fact.subject == subject:
            return True
        if name_in_text(fact.subject, subject) or name_in_text(subject, fact.subject):
            return True
    return False


_CUMULATIVE_BASE_GAP_RE = re.compile(
    r"^A 股无独立二季报；缺 (?P<base>.+) 的累计报告，(?P<focus>.+) 单季环比无法推导$"
)


def collect_gaps(
    plan: ReportPlan, facts: list[IndicatorFact], series: list[SeriesCompare]
) -> list[str]:
    gaps = coalesce_cumulative_base_gaps(list(plan.gaps))
    period_facts = [fact for fact in facts if fact.period == plan.period]
    # 单公司报告缺口句省略主语，全称由标题承担
    subjects = {fact.subject for fact in facts} | {item.subject for item in series}
    single = sole_subject(plan, subjects)
    company_type = getattr(plan, "company_type", "") or industry_of(plan.subjects)
    # 同指标多公司都缺：概括一行，避免「没有抽到」刷屏。
    missing_by_indicator: dict[str, list[str]] = {}
    for subject in plan.subjects:
        for name in plan.indicators:
            # 类型排除项不报假缺失（银行毛利率、Fabless 净息差等）。
            if is_excluded(company_type, name):
                continue
            if not has_indicator(period_facts, subject, name):
                missing_by_indicator.setdefault(name, []).append(subject)
    for name, cos in missing_by_indicator.items():
        gaps.append(format_missing_indicator_gap(cos, plan.period, name, single=bool(single)))
    # 「不可比」／「时点存量」：逐指标同一句 → 合并计数。
    # 其它 series 缺口：同一原因跨公司／跨指标 → 概括一条 + 例外单列。
    repeated: dict[str, int] = {}
    by_reason: dict[str, list[tuple[str, str]]] = {}
    for item in series:
        for gap in item.gaps:
            if gap in {NOT_COMPARABLE_NOTE, STOCK_NOTE}:
                repeated[gap] = repeated.get(gap, 0) + 1
                continue
            by_reason.setdefault(gap, []).append((item.subject, item.name))
    for note, count in repeated.items():
        gaps.append(f"{count} 项指标：{note}")
    for reason, hits in by_reason.items():
        gaps.append(format_series_gap_cluster(hits, reason, single=bool(single)))
    return gaps


def coalesce_cumulative_base_gaps(gaps: list[str]) -> list[str]:
    """缺 Q1／H1 两条「累计基期」合并成一句，避免同一规则写两遍。"""
    bases: list[str] = []
    focus = ""
    out: list[str] = []
    for gap in gaps:
        matched = _CUMULATIVE_BASE_GAP_RE.match(gap or "")
        if matched is None:
            out.append(gap)
            continue
        base = matched.group("base")
        if base not in bases:
            bases.append(base)
        focus = matched.group("focus")
    if not bases:
        return out
    out.append(
        f"A 股无独立二季报；缺 {'、'.join(bases)} 的累计报告，"
        f"{focus} 单季环比无法推导"
    )
    return out


def unique_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def format_missing_indicator_gap(
    companies: list[str], period: str, name: str, *, single: bool
) -> str:
    cos = unique_preserve(companies)
    if single or len(cos) <= 1:
        prefix = "" if single else f"{cos[0]} "
        return f"{prefix}没有抽到 {period} {name}"
    return f"{len(cos)} 家公司（{'、'.join(cos)}）没有抽到 {period} {name}"


def format_series_gap_cluster(
    hits: list[tuple[str, str]], reason: str, *, single: bool
) -> str:
    """同一原因只写一次：多家／多指标概括；单条例外保持「公司+指标：原因」。"""
    subjects = unique_preserve([subject for subject, _ in hits])
    names = unique_preserve([name for _, name in hits])
    if len(hits) == 1:
        subject, name = hits[0]
        prefix = "" if single else subject
        return f"{prefix}{name}：{reason}"
    name_part = "、".join(names)
    if single or len(subjects) == 1:
        prefix = "" if single else subjects[0]
        if len(names) == 1:
            return f"{prefix}{names[0]}：{reason}"
        return f"{prefix}{len(names)} 项指标（{name_part}）：{reason}"
    return (
        f"{len(subjects)} 家公司（{'、'.join(subjects)}）"
        f"{len(names)} 项指标（{name_part}）：{reason}"
    )


