"""赛道／单家结构梯队与比率并列句。"""

from __future__ import annotations

from ai_agent.report import registry
from ai_agent.report.extract import Evidence, IndicatorFact, citation_anchor
from ai_agent.report.industry import (
    lead_priority,
    notes_for_plan,
    profile_key_for_plan,
    structure_for_plan,
)
from ai_agent.report.parse import ReportPlan
from ai_agent.report.analyze_heuristics import structure_narrative_cues

DEFAULT_STRUCTURE_GLOSS = "此为已抽取数字的横向关系，不构成外部归因。"
DEFAULT_STRUCTURE_GLOSS_MORE = "同样只陈述本期数字结构，不引入无出处因果。"

def segment_structure_lines(
    plan: ReportPlan, facts: list[IndicatorFact]
) -> tuple[list[str], list[IndicatorFact]]:
    """用行业档案 structure 清单写梯队与结构关系（①级数据关系，非第三方因果）。"""
    from ai_agent.report.compare import parse_number
    from ai_agent.report.skills import get_skill

    indicators = structure_for_plan(plan)
    if not indicators:
        return [], []

    skill = get_skill(plan.skill_id) if plan.skill_id else None
    segment_of = skill.segment_of if skill else (lambda _name: "")
    # 行业专属的定性 gloss 来自 industries.yaml；未配置则用通用句。
    notes = notes_for_plan(plan)

    def _ranked(indicator: str) -> list[tuple[str, str, str, object]]:
        rows: list[tuple[str, str, str, object]] = []
        for fact in facts:
            if fact.name != indicator:
                continue
            parsed = parse_number(fact.value)
            if parsed is None:
                continue
            amount, _ = parsed
            cite = citation_anchor(fact.evidence.file_id)
            rows.append((fact.subject, fact.value, cite, amount))
        rows.sort(key=lambda item: item[3], reverse=True)
        return rows

    ranked = {name: _ranked(name) for name in indicators}
    if all(len(rows) < 2 for rows in ranked.values()):
        return [], []

    lines: list[str] = []
    derived: list[IndicatorFact] = []
    for name in indicators:
        rows = ranked[name]
        if not rows:
            continue
        ladder = " > ".join(
            f"{subject} {value}%{f' {cite}' if cite else ''}"
            for subject, value, cite, _ in rows
        )
        lines.append(f"{name}梯队（从高到低）：{ladder}。")

    # 按赛道聚合比率均值，只陈述高低关系，不写外部因果
    def _segment_avg(rows: list[tuple[str, str, str, object]]) -> list[tuple[str, object]]:
        buckets: dict[str, list[object]] = {}
        for name, _value, _cite, amount in rows:
            seg = segment_of(name) or "未分组"
            buckets.setdefault(seg, []).append(amount)
        out: list[tuple[str, object]] = []
        for seg, amounts in buckets.items():
            if not amounts:
                continue
            avg = sum(amounts) / len(amounts)  # type: ignore[arg-type]
            out.append((seg, avg))
        out.sort(key=lambda item: item[1], reverse=True)
        return out

    for index, name in enumerate(indicators):
        seg_rows = _segment_avg(ranked[name])
        if len(seg_rows) < 2:
            continue
        detail = " / ".join(f"{seg}≈{fmt_avg(avg)}%" for seg, avg in seg_rows)
        order = " > ".join(seg for seg, _ in seg_rows)
        gloss = notes.gloss_for(name) or (
            DEFAULT_STRUCTURE_GLOSS if index == 0 else DEFAULT_STRUCTURE_GLOSS_MORE
        )
        lines.append(f"按赛道看{name}均值排序为 {order}（{detail}）：{gloss}")
        derived.extend(
            segment_avg_fact(plan.period, name, seg, avg) for seg, avg in seg_rows
        )
    if not lines:
        return [], []
    lines.append("以上判断仅基于本期已抽取或结构化补齐的比率，不引入无出处的订单／景气推断。")
    return lines, derived


# 兼容旧测试名。
semiconductor_structure_lines = segment_structure_lines


def fmt_avg(amount: object) -> str:
    """赛道均值展示到一位小数，便于正文横比阅读。"""
    try:
        return f"{float(amount):.1f}"
    except (TypeError, ValueError):
        return str(amount)


def segment_avg_fact(
    period: str, indicator: str, segment: str, avg: object
) -> IndicatorFact:
    value = fmt_avg(avg)
    quote = f"{segment}赛道{indicator}均值≈{value}%（成员本期{indicator}算术平均）"
    return IndicatorFact(
        subject=f"{segment}赛道均值",
        period=period,
        name=f"{indicator}均值",
        value=value,
        unit="%",
        column="赛道均值推算",
        scope="derived",
        evidence=Evidence(
            file_id=f"segment-avg-{segment}-{indicator}",
            filename="derived.segment_average",
            page=None,
            page_end=None,
            section="经营解读",
            quote=quote,
        ),
    )


def company_fact_priority(plan: ReportPlan) -> tuple[str, ...]:
    """单家经营解读数字并列顺序：structure → lead → lead_generic（不写死毛利率）。"""
    structured = structure_for_plan(plan)
    if structured:
        return structured
    lead = lead_priority(profile_key_for_plan(plan))
    if lead:
        return lead
    generic = list(registry.ordering_list("lead_generic"))
    for name in sorted(registry.names_with("ladder")):
        if name not in generic:
            generic.append(name)
    return tuple(generic)


def pick_structure_ratio(
    plan: ReportPlan, facts: list[IndicatorFact]
) -> IndicatorFact | None:
    """结构并列句选用的比率：优先 structure／lead 里已有的本期事实。"""
    priority = company_fact_priority(plan)
    current = [fact for fact in facts if fact.period == plan.period]
    by_name = {fact.name: fact for fact in current}
    for name in priority:
        fact = by_name.get(name)
        if fact is None:
            continue
        if fact.unit == "%" or registry.flag(fact.name, "ladder"):
            return fact
    return next((fact for fact in current if fact.unit == "%"), None)


def company_structure_line(
    plan: ReportPlan,
    facts: list[IndicatorFact],
    *,
    has_narrative: bool,
    cited_text: str = "",
) -> str:
    """有经营原文共现线索时，用已授权比率做一句结构并列（非外部归因）。

    比率名来自行业档案 structure／lead，不再写死毛利率。
    """
    if not has_narrative:
        return ""
    # 结构句只点名正文里真实出现过的原文用语，避免「引用了量价齐升但原文没了」。
    labels = [
        label
        for label, token in structure_narrative_cues()
        if token in (cited_text or "")
    ]
    if not labels:
        return ""
    ratio = pick_structure_ratio(plan, facts)
    if ratio is None:
        return ""
    label_text = "／".join(labels)
    anchor = citation_anchor(ratio.evidence.file_id)
    source = ratio.evidence.filename or ""
    unit = ratio.unit or "%"
    # akshare 财务摘要列是报告期末累计口径，须写明，避免与单季营收混读。
    if source.startswith("akshare."):
        scope = "前三季度累计口径" if plan.period.endswith("Q3") else "结构化摘要累计口径"
        cite = f"（{scope} {anchor}）" if anchor else f"（{scope}）"
    else:
        page = f"第{ratio.evidence.page}页" if ratio.evidence.page is not None else ""
        cite = f"（{page} {anchor}）" if page or anchor else ""
    return (
        f"与原文「{label_text}」并列：本期{ratio.name}为 {ratio.value}{unit}"
        f"{cite}——只陈述已授权比率与原文用语的共现，不构成外部归因。"
    )
