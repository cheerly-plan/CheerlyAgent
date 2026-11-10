"""赛道／单家结构梯队与比率并列句。"""

from __future__ import annotations

from decimal import Decimal

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

# 表内利润质量三角：只并列已抽取金额关系，不推断收入确认／回款原因。
QUALITY_PROFIT = "扣非净利润"
QUALITY_OCF = "经营活动产生的现金流量净额"
QUALITY_CONTRACT = "合同负债"
QUALITY_CROSS_MAX = 8
_MONEY_SCALE = {
    "元": Decimal(1),
    "万元": Decimal(10_000),
    "百万元": Decimal(1_000_000),
    "亿元": Decimal(100_000_000),
}


def segment_structure_lines(
    plan: ReportPlan, facts: list[IndicatorFact]
) -> tuple[list[str], list[IndicatorFact]]:
    """用行业档案 structure 清单写梯队与结构关系（①级数据关系，非第三方因果）。

    若本期已抽到扣非／经营现金流（合同负债可选），追加表内利润质量交叉句。
    """
    from ai_agent.report.compare import parse_number
    from ai_agent.report.skills import get_skill

    lines: list[str] = []
    derived: list[IndicatorFact] = []
    indicators = structure_for_plan(plan)
    if indicators:
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
        if not all(len(rows) < 2 for rows in ranked.values()):
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
            def _segment_avg(
                rows: list[tuple[str, str, str, object]]
            ) -> list[tuple[str, object]]:
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
                detail = " / ".join(
                    f"{seg}≈{fmt_avg(avg)}%" for seg, avg in seg_rows
                )
                order = " > ".join(seg for seg, _ in seg_rows)
                gloss = notes.gloss_for(name) or (
                    DEFAULT_STRUCTURE_GLOSS
                    if index == 0
                    else DEFAULT_STRUCTURE_GLOSS_MORE
                )
                lines.append(
                    f"按赛道看{name}均值排序为 {order}（{detail}）：{gloss}"
                )
                derived.extend(
                    segment_avg_fact(plan.period, name, seg, avg)
                    for seg, avg in seg_rows
                )

    quality = quality_cross_lines(facts)
    if quality:
        lines.extend(quality)
    if not lines:
        return [], []
    lines.append(
        "以上判断仅基于本期已抽取或结构化补齐的指标，不引入无出处的订单／景气推断。"
    )
    return lines, derived


def quality_cross_lines(facts: list[IndicatorFact]) -> list[str]:
    """扣非 × 经营现金流 × 合同负债交叉：只写表内金额关系，不做外部归因。"""
    from ai_agent.report.compare import parse_number

    by_subject: dict[str, dict[str, IndicatorFact]] = {}
    for fact in facts:
        if fact.name not in {QUALITY_PROFIT, QUALITY_OCF, QUALITY_CONTRACT}:
            continue
        if parse_number(fact.value) is None:
            continue
        by_subject.setdefault(fact.subject, {})[fact.name] = fact

    scored: list[tuple[Decimal, str]] = []
    for subject, bundle in by_subject.items():
        profit = bundle.get(QUALITY_PROFIT)
        ocf = bundle.get(QUALITY_OCF)
        if profit is None or ocf is None:
            continue
        scored.append((_quality_divergence(profit, ocf), subject))
    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    has_contract = any(
        QUALITY_CONTRACT in by_subject[subject] for _, subject in scored
    )
    head = (
        "利润—现金流—合同负债交叉（表内数字关系，不推断收入确认或回款）："
        if has_contract
        else "利润—经营现金流交叉（表内数字关系，不推断收入确认或回款）："
    )
    lines = [head]
    for _, subject in scored[:QUALITY_CROSS_MAX]:
        line = _quality_company_line(subject, by_subject[subject])
        if line:
            lines.append(line)
    return lines if len(lines) > 1 else []


def _quality_divergence(profit: IndicatorFact, ocf: IndicatorFact) -> Decimal:
    """优先展示正负号分歧；同号则按相对落差排序，避免大盘股绝对额盖过异常。"""
    profit_yuan = _fact_yuan(profit)
    ocf_yuan = _fact_yuan(ocf)
    if profit_yuan is None or ocf_yuan is None:
        return Decimal(0)
    gap = abs(profit_yuan - ocf_yuan)
    scale = max(abs(profit_yuan), abs(ocf_yuan), Decimal(1))
    relative = gap / scale
    # 双负／双正以外的符号组合，或双负并存，都抬到同号相对落差之上。
    if (profit_yuan > 0) != (ocf_yuan > 0):
        return Decimal(200) + relative
    if profit_yuan < 0 and ocf_yuan < 0:
        return Decimal(100) + relative
    return relative


def _fact_yuan(fact: IndicatorFact) -> Decimal | None:
    from ai_agent.report.compare import parse_number

    parsed = parse_number(fact.value)
    if parsed is None:
        return None
    amount, _ = parsed
    unit = (fact.unit or "").strip()
    if unit in {"", "元", "人民币元"}:
        return amount
    scale = _MONEY_SCALE.get(unit)
    if scale is None:
        return None
    return amount * scale


def _quality_company_line(subject: str, bundle: dict[str, IndicatorFact]) -> str:
    profit = bundle[QUALITY_PROFIT]
    ocf = bundle[QUALITY_OCF]
    contract = bundle.get(QUALITY_CONTRACT)
    bits = [
        f"{QUALITY_PROFIT} {_fact_display(profit)}",
        f"{QUALITY_OCF} {_fact_display(ocf)}",
    ]
    if contract is not None:
        bits.append(f"{QUALITY_CONTRACT} {_fact_display(contract)}（时点余额）")
    relation = _quality_relation(profit, ocf)
    body = "、".join(bits)
    if relation:
        return f"{subject}：{body}——{relation}。"
    return f"{subject}：{body}。"


def _quality_relation(profit: IndicatorFact, ocf: IndicatorFact) -> str:
    profit_yuan = _fact_yuan(profit)
    ocf_yuan = _fact_yuan(ocf)
    if profit_yuan is None or ocf_yuan is None:
        return ""
    profit_pos = profit_yuan > 0
    ocf_pos = ocf_yuan > 0
    if profit_pos and ocf_pos and ocf_yuan > profit_yuan:
        return "经营现金流高于扣非净利润"
    if profit_pos and ocf_pos and ocf_yuan < profit_yuan:
        return "经营现金流低于扣非净利润"
    if profit_pos and not ocf_pos:
        return "扣非净利润为正、经营活动产生的现金流量净额为负"
    if not profit_pos and not ocf_pos:
        return "扣非净利润与经营活动产生的现金流量净额均为负"
    if not profit_pos and ocf_pos:
        return "扣非净利润为负、经营活动产生的现金流量净额为正"
    return "扣非净利润与经营活动产生的现金流量净额同号且量级接近"


def _fact_display(fact: IndicatorFact) -> str:
    unit = (fact.unit or "").strip()
    if unit == "%":
        text = f"{fact.value}%"
    elif unit:
        text = f"{fact.value}{unit}"
    else:
        text = str(fact.value)
    cite = citation_anchor(fact.evidence.file_id)
    return f"{text}{f' {cite}' if cite else ''}"


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
