"""赛道分组／组内对比／梯队表／公司间对比。"""

from __future__ import annotations

from ai_agent.kb.ingest.meta import name_in_text
from ai_agent.report import registry
from ai_agent.report.compare import SeriesCompare, parse_number
from ai_agent.report.extract import IndicatorFact
from ai_agent.report.industry import (
    INDUSTRY_SHARED,
    columns_for_segment,
    default_indicators,
    profile_key_for_plan,
    structure_for_plan,
    type_for_segment,
)
from ai_agent.report.parse import ReportPlan
from ai_agent.report.skills import get_skill
from ai_agent.report.compose_common import (
    cell_with_unit,
    cite,
    find_fact,
    is_segment_report,
    normalize_unit_label,
    scope_text,
)

def skill_segments(plan: ReportPlan) -> list[tuple[str, list[str]]]:
    skill = get_skill(plan.skill_id) if plan.skill_id else None
    if skill and skill.companies:
        groups = skill.companies_by_segment()
        if plan.subjects:
            wanted = plan.subjects
            filtered: list[tuple[str, list[str]]] = []
            for segment, names in groups:
                keep = [
                    name
                    for name in names
                    if any(
                        name == subject or name in subject or subject in name
                        for subject in wanted
                    )
                ]
                if keep:
                    filtered.append((segment, keep))
            return filtered or groups
        return groups
    if plan.subjects:
        return [("未分组", list(plan.subjects))]
    return []


def facts_for_subjects(
    current: list[IndicatorFact], subjects: list[str]
) -> list[IndicatorFact]:
    rows: list[IndicatorFact] = []
    for fact in current:
        if any(
            name_in_text(fact.subject, name)
            or name_in_text(name, fact.subject)
            or fact.subject == name
            for name in subjects
        ):
            rows.append(fact)
    return rows


def normalize_unit_label(unit: str) -> str:
    text = (unit or "").strip()
    if text in {"", "元", "人民币元"}:
        return "元"
    if text in {"千元", "人民币千元"}:
        return "千元"
    if text in {"万元", "人民币万元"}:
        return "万元"
    if text in {"百万元", "人民币百万元"}:
        return "百万元"
    if text in {"亿元", "人民币亿元"}:
        return "亿元"
    return text


def peer_table_block(
    title: str, rows: list[IndicatorFact], *, min_subjects: int = 1
) -> list[str]:
    """组内对比表：金额已在抽取层尽量归一为元；仍有单位例外时脚注说明。"""
    subjects = {row.subject for row in rows}
    if len(subjects) < min_subjects:
        return []
    units = {normalize_unit_label(row.unit) for row in rows if row.unit}
    lines = [title, "| 公司 | 数值 | 出处 |", "| --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row.subject} | {row.value}{row.unit} | {cite(row)} |")
    money = units & {"元", "千元", "万元", "亿元", "百万元"}
    if len(money) > 1:
        lines.append(
            f"注：本表金额单位未统一（{'／'.join(sorted(money))}），"
            "暂不混比；抽取层未能归一的项保留原单位。"
        )
    return lines


def segment_roster_lines(
    plan: ReportPlan, current: list[IndicatorFact]
) -> list[str]:
    groups = skill_segments(plan)
    if not groups:
        return ["没有可分组的代表公司。"]
    lines = [
        f"{plan.industry or ''}行业 {plan.period} 按赛道覆盖如下（只列名单与是否抽到本期财报，不写赛道景气度）。".strip()
    ]
    for segment, names in groups:
        bits: list[str] = []
        for name in names:
            ready = any(
                name_in_text(fact.subject, name) or name_in_text(name, fact.subject)
                for fact in current
            )
            bits.append(f"{name}（{'已抽到本期' if ready else '本期未抽到'}）")
        lines.append(f"**{segment}**：{'；'.join(bits)}。")
    return lines


def cell_with_unit(fact: IndicatorFact | None, *, indicator: str = "") -> str:
    if fact is None:
        return "—"
    unit = (fact.unit or "").strip()
    if unit == "%":
        return f"{fact.value}%"
    if unit:
        return f"{fact.value}{unit}"
    # 金额类缺单位时默认补「元」，避免同行带单位、本格裸数字。
    money = registry.names_with("money_cell_unit")
    if indicator in money or (not unit and _looks_like_money(fact.value)):
        return f"{fact.value}元"
    return str(fact.value)


def _looks_like_money(value: str) -> bool:
    text = (value or "").replace(",", "")
    return bool(text) and text.replace(".", "", 1).isdigit() and "," in (value or "")


def find_fact(
    facts: list[IndicatorFact], subject: str, indicator: str
) -> IndicatorFact | None:
    for fact in facts:
        if fact.name != indicator:
            continue
        if (
            fact.subject == subject
            or name_in_text(fact.subject, subject)
            or name_in_text(subject, fact.subject)
        ):
            return fact
    return None


def wide_segment_table(
    segment: str,
    names: list[str],
    facts: list[IndicatorFact],
    indicators: list[str],
) -> list[str]:
    """行=公司、列=指标的赛道宽表；金额格带原单位，缺数用 —。"""
    header = "| 公司 | " + " | ".join(indicators) + " |"
    sep = "| --- | " + " | ".join("---" for _ in indicators) + " |"
    lines = [header, sep]
    money_units: set[str] = set()
    converted_raw: set[str] = set()
    money_names = registry.names_with("money_wide_table")
    for company in names:
        cells = [company]
        for indicator in indicators:
            fact = find_fact(facts, company, indicator)
            cells.append(cell_with_unit(fact, indicator=indicator))
            if fact and indicator in money_names:
                money_units.add(normalize_unit_label(fact.unit or "元"))
                if getattr(fact, "unit_converted", False) and getattr(fact, "raw_unit", ""):
                    converted_raw.add(str(fact.raw_unit))
        lines.append("| " + " | ".join(cells) + " |")
    if len(names) < 2:
        lines.append("注：本组仅 1 家代表公司，不做组内横比，仅列示本期已抽到指标。")
    money = money_units & {"元", "千元", "万元", "亿元", "百万元"}
    if len(money) > 1:
        lines.append(
            f"注：金额列单位未统一（{'／'.join(sorted(money))}），"
            "抽取层未能归一；定稿前须折算为元后再比。"
        )
    elif converted_raw:
        lines.append(
            f"注：上表部分金额由{'、'.join(sorted(converted_raw))}折算为元，原值见溯源附件。"
        )
    # 出处脚注：每公司一行，避免宽表塞满出处列
    cites: list[str] = []
    for company in names:
        bits: list[str] = []
        for indicator in indicators:
            fact = find_fact(facts, company, indicator)
            if fact is None:
                continue
            bits.append(f"{indicator}←{cite(fact)}")
        if bits:
            cites.append(f"{company}：{'；'.join(bits)}")
    for line in cites:
        lines.append(f"出处：{line}。")
    return lines


def structure_ladder_table(
    plan: ReportPlan, current: list[IndicatorFact]
) -> list[str]:
    """全行业结构比率梯队总表；指标名来自行业档案 structure（否则 ladder 标记）。"""
    from ai_agent.report.compare import parse_number

    wanted = list(structure_for_plan(plan))
    if not wanted:
        # 未声明 structure 时，仍允许 indicators.yaml 里 ladder=true 的指标出表。
        seen: set[str] = set()
        for fact in current:
            if registry.flag(fact.name, "ladder") and fact.name not in seen:
                seen.add(fact.name)
                wanted.append(fact.name)
    for indicator in wanted:
        rows: list[tuple[object, str, IndicatorFact]] = []
        for fact in current:
            if fact.name != indicator:
                continue
            parsed = parse_number(fact.value)
            if parsed is None:
                continue
            rows.append((parsed[0], fact.subject, fact))
        if len(rows) < 2:
            continue
        rows.sort(key=lambda item: item[0], reverse=True)
        lines = [
            f"### 全行业{indicator}梯队",
            f"| 名次 | 公司 | {indicator} | 出处 |",
            "| --- | --- | --- | --- |",
        ]
        for index, (_amount, subject, fact) in enumerate(rows, start=1):
            lines.append(
                f"| {index} | {subject} | {fact.value}% | {cite(fact)} |"
            )
        return lines
    return []


SEGMENT_TABLE_CORE = registry.ordering_int("segment_table_core")
OCF = registry.ordering_str("ocf")


def segment_table_indicators(typed: list[str]) -> list[str]:
    """组内宽表列：主看项最多六列；模板含经营现金流则必保留（与封测表对齐）。"""
    core = list(typed[:SEGMENT_TABLE_CORE])
    if OCF in typed and OCF not in core:
        core.append(OCF)
    return core


def segment_peer_lines(
    plan: ReportPlan, current: list[IndicatorFact]
) -> list[str]:
    groups = skill_segments(plan)
    if not groups:
        return peer_lines(plan, current)
    # 组内对比按赛道类型路由指标；共享集作兜底，避免把 Fabless 指标硬套 Foundry。
    # 赛道语义按当前行业档案解析：同名赛道在不同行业可以指向不同模板。
    shared = plan.indicators or default_indicators(INDUSTRY_SHARED)
    segment_key = profile_key_for_plan(plan)
    lines: list[str] = []
    ladder = structure_ladder_table(plan, current)
    if ladder:
        lines.extend(ladder)
        lines.append("")
    for segment, names in groups:
        segment_facts = facts_for_subjects(current, names)
        lines.append(f"### {segment}")
        if not segment_facts:
            lines.append("本期未抽到该赛道代表公司指标。")
            lines.append("")
            continue
        # 赛道显式 columns 优先（系统集成不要「产能利用率」）；未声明才回退模板截列。
        override = columns_for_segment(segment, segment_key)
        if override:
            indicators = override
        else:
            typed = default_indicators(type_for_segment(segment, segment_key))
            # 赛道专属指标优先；抽不到的列在宽表里留空，缺口进「缺失」节。
            indicators = segment_table_indicators(typed or list(shared))
        lines.extend(wide_segment_table(segment, names, segment_facts, indicators))
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return lines or ["组内没有可对比的已抽取指标。"]


def chart_guide_lines(
    series: list[SeriesCompare],
    plan: ReportPlan | None = None,
    facts: list[IndicatorFact] | None = None,
) -> list[str]:
    lines = [
        "PDF 附图规则：赛道报告以「组内横比」柱状图为主（同赛道、同指标、公司并列）；"
        "单公司多期仅两期用柱状两点对比，至少四期同口径才画折线，避免伪趋势。"
        "柱状图仅出现在 PDF 交付件；本 Markdown 只列将生成的图题。",
    ]
    if plan is not None and is_segment_report(plan):
        from ai_agent.report.pdf import charts_for_report

        charts = charts_for_report(plan, facts or [], series)
        if charts:
            lines.append("本期 PDF 将附以下组内横比柱状图：")
            for index, chart in enumerate(charts, start=1):
                lines.append(f"{index}. {chart.title}")
        else:
            lines.append("本期无足够的同赛道横比数据时，不附组内柱状图。")
        return lines
    comparable = sum(1 for item in series if len(item.points) >= 2)
    if comparable:
        lines.append(f"本期另有 {comparable} 组多期序列可作附录对照。")
    else:
        lines.append("本期无可用同口径多期序列时，不附伪趋势折线。")
    return lines


def peer_lines(plan: ReportPlan, current: list[IndicatorFact]) -> list[str]:
    grouped: dict[tuple[str, str], list[IndicatorFact]] = {}
    for fact in current:
        if plan.indicators and fact.name not in plan.indicators:
            continue
        grouped.setdefault((fact.name, fact.scope), []).append(fact)
    lines: list[str] = []
    for (name, scope), rows in grouped.items():
        subjects = {row.subject for row in rows}
        if len(subjects) < 2:
            continue
        block = peer_table_block(f"{name}（{scope_text(scope)}）", rows, min_subjects=2)
        if lines and block:
            lines.append("")
        lines.extend(block)
    return lines or ["同一指标没有两家以上的公司，不做公司间对比。"]


