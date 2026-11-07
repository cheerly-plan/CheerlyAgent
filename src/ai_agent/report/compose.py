"""按大纲写成研报。正文里的数字只来自抽取结果和计算工具。

不调用模型，因此不会补一个大纲里有、但表格里没抽到的数。

章节渲染按职责拆到 compose_*.py；本模块负责编排、结论句与 renderer 注册。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_agent.report import registry
from ai_agent.report import sections as report_sections
from ai_agent.report.basis import POINT, STOCK, disclosure_note
from ai_agent.report.compare import (
    Change,
    SeriesCompare,
    compare_measures,
    parse_number,
)
from ai_agent.report.extract import IndicatorFact, ensure_amount_units
from ai_agent.report.industry import (
    lead_hint_profiles,
    lead_priority,
    notes_for_plan,
    profile_key_for_plan,
)
from ai_agent.report.parse import OutlineItem, ReportPlan
from ai_agent.report.phrases import ANALYSIS_FALLBACK
from ai_agent.report.features import Features, compute_features
from ai_agent.report.rules import (
    equity_share_applicable,
    equity_share_exemption_line,
    is_equity_row,
    render_yoy_rate_bit,
    render_yoy_sentence,
)
from ai_agent.report.supplemental import (
    SupplementalResult,
    format_money_delta,
    signed_delta,
)
from ai_agent.report import compose_common as _common
from ai_agent.report import compose_format as _format
from ai_agent.report import compose_gaps as _gaps_mod
from ai_agent.report import compose_segment as _segment

# —— 兼容旧私有名（测试与本文件结论句仍用下划线风格）——
_READING_NOTE = _common.READING_NOTE
_NEXT_PERIOD_NOTE = _common.NEXT_PERIOD_NOTE
_is_segment_report = _common.is_segment_report
_segment_reading_note = _common.segment_reading_note
_roster_rename_note_lines = _common.roster_rename_note_lines
_cite = _common.cite
_cite_evidence = _common.cite_evidence
_scope = _common.scope_text
_fact_scope_label = _common.fact_scope_label
_column = _common.column_label
_period_label = _common.period_label
_cell_with_unit = _common.cell_with_unit
_sole_subject = _common.sole_subject
_prefix = _common.subject_prefix

_markdown = _format.markdown
_html = _format.html_doc

_segment_roster_lines = _segment.segment_roster_lines
_segment_peer_lines = _segment.segment_peer_lines
_segment_table_indicators = _segment.segment_table_indicators
_peer_lines = _segment.peer_lines
_chart_guide_lines = _segment.chart_guide_lines
_gap_lines = _gaps_mod.gap_lines
_gaps = _gaps_mod.collect_gaps

# 「本期结论」只提炼几条主线，完整指标集留给「主要指标与出处」表。
_LEAD_GENERIC = registry.ordering_list("lead_generic")
_LEAD_MAX = registry.ordering_int("lead_max")
_LEAD_NET_PROFIT = frozenset(registry.ordering_list("lead_net_profit"))
_LEAD_OPERATING = registry.ordering_list("lead_operating")
_EQUITY_COMPOSITION = registry.ordering_list("equity_composition_order")
_ASSET_DRIVERS = registry.ordering_list("asset_drivers_order")


@dataclass
class ReportDocument:
    title: str
    markdown: str
    html: str
    deliverable: str
    gaps: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        """要 PDF 时正文用 HTML，其余用 Markdown。两份内容的数字相同。"""
        if self.deliverable == "pdf":
            return self.html
        return self.markdown


def compose_report(
    plan: ReportPlan,
    facts: list[IndicatorFact],
    views: SupplementalResult | None = None,
    analysis: list[str] | None = None,
) -> ReportDocument:
    """按计划里的大纲逐节填写。某一节没有数据时只写缺失，不留空头结论。

    ``analysis`` 为「经营解读」节的段落（通常来自受约束 LLM 后处理）；
    未传入或为空时写从略句，不阻断整份报告。
    """
    supplemental = views or SupplementalResult()
    # 千元／万元等入口再折一次，避免抽取漏识别时组内表混单位。
    facts = ensure_amount_units(facts)
    series = compare_measures([fact.to_measure() for fact in facts])
    features = compute_features(plan, facts, supplemental, series)
    gaps = _gaps(plan, facts, series)
    # 结论句已不再自带出处，出处全靠主要指标表承载。大纲若缺这一节（外部
    # 构造的 ReportPlan 可能没有），补上，避免产出一份全文无出处的报告。
    outline = list(plan.outline)
    # 赛道分组版已用 segment_peer 承载指标表，不再追加 indicator_table 以免重复。
    if outline and not report_sections.outline_has(
        outline, "indicator_table", "segment_peer"
    ):
        outline.append(OutlineItem("主要指标与出处", "", "indicator_table"))
    note = _segment_reading_note(plan) if _is_segment_report(plan) else _READING_NOTE
    # 大纲自带「阅读说明」时不再额外 prepend 通用阅读注，避免双份。
    prepend_note = (
        note
        if outline and not report_sections.outline_has(outline, "reading_note")
        else ()
    )
    sections = [
        (
            item.title,
            _section(
                item.title,
                plan,
                facts,
                series,
                gaps,
                supplemental,
                analysis=analysis,
                features=features,
                section_id=getattr(item, "section_id", ""),
            ),
        )
        for item in outline
    ]
    if not sections:
        sections = [("正文", ["没有大纲，未写正文。"])]
    # 无效请求不该套一份财报阅读说明；大纲已含「阅读说明」时也不再 prepend。
    note = prepend_note if outline else ()
    title = plan.title or "研报"
    markdown = _markdown(title, sections, note)
    return ReportDocument(
        title=title,
        markdown=markdown,
        html=_html(title, sections, note),
        deliverable=plan.deliverable,
        gaps=gaps,
    )


# —— 章节 renderer 注册表 ——
# 每节只负责「把已有数据摆成文字」。数字仍全部来自 facts / series / features，
# sections.yaml 与 SKILL.md 只能选 renderer，不能注入正文或改计算。
# 新增章节：写一个 renderer + 注册一次；或直接在 SKILL.md 里复用已有 id。
@report_sections.register("reading_note")
def _render_reading_note(ctx: report_sections.SectionContext) -> list[str]:
    lines = list(
        _segment_reading_note(ctx.plan) if _is_segment_report(ctx.plan) else _READING_NOTE
    )
    # 改名备注唯一来源：skill.companies 第 4 段「披露又名」，只在阅读说明写一次。
    rename = _roster_rename_note_lines(ctx.plan)
    if rename:
        # 插在归因边界之前：通用说明末句仍收尾。
        insert_at = max(len(lines) - 1, 0)
        for offset, line in enumerate(rename):
            lines.insert(insert_at + offset, line)
    return lines


@report_sections.register("lead")
def _render_lead(ctx: report_sections.SectionContext) -> list[str]:
    return _lead(ctx.plan, ctx.current, ctx.series, ctx.views, ctx.state)


@report_sections.register("indicator_table")
def _render_indicator_table(ctx: report_sections.SectionContext) -> list[str]:
    return _indicator_lines(ctx.plan, ctx.current)


@report_sections.register("change")
def _render_change(ctx: report_sections.SectionContext) -> list[str]:
    return _change_lines(ctx.plan, ctx.series, ctx.state)


@report_sections.register("quarter")
def _render_quarter(ctx: report_sections.SectionContext) -> list[str]:
    return _quarter_lines(ctx.plan, ctx.views)


@report_sections.register("year_end")
def _render_year_end(ctx: report_sections.SectionContext) -> list[str]:
    return _year_end_lines(ctx.views, ctx.state)


@report_sections.register("observation")
def _render_observation(ctx: report_sections.SectionContext) -> list[str]:
    return _observation_lines(ctx.plan, ctx.current, ctx.views)


@report_sections.register("analysis")
def _render_analysis(ctx: report_sections.SectionContext) -> list[str]:
    lines = [line.strip() for line in (ctx.analysis or []) if line and line.strip()]
    return lines or [ANALYSIS_FALLBACK]


@report_sections.register("segment_roster")
def _render_segment_roster(ctx: report_sections.SectionContext) -> list[str]:
    return _segment_roster_lines(ctx.plan, ctx.current)


@report_sections.register("segment_peer")
def _render_segment_peer(ctx: report_sections.SectionContext) -> list[str]:
    return _segment_peer_lines(ctx.plan, ctx.current)


@report_sections.register("chart_guide")
def _render_chart_guide(ctx: report_sections.SectionContext) -> list[str]:
    return _chart_guide_lines(ctx.series, ctx.plan, ctx.current)


@report_sections.register("next_period")
def _render_next_period(ctx: report_sections.SectionContext) -> list[str]:
    # 「景气周期」「细分赛道」等旧大纲标题也映射到这里：宏观从略，指向下期预告。
    # 行业文案按 indicator_set 从配置取，配置只提供定性句、不提供数字。
    notes = notes_for_plan(ctx.plan)
    return list(notes.next_period or _NEXT_PERIOD_NOTE)


@report_sections.register("key_companies")
def _render_key_companies(ctx: report_sections.SectionContext) -> list[str]:
    peers = _peer_lines(ctx.plan, ctx.current)
    table = _indicator_lines(ctx.plan, ctx.current)
    no_peers = "同一指标没有两家以上的公司，不做公司间对比。"
    no_table = "没有抽到大纲里的指标。"
    if peers == [no_peers] and table == [no_table]:
        return ["重点公司本期没有抽到可核对指标。"]
    lines: list[str] = []
    if peers and peers[0] != no_peers:
        lines.extend(peers)
    if table and table[0] != no_table:
        if lines:
            lines.append("")
        lines.extend(table)
    return lines or ["重点公司本期没有抽到可核对指标。"]


@report_sections.register("trend")
def _render_trend(ctx: report_sections.SectionContext) -> list[str]:
    return _trend_lines(ctx.series, ctx.plan)


@report_sections.register("peer")
def _render_peer(ctx: report_sections.SectionContext) -> list[str]:
    return _peer_lines(ctx.plan, ctx.current)


@report_sections.register("gaps")
def _render_gaps(ctx: report_sections.SectionContext) -> list[str]:
    return _gap_lines(ctx.plan, ctx.current, ctx.gaps)


def _section(
    title: str,
    plan: ReportPlan,
    facts: list[IndicatorFact],
    series: list[SeriesCompare],
    gaps: list[str],
    views: SupplementalResult,
    analysis: list[str] | None = None,
    features: Features | None = None,
    section_id: str = "",
) -> list[str]:
    """按 section_id（SKILL.md 声明）或标题别名查注册表；未绑定写占位句。

    这里不再有标题 if-else：新增报告类型/章节不用改本函数，只要章节能
    绑定到一个已注册 renderer。查不到时保留原有占位文案，不静默留空。
    """
    current = [fact for fact in facts if fact.period == plan.period]
    state = features or compute_features(plan, facts, views, series)
    ctx = report_sections.SectionContext(
        title=title,
        plan=plan,
        facts=facts,
        current=current,
        series=series,
        gaps=gaps,
        views=views,
        state=state,
        analysis=analysis,
    )
    resolved = report_sections.resolve(title, section_id)
    return resolved[1](ctx) if resolved is not None else report_sections.fallback_lines()


def _lead(
    plan: ReportPlan,
    current: list[IndicatorFact],
    series: list[SeriesCompare],
    views: SupplementalResult | None = None,
    features: Features | None = None,
) -> list[str]:
    if plan.kind == "industry":
        names = "、".join(plan.subjects) if plan.subjects else "没有公司"
        line = f"{plan.industry}行业 {plan.period} 覆盖：{names}。"
        return [line, *_conclusion_sentences(plan, current, series, views, features)]
    if not current:
        return [f"{plan.period} 没有抽到指标。"]
    return _conclusion_sentences(plan, current, series, views, features)


def _conclusion_sentences(
    plan: ReportPlan,
    facts: list[IndicatorFact],
    series: list[SeriesCompare],
    views: SupplementalResult | None = None,
    features: Features | None = None,
) -> list[str]:
    """本期结论：2–3 句主线提炼，不是指标平铺。

    只写已抽取的数 + 同比／较上年末构成关系；不写因果、不写「低位／稳健」等主观词。
    完整清单仍在「主要指标与出处」。
    """
    yoy_changes = _lead_yoy_changes(series)
    year_end = {
        item.name: item
        for item in (views.year_ends if views else [])
    }
    by_name: dict[str, IndicatorFact] = {}
    for fact in facts:
        by_name.setdefault(fact.name, fact)
    custom_lead = _lead_priority(plan, by_name)
    subjects = {fact.subject for fact in facts}
    lines: list[str] = []

    if not custom_lead:
        # 经营主线压成一句：营收／扣非／经营现金流并列，避免结论只剩一句营收。
        operating = _operating_lead_clause(plan, by_name, yoy_changes, subjects)
        if operating:
            lines.append(operating)
        equity = _equity_revaluation_lead(views, features)
        if equity:
            lines.append(equity)
        if lines:
            return lines[:_LEAD_MAX]

    # 有行业口径就按档案出句，否则用通用优先级：配置顺序在前，带 lead 标记的补在后面。
    priority = custom_lead or _generic_lead_priority()
    used_profit = False
    for name in priority:
        if len(lines) >= _LEAD_MAX:
            break
        if name not in by_name:
            continue
        if name in _LEAD_NET_PROFIT and used_profit:
            continue
        fact = by_name[name]
        if name in _LEAD_NET_PROFIT:
            used_profit = True
        period_text = "" if fact.basis in (POINT, STOCK) else _period_label(fact)
        bits = [
            f"{_prefix(plan, fact.subject, subjects)}{period_text}{fact.name}为 "
            f"{fact.value}{fact.unit}"
        ]
        rate_bit = _lead_rate_bit(fact, yoy_changes)
        if rate_bit:
            bits.append(rate_bit)
        ye = year_end.get(name)
        if ye and ye.change:
            bits.append(f"较上年末 {ye.change}")
        lines.append("，".join(bits) + "。")
    if lines:
        return lines
    return _indicator_sentences(plan, facts[:_LEAD_MAX], series)


def _lead_priority(
    plan: ReportPlan, by_name: dict[str, IndicatorFact]
) -> tuple[str, ...]:
    """本期结论的行业口径；没有就用空元组表示「走通用口径」。

    顺序：报告所属行业档案的 ``lead`` → 出现某品类专属指标（``lead_hint``）
    就取该品类口径。两条依据都在 industries.yaml，本函数不出现任何行业名。
    """
    configured = lead_priority(profile_key_for_plan(plan))
    if configured:
        return configured
    for hints, lead in lead_hint_profiles():
        if any(name in by_name for name in hints):
            return lead
    return ()


def _generic_lead_priority() -> tuple[str, ...]:
    """通用结论优先级：lead_generic 的配置顺序在前，带 lead 标记的指标补在后面。

    配置降级（indicators.yaml 读不到）时 lead_names 返回空，退化成纯配置顺序，
    不会因为缺一份派生名单而漏写结论。
    """
    extra = tuple(name for name in registry.lead_names() if name not in _LEAD_GENERIC)
    return (*_LEAD_GENERIC, *extra)


def _lead_yoy_changes(series: list[SeriesCompare]) -> dict[tuple[str, str, str], Change]:
    """结论与正文共用的同比 Change；跳过环比推导值。"""
    changes: dict[tuple[str, str, str], Change] = {}
    for item in series:
        for period, change in item.yoy.items():
            if change.source == "derived" or not change.rate:
                continue
            changes[(item.subject, item.name, period)] = change
    return changes


def _lead_rate_bit(
    fact: IndicatorFact, yoy_changes: dict[tuple[str, str, str], Change]
) -> str:
    change = yoy_changes.get((fact.subject, fact.name, fact.period))
    if change is None:
        return ""
    return render_yoy_rate_bit(fact.value, change)


def _operating_lead_clause(
    plan: ReportPlan,
    by_name: dict[str, IndicatorFact],
    yoy_changes: dict[tuple[str, str, str], Change],
    subjects: set[str],
) -> str:
    """营收／扣非／经营现金流压成一句；缺项就跳过，不硬凑。"""
    bits: list[str] = []
    period_text = ""
    for name in _LEAD_OPERATING:
        fact = by_name.get(name)
        if fact is None:
            continue
        if not period_text and fact.basis not in (POINT, STOCK):
            period_text = _period_label(fact)
        piece = f"{fact.name}为 {fact.value}{fact.unit}"
        rate_bit = _lead_rate_bit(fact, yoy_changes)
        if rate_bit:
            piece += f"（{rate_bit}）"
        bits.append(piece)
    if not bits:
        return ""
    head = _prefix(plan, next(iter(subjects), ""), subjects) + (period_text or "")
    return f"{head}{'、'.join(bits)}。"


def _equity_revaluation_lead(
    views: SupplementalResult | None, features: Features | None = None
) -> str:
    """把「权益大增但主因是 OCI 重估」提进结论；增量过小或缺拆解则不写。"""
    if features is not None and not features.equity_growth_large:
        return ""
    if views is None or not views.year_ends or not views.balance_moves:
        return ""
    equity = next((item for item in views.year_ends if is_equity_row(item.name)), None)
    oci = next((item for item in views.balance_moves if item.name == "其他综合收益"), None)
    retained = next((item for item in views.balance_moves if item.name == "未分配利润"), None)
    if equity is None or oci is None or not equity.prior_value or not equity.change:
        return ""
    equity_delta = signed_delta(equity.value, equity.prior_value)
    oci_delta = signed_delta(oci.end_value, oci.prior_value)
    if equity_delta is None or oci_delta is None or equity_delta == 0:
        return ""
    oci_share = _share_of(oci_delta, equity_delta)
    if oci_share is None or oci_share < 40:
        # 重估不是主导时不占用结论篇幅，细节仍在「较上年末」节。
        return ""
    retained_bit = ""
    if retained is not None:
        retained_delta = signed_delta(retained.end_value, retained.prior_value)
        if retained_delta is not None:
            retained_share = _share_of(retained_delta, equity_delta)
            if retained_share is not None:
                retained_bit = f"，经营留存（未分配利润增减）约 {retained_share:.0f}%"
    return (
        f"需注意：{equity.name}为 {equity.value}{equity.unit}"
        f"（较上年末 {equity.change}），其中其他综合收益增减约 {oci_share:.0f}%"
        f"{retained_bit}"
        "——账面权益扩张主要来自其他权益工具投资公允价值重估（计入 OCI），"
        "不把公允价值变动等同于经营利润。"
    )


def _indicator_sentences(
    plan: ReportPlan, facts: list[IndicatorFact], series: list[SeriesCompare]
) -> list[str]:
    """兜底：按事实顺序写「是什么 + 变化多少」，条数由调用方截断。"""
    yoy_changes = _lead_yoy_changes(series)
    subjects = {fact.subject for fact in facts}
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in facts:
        key = (fact.subject, fact.name, fact.scope)
        if key in seen:
            continue
        seen.add(key)
        period_text = "" if fact.basis in (POINT, STOCK) else _period_label(fact)
        rate_bit = _lead_rate_bit(fact, yoy_changes)
        change_text = f"，{rate_bit}" if rate_bit else ""
        lines.append(
            f"{_prefix(plan, fact.subject, subjects)}{period_text}{fact.name}为 {fact.value}{fact.unit}"
            f"{change_text}。"
        )
    return lines or ["本期没有抽到指标。"]


def _indicator_lines(plan: ReportPlan, current: list[IndicatorFact]) -> list[str]:
    rows = [fact for fact in current if not plan.indicators or fact.name in plan.indicators]
    if not rows:
        return ["没有抽到大纲里的指标。"]
    # 单公司报告没有「公司」列：21 行重复同一全称不提供信息，主体看标题。
    subjects = {fact.subject for fact in rows}
    if _sole_subject(plan, subjects):
        lines = [
            "| 指标 | 数值 | 期间口径／范围／取数列 | 出处 |",
            "| --- | --- | --- | --- |",
        ]
    else:
        lines = [
            "| 公司 | 指标 | 数值 | 期间口径／范围／取数列 | 出处 |",
            "| --- | --- | --- | --- | --- |",
        ]
    with_subject = "| 公司 |" in lines[0]
    for fact in rows:
        scope = _fact_scope_label(fact)
        subject_cell = f"{fact.subject} | " if with_subject else ""
        display = _cell_with_unit(fact, indicator=fact.name)
        lines.append(
            f"| {subject_cell}{fact.name} | {display} | "
            f"{_period_label(fact)}／{scope}／{_column(fact)} | {_cite(fact)} |"
        )
    # 个别指标的特别注记（如净息差未注明年化、单位折算）跟在表格下面当脚注。
    notes = sorted({note for fact in rows if (note := disclosure_note(fact.name))})
    lines.extend(f"注：{note}。" for note in notes)
    converted = [
        fact
        for fact in rows
        if getattr(fact, "unit_converted", False) and getattr(fact, "raw_unit", "")
    ]
    if converted:
        units = "、".join(sorted({fact.raw_unit for fact in converted}))
        lines.append(f"注：上表部分金额由{units}折算为元，原值见溯源附件。")
    return lines


def _quarter_lines(plan: ReportPlan, views: SupplementalResult) -> list[str]:
    """单季列与累计列并排。两个同比都是表格原文披露值，不做倒推。"""
    rows = [
        item
        for item in views.quarters
        if not plan.indicators or item.name in plan.indicators
    ]
    if not rows:
        return ["本期季报没有同时披露单季列与累计列，不写单季节奏。"]
    lines = [
        f"| 指标 | {plan.period} 单季（本报告期） | 单季同比（原文披露） | "
        "年初至报告期末累计 | 累计同比（原文披露） | 出处 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in rows:
        lines.append(
            f"| {item.name} | {item.quarter_value}{item.unit} | "
            f"{item.quarter_yoy or '未披露'} | {item.cumulative_value}{item.unit} | "
            f"{item.cumulative_yoy or '未披露'} | {_cite_evidence(item.evidence)} |"
        )
    lines.append(
        "单季列与累计列来自同一张主要财务数据表；单季值是原文披露的「本报告期」，"
        "不是由累计值相减还原。单季同比比的是上年同期单季，累计同比比的是上年同期"
        "累计，「同比与环比」一节里的单季环比比的是相邻单季，三者基准不同，不互相换算。"
    )
    return lines


def _year_end_lines(
    views: SupplementalResult, features: Features | None = None
) -> list[str]:
    if not views.year_ends:
        return ["本期季报没有披露较上年末变动，不写这一节。"]
    lines = [
        "| 指标 | 期末值 | 较上年末（原文披露） | 出处 |",
        "| --- | --- | --- | --- |",
    ]
    for item in views.year_ends:
        lines.append(
            f"| {item.name} | {item.value}{item.unit} | {item.change} | "
            f"{_cite_evidence(item.evidence)} |"
        )
    lines.append(
        "「较上年末」是与上年度末时点值比较，不是同比、也不是环比；"
        "期末时点存量不能与累计发生额相减。期末值按原文所在段落的单位呈现，"
        "同一指标在主要财务数据表可能换用另一种单位（如资产总计正文用亿元、"
        "表格用百万元），数值等价、不重复计入。"
    )
    composition = _equity_composition_lines(views, features)
    if composition:
        lines.append("")
        lines.extend(composition)
    return lines


# 归母权益构成拆解优先科目（顺序即叙述优先级）。
_EQUITY_COMPOSITION = registry.ordering_list("equity_composition_order")
_ASSET_DRIVERS = registry.ordering_list("asset_drivers_order")


def _equity_composition_lines(
    views: SupplementalResult, features: Features | None = None
) -> list[str]:
    """用合并资产负债表科目增减拆归母权益——只并列①级数字关系 + ②级原文原因。

    权益增量相对构成科目变动过小（Features.equity_growth_large=False）时不写占比。
    """
    moves = {item.name: item for item in views.balance_moves}
    equity = next((item for item in views.year_ends if is_equity_row(item.name)), None)
    if equity is None or not equity.prior_value:
        return []
    equity_delta = signed_delta(equity.value, equity.prior_value)
    if equity_delta is None:
        return []
    parts: list[tuple[str, object, str, object]] = []
    for name in _EQUITY_COMPOSITION:
        move = moves.get(name)
        if move is None:
            continue
        delta = signed_delta(move.end_value, move.prior_value)
        if delta is None:
            continue
        parts.append((name, delta, format_money_delta(delta), move.evidence))
    if not parts:
        return []

    show_share = equity_share_applicable(features) if features else True
    share_header = "占归母权益增量" if show_share else "占权益增量"
    lines = [
        "### 归母权益较上年末构成（合并资产负债表科目增减）",
        f"| 科目 | 本期末 | 上年末 | 增减 | {share_header} | 出处 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    share_bits: list[str] = []
    for name, delta, delta_text, evidence in parts:
        move = moves[name]
        if show_share:
            share = _share_of(delta, equity_delta)
            share_text = f"{share:.0f}%" if share is not None else "—"
            if share is not None:
                share_bits.append(f"{name}约 {share_text}")
        else:
            share_text = "不适用"
        lines.append(
            f"| {name} | {move.end_value}{move.unit} | {move.prior_value}{move.unit} | "
            f"{delta_text} | {share_text} | {_cite_evidence(evidence)} |"
        )
    equity_delta_text = format_money_delta(equity_delta)
    lines.append(
        f"归属于上市公司股东的所有者权益本期末 {equity.value}{equity.unit}、"
        f"上年度末 {equity.prior_value}{equity.unit}，增量 {equity_delta_text}"
        f"（由主表两列相减，非原文单独披露的增减额）。"
    )
    if not show_share and features is not None:
        lines.append(equity_share_exemption_line(features))
    elif share_bits:
        lines.append(
            "上述三项增减占归母权益增量的比重为 "
            + "、".join(share_bits)
            + "——只陈述资产负债表已披露科目的数量关系，"
            "不把公允价值变动等同于经营利润。"
        )
    asset_bits: list[str] = []
    for name in _ASSET_DRIVERS:
        move = moves.get(name)
        if move is None:
            continue
        delta = signed_delta(move.end_value, move.prior_value)
        if delta is None:
            continue
        asset_bits.append(
            f"{name} 本期末 {move.end_value}{move.unit}、上年末 {move.prior_value}{move.unit}，"
            f"增减 {format_money_delta(delta)}（{_cite_evidence(move.evidence)}）"
        )
    if asset_bits:
        lines.append("资产端对照：" + "；".join(asset_bits) + "。")
    for reason in views.year_end_reasons:
        cite = _cite_evidence(reason.evidence)
        text = reason.reason.strip().rstrip("。")
        lines.append(f"原文披露（{reason.name}）：{text}（{cite}）。")
    if any("其他权益工具投资" in (item.reason or "") for item in views.year_end_reasons):
        lines.append(
            "季报变动原因未点名「其他权益工具投资」的具体标的名称；本期不推断投资对象。"
        )
    return lines


def _share_of(part, whole) -> float | None:
    if whole is None or whole == 0:
        return None
    return float(part / whole * 100)


def _observation_lines(
    plan: ReportPlan, current: list[IndicatorFact], views: SupplementalResult
) -> list[str]:
    """只并列可核对的数字关系，不写「因为…所以…」。

    这一节的边界是刻意的：季报「经营情况简要分析」通篇是数字罗列、零归因句，
    任何因果解释都得靠模型补，补出来的句子没有原文可挂。宁可留白。
    """
    lines: list[str] = []
    for item in views.quarters:
        if not item.quarter_yoy or not item.cumulative_yoy:
            continue
        relation = _rate_relation(item.quarter_yoy, item.cumulative_yoy)
        if not relation:
            continue
        lines.append(
            f"{item.name}：年初至报告期末累计同比 {item.cumulative_yoy}，"
            f"{plan.period} 单季同比 {item.quarter_yoy}，单季增速{relation}累计增速"
            "（两者均为季报原文披露值）。"
        )
    for item in views.year_ends:
        if "不良" not in item.name:
            continue
        lines.append(
            f"{item.name} {item.value}{item.unit}，较上年末 {item.change}（原文披露值）。"
        )
    stocks = sorted(
        {fact.name for fact in current if fact.basis == STOCK and fact.period == plan.period}
    )
    if not stocks and views.year_ends:
        # 制造业季报「较上年末」表常只进 supplemental，不进 IndicatorFact；
        # 仍须标明期末存量，避免读者把 +300% 当成期间发生额取错。
        stocks = [item.name for item in views.year_ends]
    if stocks:
        lines.append(
            f"{'、'.join(stocks)}为期末时点余额（较上年末变动为季报原文披露值），"
            "不是累计发生额，不与前三季度累计值相减。"
        )
    # 观察收尾句按行业档案 notes.observation；银行等专属口径只写在 YAML。
    observation = notes_for_plan(plan).observation.strip()
    lines.append(
        observation
        or "以上只并列原文披露的数字关系，不构成因果归因。"
    )
    return lines or ["没有可核对的数字关系，不写观察。"]


def _rate_relation(single: str, cumulative: str) -> str:
    left, right = parse_number(single), parse_number(cumulative)
    if not left or not right:
        return ""
    if left[0] > right[0]:
        return "高于"
    if left[0] < right[0]:
        return "低于"
    return "持平于"


def _change_lines(
    plan: ReportPlan,
    series: list[SeriesCompare],
    features: Features | None = None,
) -> list[str]:
    state = features or Features()
    lines: list[str] = []
    subjects = {item.subject for item in series}
    current_values = {
        (item.subject, item.name): next(
            (value for period, value in item.points if period == plan.period),
            "",
        )
        for item in series
    }
    for item in series:
        if plan.indicators and item.name not in plan.indicators:
            continue
        prefix = _prefix(plan, item.subject, subjects)
        yoy = item.yoy.get(plan.period) if "同比" in plan.dimensions else None
        qoq = item.qoq.get(plan.period) if "环比" in plan.dimensions else None
        if yoy and yoy.source == "reported":
            # 原文只给了增减幅度、没给基期绝对值，不能编一个增减额出来。
            note = _join_notes(
                _basis_note(item, plan.period, "同比", yoy), "季报原文披露", yoy.note
            )
            lines.append(
                f"{prefix}{item.name} {plan.period} 同比 {yoy.rate}（{note}）。"
            )
        elif yoy:
            note = _join_notes(_basis_note(item, plan.period, "同比", yoy), yoy.note)
            current = current_values.get((item.subject, item.name), "")
            lines.append(
                render_yoy_sentence(
                    prefix=prefix,
                    name=item.name,
                    period=plan.period,
                    current_value=current or yoy.delta,
                    unit=item.unit,
                    change=yoy,
                    note=note,
                    features=state,
                )
            )
        if qoq:
            base = (
                f"{qoq.base_period}（由累计值推导）"
                if qoq.source == "derived"
                else qoq.base_period
            )
            note = _join_notes(_basis_note(item, plan.period, "环比", qoq), qoq.note)
            lines.append(
                _change_sentence(prefix, item.name, "环比", base, qoq.delta, qoq.rate, note)
            )
    return lines or ["没有可写入的同比或环比。"]


def _basis_note(
    item: SeriesCompare, period: str, label_text: str, change: Change
) -> str:
    """同比/环比句必须自带口径，累计同比和推导单季环比不能并列裸奔。"""
    if change.source == "derived":
        # change.note 已经写明「单季环比由相邻累计值相减还原，非原文披露」，
        # 再叠一层口径说明只会变成重复括号。
        return ""
    key = item.bases.get(period, "")
    if key.startswith("cumulative"):
        return f"累计口径{label_text}"
    if key.startswith("single"):
        return f"单季口径{label_text}"
    if key.startswith("point"):
        return "比率口径，变动用百分点"
    return ""


def _join_notes(*parts: str) -> str:
    return "；".join(part for part in parts if part)



def _change_sentence(
    subject: str, name: str, label: str, base: str, delta: str, rate: str, note: str
) -> str:
    tail = f"，幅度 {rate}" if rate else ""
    extra = f"（{note}）" if note else ""
    return f"{subject}{name}较 {base} {label}增减 {delta}{tail}{extra}。"


def _trend_lines(series: list[SeriesCompare], plan: ReportPlan | None = None) -> list[str]:
    """趋势句。plan 传行业/多主体计划时带全称主语，单公司报告省略。

    plan 允许缺省：纯计算调用（不含报告上下文）按多主体处理，保留全称。
    """
    subjects = {item.subject for item in series}
    single = plan is not None and _sole_subject(plan, subjects)
    lines: list[str] = []
    for item in series:
        prefix = "" if single else item.subject
        for step in item.trend:
            kind = "环比" if step.adjacent else "跨期"
            rate = f"，幅度 {step.rate}" if step.rate else ""
            note = f"（{step.note}）" if step.note else ""
            lines.append(
                f"{prefix}{item.name} {step.earlier} → {step.later}"
                f"（{kind}）增减 {step.delta}{rate}{note}。"
            )
    if lines:
        return lines
    if any(len(item.points) > 1 for item in series):
        # 有多期但全不可比：明确说明，不用一句「没有多期数据」糊弄过去。
        return [
            "各期口径或覆盖期数不同，不做跨期增减计算；"
            "累计值只用于同口径同比与单季还原。"
        ]
    return ["没有多期数据，不写趋势。"]


