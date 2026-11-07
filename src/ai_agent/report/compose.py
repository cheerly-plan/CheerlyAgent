"""按大纲写成研报。正文里的数字只来自抽取结果和计算工具。

不调用模型，因此不会补一个大纲里有、但表格里没抽到的数。

章节渲染按职责拆到 compose_*.py；本模块负责编排与 renderer 注册。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_agent.report import sections as report_sections
from ai_agent.report.compare import SeriesCompare, compare_measures
from ai_agent.report.extract import IndicatorFact, ensure_amount_units
from ai_agent.report.industry import notes_for_plan
from ai_agent.report.parse import OutlineItem, ReportPlan
from ai_agent.report.phrases import ANALYSIS_FALLBACK
from ai_agent.report.features import Features, compute_features
from ai_agent.report.supplemental import SupplementalResult
from ai_agent.report import compose_common as _common
from ai_agent.report import compose_format as _format
from ai_agent.report import compose_gaps as _gaps_mod
from ai_agent.report import compose_lead as _lead_mod
from ai_agent.report import compose_segment as _segment

# —— 兼容旧私有名（测试与 renderer 仍用下划线风格）——
_READING_NOTE = _common.READING_NOTE
_NEXT_PERIOD_NOTE = _common.NEXT_PERIOD_NOTE
_is_segment_report = _common.is_segment_report
_segment_reading_note = _common.segment_reading_note
_roster_rename_note_lines = _common.roster_rename_note_lines

_markdown = _format.markdown
_html = _format.html_doc

_segment_roster_lines = _segment.segment_roster_lines
_segment_peer_lines = _segment.segment_peer_lines
_segment_table_indicators = _segment.segment_table_indicators
_peer_lines = _segment.peer_lines
_chart_guide_lines = _segment.chart_guide_lines
_gap_lines = _gaps_mod.gap_lines
_gaps = _gaps_mod.collect_gaps

_LEAD_GENERIC = _lead_mod.LEAD_GENERIC
_LEAD_MAX = _lead_mod.LEAD_MAX
_LEAD_NET_PROFIT = _lead_mod.LEAD_NET_PROFIT
_LEAD_OPERATING = _lead_mod.LEAD_OPERATING
_lead = _lead_mod.lead_lines
_indicator_lines = _lead_mod.indicator_lines
_quarter_lines = _lead_mod.quarter_lines
_year_end_lines = _lead_mod.year_end_lines
_observation_lines = _lead_mod.observation_lines
_change_lines = _lead_mod.change_lines
_trend_lines = _lead_mod.trend_lines
_equity_composition_lines = _lead_mod.equity_composition_lines


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
