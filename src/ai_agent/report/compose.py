"""按大纲写成研报。正文里的数字只来自抽取结果和计算工具。

不调用模型，因此不会补一个大纲里有、但表格里没抽到的数。
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from ai_agent.kb.ingest.meta import name_in_text
from ai_agent.report import registry, sections
from ai_agent.report.basis import POINT, STOCK, disclosure_note, label as basis_label
from ai_agent.report.compare import (
    NOT_COMPARABLE_NOTE,
    STOCK_NOTE,
    Change,
    Measure,
    SeriesCompare,
    compare_measures,
    parse_number,
)
from ai_agent.report.extract import IndicatorFact, citation_anchor, ensure_amount_units
from ai_agent.report.industry import (
    BANK,
    default_indicators,
    industry_of,
    is_excluded,
    type_for_segment,
)
from ai_agent.report.parse import OutlineItem, ReportPlan
from ai_agent.report.skills import get_skill
from ai_agent.report.features import Features, compute_features
from ai_agent.report.meta import infer_scope, scope_label
from ai_agent.report.rules import (
    equity_share_applicable,
    equity_share_exemption_line,
    render_yoy_rate_bit,
    render_yoy_sentence,
)
from ai_agent.report.supplemental import (
    SupplementalResult,
    format_money_delta,
    signed_delta,
)

_SCOPE = {"consolidated": "合并", "parent": "母公司"}
_UNKNOWN_SCOPE = "—"

# 开头统一声明一次：能按模板推定的不写「找不到」；文案避免阿拉伯数字（校验器）。
_READING_NOTE = (
    "阅读说明：金额默认统一为元；若由千元／万元／百万元折算，指标表会注明原单位。",
    "范围：主要财务数据等主表按季报模板推定为合并，母公司口径单独标注；"
    "期间按列名区分——本报告期为单季、年初至报告期末为累计、期末为时点余额。",
    "基期文件未入库属于数据缺口，与口径推定无关，见「缺失与口径」。"
    "方括号短标识为知识库文件前缀，完整出处见溯源附件。",
)

_READING_NOTE_SEGMENT = (
    "阅读说明：本文为半导体代表公司财报对比分析（赛道分组版）；"
    "金额默认统一为元，主表范围按季报模板推定为合并，期间按列名区分单季／累计／时点。",
    "名单备注：韦尔股份现名豪威集团（股票代码暂略）；表头沿用技能名单「韦尔股份」，"
    "出处文件名以披露主体「豪威集团」为准。",
    "三级归因边界：已抽取的数据事实可写；原文经营说明须标出处后方可引用；"
    "无出处的第三方推断一律不写。完整标识与原文片段见溯源附件。",
)

# 「本期结论」只提炼几条主线，完整指标集留给「主要指标与出处」表。
# 顺序即优先级；同组里靠前的命中后，靠后的同组名不再重复（如已有归母则不再写净利润）。
_LEAD_BANK = (
    "营业收入",
    "归属于母公司所有者的净利润",
    "净利润",
    "净息差",
    "不良贷款率",
    "不良贷款拨备覆盖率",
)
_LEAD_GENERIC = (
    "营业收入",
    "扣非净利润",
    "净利润",
    "经营活动产生的现金流量净额",
    "毛利率",
)
_LEAD_MAX = 3
_LEAD_NET_PROFIT = frozenset({"归属于母公司所有者的净利润", "净利润", "扣非净利润"})
# 结论首句优先并列的经营主线（有则写入同一句，不再拆成多句平铺）。
_LEAD_OPERATING = (
    "营业收入",
    "扣非净利润",
    "经营活动产生的现金流量净额",
)


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
    series = compare_measures([_measure(fact) for fact in facts])
    features = compute_features(plan, facts, supplemental, series)
    gaps = _gaps(plan, facts, series)
    # 结论句已不再自带出处，出处全靠主要指标表承载。大纲若缺这一节（外部
    # 构造的 ReportPlan 可能没有），补上，避免产出一份全文无出处的报告。
    outline = list(plan.outline)
    # 赛道分组版已用「组内对比」承载指标表，不再追加「主要指标与出处」以免重复。
    if (
        outline
        and not any(item.title == "主要指标与出处" for item in outline)
        and not any(item.title == "组内对比" for item in outline)
    ):
        outline.append(OutlineItem("主要指标与出处", ""))
    note = _READING_NOTE_SEGMENT if _is_segment_report(plan) else _READING_NOTE
    # 大纲自带「阅读说明」时不再额外 prepend 通用阅读注，避免双份。
    prepend_note = note if outline and not any(item.title == "阅读说明" for item in outline) else ()
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


def _measure(fact: IndicatorFact) -> Measure:
    return Measure(
        subject=fact.subject,
        name=fact.name,
        period=fact.period,
        value=fact.value,
        unit=fact.unit,
        scope=fact.scope,
        reported_yoy=fact.reported_yoy,
        cumulative=fact.cumulative,
        basis=fact.basis,
        coverage_months=fact.coverage_months,
        source_key=(fact.evidence.file_id or fact.evidence.filename or "").strip(),
    )


_ANALYSIS_FALLBACK = "经营解读：本期季报未提供可核对的归因表述，本节从略。"

_NEXT_PERIOD_NOTE = (
    "景气周期（全球半导体销售额增速、晶圆厂稼动率、存储／模拟价格等）"
    "依赖外部数据源（WSTS／SEMI／协会），本期流水线未接入，不推断周期位置。",
    "下一期预告需明确：数据源清单、入库表结构、更新频率；接入前本报告只做公司财报赛道分组对比。",
)


def _is_segment_report(plan: ReportPlan) -> bool:
    return plan.skill_id == "semiconductor_industry_report" or any(
        item.title in {"赛道分组", "组内对比"} for item in plan.outline
    )


# —— 章节 renderer 注册表 ——
# 每节只负责「把已有数据摆成文字」。数字仍全部来自 facts / series / features，
# sections.yaml 与 SKILL.md 只能选 renderer，不能注入正文或改计算。
# 新增章节：写一个 renderer + 注册一次；或直接在 SKILL.md 里复用已有 id。
@sections.register("reading_note")
def _render_reading_note(ctx: sections.SectionContext) -> list[str]:
    return list(_READING_NOTE_SEGMENT if _is_segment_report(ctx.plan) else _READING_NOTE)


@sections.register("lead")
def _render_lead(ctx: sections.SectionContext) -> list[str]:
    return _lead(ctx.plan, ctx.current, ctx.series, ctx.views, ctx.state)


@sections.register("indicator_table")
def _render_indicator_table(ctx: sections.SectionContext) -> list[str]:
    return _indicator_lines(ctx.plan, ctx.current)


@sections.register("change")
def _render_change(ctx: sections.SectionContext) -> list[str]:
    return _change_lines(ctx.plan, ctx.series, ctx.state)


@sections.register("quarter")
def _render_quarter(ctx: sections.SectionContext) -> list[str]:
    return _quarter_lines(ctx.plan, ctx.views)


@sections.register("year_end")
def _render_year_end(ctx: sections.SectionContext) -> list[str]:
    return _year_end_lines(ctx.views, ctx.state)


@sections.register("observation")
def _render_observation(ctx: sections.SectionContext) -> list[str]:
    return _observation_lines(ctx.plan, ctx.current, ctx.views)


@sections.register("analysis")
def _render_analysis(ctx: sections.SectionContext) -> list[str]:
    lines = [line.strip() for line in (ctx.analysis or []) if line and line.strip()]
    return lines or [_ANALYSIS_FALLBACK]


@sections.register("segment_roster")
def _render_segment_roster(ctx: sections.SectionContext) -> list[str]:
    return _segment_roster_lines(ctx.plan, ctx.current)


@sections.register("segment_peer")
def _render_segment_peer(ctx: sections.SectionContext) -> list[str]:
    return _segment_peer_lines(ctx.plan, ctx.current)


@sections.register("chart_guide")
def _render_chart_guide(ctx: sections.SectionContext) -> list[str]:
    return _chart_guide_lines(ctx.series, ctx.plan, ctx.current)


@sections.register("next_period")
def _render_next_period(ctx: sections.SectionContext) -> list[str]:
    # 「景气周期」「细分赛道」等旧大纲标题也映射到这里：宏观从略，指向下期预告。
    return list(_NEXT_PERIOD_NOTE)


@sections.register("key_companies")
def _render_key_companies(ctx: sections.SectionContext) -> list[str]:
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


@sections.register("trend")
def _render_trend(ctx: sections.SectionContext) -> list[str]:
    return _trend_lines(ctx.series, ctx.plan)


@sections.register("peer")
def _render_peer(ctx: sections.SectionContext) -> list[str]:
    return _peer_lines(ctx.plan, ctx.current)


@sections.register("gaps")
def _render_gaps(ctx: sections.SectionContext) -> list[str]:
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
    ctx = sections.SectionContext(
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
    resolved = sections.resolve(title, section_id)
    return resolved[1](ctx) if resolved is not None else sections.fallback_lines()


def _sole_subject(plan: ReportPlan, subjects: set[str]) -> bool:
    """单公司报告正文省略主语，全称由标题承担。

    行业报告或多主体并列时必须保留全称——那时省略主语就分不清说的是哪家。
    kind 不是 industry 且主体唯一即视为单公司报告；fact 的主体与计划主体
    不一致（别名解析差异之类）时主体仍是唯一的，省略同样成立。
    """
    return plan.kind != "industry" and len(subjects) <= 1


def _prefix(plan: ReportPlan, subject: str, subjects: set[str]) -> str:
    """句首主语：单公司报告为空串，多主体报告用全称。"""
    return "" if _sole_subject(plan, subjects) else subject


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
    is_bank = _is_bank_lead(plan, by_name)
    subjects = {fact.subject for fact in facts}
    lines: list[str] = []

    if not is_bank:
        # 经营主线压成一句：营收／扣非／经营现金流并列，避免结论只剩一句营收。
        operating = _operating_lead_clause(plan, by_name, yoy_changes, subjects)
        if operating:
            lines.append(operating)
        equity = _equity_revaluation_lead(views, features)
        if equity:
            lines.append(equity)
        if lines:
            return lines[:_LEAD_MAX]

    # 前四项是固定的经营主线，其后的优先级由注册表按 lead_order 派生；
    # 配置降级时回退 _LEAD_GENERIC 原有后缀，行为不变。
    priority = (
        _LEAD_BANK
        if is_bank
        else (*_LEAD_GENERIC[:4], *registry.lead_names(fallback=_LEAD_GENERIC[4:]))
    )
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


def _is_bank_lead(plan: ReportPlan, by_name: dict[str, IndicatorFact]) -> bool:
    """银行结论走息差／资产质量优先级；不能仅凭「有营业收入」就判成银行。"""
    if getattr(plan, "company_type", "") == BANK:
        return True
    if plan.skill_id == "bank_financial_report":
        return True
    if industry_of(plan.subjects) == BANK:
        return True
    return any(
        name in by_name
        for name in ("净息差", "不良贷款率", "不良贷款拨备覆盖率", "资本充足率")
    )


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
    equity = next(
        (
            item
            for item in views.year_ends
            if "所有者权益" in item.name or "净资产" in item.name
        ),
        None,
    )
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
_EQUITY_COMPOSITION = ("其他综合收益", "未分配利润", "资本公积")
_ASSET_DRIVERS = ("其他权益工具投资", "其他非流动金融资产", "货币资金")


def _equity_composition_lines(
    views: SupplementalResult, features: Features | None = None
) -> list[str]:
    """用合并资产负债表科目增减拆归母权益——只并列①级数字关系 + ②级原文原因。

    权益增量相对构成科目变动过小（Features.equity_growth_large=False）时不写占比。
    """
    moves = {item.name: item for item in views.balance_moves}
    equity = next(
        (
            item
            for item in views.year_ends
            if "所有者权益" in item.name or "净资产" in item.name
        ),
        None,
    )
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
    # 银行专属口径注记只挂银行报告；其它类型模板严禁写净息差／PPOP。
    company_type = getattr(plan, "company_type", "") or industry_of(plan.subjects)
    if (
        company_type == BANK
        or industry_of(plan.subjects) == BANK
        or plan.skill_id == "bank_financial_report"
    ):
        lines.append(
            "以上只并列原文披露的数字关系，不构成因果归因。季报未披露单季净息差拆分"
            "所需的平均生息资产，也未披露拨备前利润（PPOP）的统一口径，两项均不推算。"
        )
    else:
        lines.append("以上只并列原文披露的数字关系，不构成因果归因。")
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


def _period_label(fact: IndicatorFact) -> str:
    """手工构造、没标口径的 fact 沿用旧文案，避免破坏纯计算调用。"""
    if not fact.basis:
        return fact.period
    return basis_label(fact.period, fact.basis, fact.coverage_months)


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


def _skill_segments(plan: ReportPlan) -> list[tuple[str, list[str]]]:
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


def _facts_for_subjects(
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


def _normalize_unit_label(unit: str) -> str:
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


def _peer_table_block(
    title: str, rows: list[IndicatorFact], *, min_subjects: int = 1
) -> list[str]:
    """组内对比表：金额已在抽取层尽量归一为元；仍有单位例外时脚注说明。"""
    subjects = {row.subject for row in rows}
    if len(subjects) < min_subjects:
        return []
    units = {_normalize_unit_label(row.unit) for row in rows if row.unit}
    lines = [title, "| 公司 | 数值 | 出处 |", "| --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row.subject} | {row.value}{row.unit} | {_cite(row)} |")
    money = units & {"元", "千元", "万元", "亿元", "百万元"}
    if len(money) > 1:
        lines.append(
            f"注：本表金额单位未统一（{'／'.join(sorted(money))}），"
            "暂不混比；抽取层未能归一的项保留原单位。"
        )
    return lines


def _segment_roster_lines(
    plan: ReportPlan, current: list[IndicatorFact]
) -> list[str]:
    groups = _skill_segments(plan)
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


def _cell_with_unit(fact: IndicatorFact | None, *, indicator: str = "") -> str:
    if fact is None:
        return "—"
    unit = (fact.unit or "").strip()
    if unit == "%":
        return f"{fact.value}%"
    if unit:
        return f"{fact.value}{unit}"
    # 金额类缺单位时默认补「元」，避免同行带单位、本格裸数字。
    money = {
        "营业收入",
        "扣非净利润",
        "经营活动产生的现金流量净额",
        "合同负债",
        "存货",
        "资本开支",
        "折旧",
    }
    if indicator in money or (not unit and _looks_like_money(fact.value)):
        return f"{fact.value}元"
    return str(fact.value)


def _looks_like_money(value: str) -> bool:
    text = (value or "").replace(",", "")
    return bool(text) and text.replace(".", "", 1).isdigit() and "," in (value or "")


def _find_fact(
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


def _wide_segment_table(
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
    money_names = {
        "营业收入",
        "扣非净利润",
        "经营活动产生的现金流量净额",
        "合同负债",
        "存货",
        "在手订单",
        "资本开支",
        "折旧",
    }
    for company in names:
        cells = [company]
        for indicator in indicators:
            fact = _find_fact(facts, company, indicator)
            cells.append(_cell_with_unit(fact, indicator=indicator))
            if fact and indicator in money_names:
                money_units.add(_normalize_unit_label(fact.unit or "元"))
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
            fact = _find_fact(facts, company, indicator)
            if fact is None:
                continue
            bits.append(f"{indicator}←{_cite(fact)}")
        if bits:
            cites.append(f"{company}：{'；'.join(bits)}")
    for cite in cites:
        lines.append(f"出处：{cite}。")
    if any(name == "韦尔股份" for name in names):
        lines.append(
            "注：韦尔股份现名豪威集团（603501）；表头沿用技能名单，出处以披露主体为准。"
        )
    return lines


def _margin_ladder_table(current: list[IndicatorFact]) -> list[str]:
    """全行业毛利率梯队总表（一眼可比）。"""
    from ai_agent.report.compare import parse_number

    rows: list[tuple[object, str, IndicatorFact]] = []
    for fact in current:
        if fact.name != "毛利率":
            continue
        parsed = parse_number(fact.value)
        if parsed is None:
            continue
        rows.append((parsed[0], fact.subject, fact))
    if len(rows) < 2:
        return []
    rows.sort(key=lambda item: item[0], reverse=True)
    lines = [
        "### 全行业毛利率梯队",
        "| 名次 | 公司 | 毛利率 | 出处 |",
        "| --- | --- | --- | --- |",
    ]
    for index, (_amount, subject, fact) in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {subject} | {fact.value}% | {_cite(fact)} |"
        )
    return lines


_SEGMENT_TABLE_CORE = 6
_OCF = "经营活动产生的现金流量净额"


def _segment_table_indicators(typed: list[str]) -> list[str]:
    """组内宽表列：主看项最多六列；模板含经营现金流则必保留（与封测表对齐）。"""
    core = list(typed[:_SEGMENT_TABLE_CORE])
    if _OCF in typed and _OCF not in core:
        core.append(_OCF)
    return core


def _segment_peer_lines(
    plan: ReportPlan, current: list[IndicatorFact]
) -> list[str]:
    groups = _skill_segments(plan)
    if not groups:
        return _peer_lines(plan, current)
    # 组内对比按赛道类型路由指标；共享集作兜底，避免把 Fabless 指标硬套 Foundry。
    shared = plan.indicators or default_indicators("semiconductor")
    lines: list[str] = []
    ladder = _margin_ladder_table(current)
    if ladder:
        lines.extend(ladder)
        lines.append("")
    for segment, names in groups:
        segment_facts = _facts_for_subjects(current, names)
        lines.append(f"### {segment}")
        if not segment_facts:
            lines.append("本期未抽到该赛道代表公司指标。")
            lines.append("")
            continue
        typed = default_indicators(type_for_segment(segment))
        # 赛道专属指标优先；抽不到的列在宽表里留空，缺口进「缺失」节。
        indicators = _segment_table_indicators(typed or list(shared))
        lines.extend(_wide_segment_table(segment, names, segment_facts, indicators))
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return lines or ["组内没有可对比的已抽取指标。"]


def _chart_guide_lines(
    series: list[SeriesCompare],
    plan: ReportPlan | None = None,
    facts: list[IndicatorFact] | None = None,
) -> list[str]:
    lines = [
        "PDF 附图规则：赛道报告以「组内横比」柱状图为主（同赛道、同指标、公司并列）；"
        "单公司多期仅两期用柱状两点对比，至少四期同口径才画折线，避免伪趋势。"
        "柱状图仅出现在 PDF 交付件；本 Markdown 只列将生成的图题。",
    ]
    if plan is not None and plan.skill_id == "semiconductor_industry_report":
        from ai_agent.report.pdf import charts_for_report

        charts = charts_for_report(plan, facts or [], series)
        if charts:
            lines.append("本期 PDF 将附以下组内横比柱状图：")
            for index, chart in enumerate(charts, start=1):
                lines.append(f"{index}. {chart.title}")
        else:
            lines.append("本期重点看设备／设计赛道的营收、毛利率、研发费用率横比图。")
        return lines
    comparable = sum(1 for item in series if len(item.points) >= 2)
    if comparable:
        lines.append(f"本期另有 {comparable} 组多期序列可作附录对照。")
    else:
        lines.append("本期无可用同口径多期序列时，不附伪趋势折线。")
    return lines


def _peer_lines(plan: ReportPlan, current: list[IndicatorFact]) -> list[str]:
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
        block = _peer_table_block(f"{name}（{_scope(scope)}）", rows, min_subjects=2)
        if lines and block:
            lines.append("")
        lines.extend(block)
    return lines or ["同一指标没有两家以上的公司，不做公司间对比。"]


def _gap_lines(plan: ReportPlan, current: list[IndicatorFact], gaps: list[str]) -> list[str]:
    # 范围／期间能推定的在阅读说明与指标表已交代；这里只列例外与真缺口。
    lines: list[str] = []
    parents = [fact.name for fact in current if fact.scope == "parent"]
    if parents:
        lines.append(f"母公司口径（例外）：{'、'.join(sorted(set(parents)))}。")
    inferred = [fact for fact in current if getattr(fact, "scope_inferred", False)]
    if inferred and not parents:
        lines.append("范围：本期主表指标按季报模板推定为合并（详见阅读说明）。")
    bases = sorted({_period_label(fact) for fact in current if fact.basis})
    if bases:
        lines.append(f"期间口径：{'、'.join(bases)}（同比只用同口径同期）。")
    # 真缺口：缺指标／缺基期文件。把「口径不齐」类与「待入库」分开列。
    data_gaps = [gap for gap in gaps if _is_data_gap(gap)]
    other_gaps = [gap for gap in gaps if not _is_data_gap(gap)]
    if data_gaps:
        lines.append("数据缺口（待入库或未抽到）：")
        lines.extend(data_gaps)
    if other_gaps:
        lines.append("其他说明：")
        lines.extend(other_gaps)
    return lines or ["没有缺失项。"]


def _is_data_gap(gap: str) -> bool:
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


def _has_indicator(
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


def _gaps(
    plan: ReportPlan, facts: list[IndicatorFact], series: list[SeriesCompare]
) -> list[str]:
    gaps = _coalesce_cumulative_base_gaps(list(plan.gaps))
    period_facts = [fact for fact in facts if fact.period == plan.period]
    # 单公司报告缺口句省略主语，全称由标题承担
    subjects = {fact.subject for fact in facts} | {item.subject for item in series}
    single = _sole_subject(plan, subjects)
    company_type = getattr(plan, "company_type", "") or industry_of(plan.subjects)
    # 同指标多公司都缺：概括一行，避免「没有抽到」刷屏。
    missing_by_indicator: dict[str, list[str]] = {}
    for subject in plan.subjects:
        for name in plan.indicators:
            # 类型排除项不报假缺失（银行毛利率、Fabless 净息差等）。
            if is_excluded(company_type, name):
                continue
            if not _has_indicator(period_facts, subject, name):
                missing_by_indicator.setdefault(name, []).append(subject)
    for name, cos in missing_by_indicator.items():
        gaps.append(_format_missing_indicator_gap(cos, plan.period, name, single=bool(single)))
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
        gaps.append(_format_series_gap_cluster(hits, reason, single=bool(single)))
    return gaps


def _coalesce_cumulative_base_gaps(gaps: list[str]) -> list[str]:
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


def _unique_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _format_missing_indicator_gap(
    companies: list[str], period: str, name: str, *, single: bool
) -> str:
    cos = _unique_preserve(companies)
    if single or len(cos) <= 1:
        prefix = "" if single else f"{cos[0]} "
        return f"{prefix}没有抽到 {period} {name}"
    return f"{len(cos)} 家公司（{'、'.join(cos)}）没有抽到 {period} {name}"


def _format_series_gap_cluster(
    hits: list[tuple[str, str]], reason: str, *, single: bool
) -> str:
    """同一原因只写一次：多家／多指标概括；单条例外保持「公司+指标：原因」。"""
    subjects = _unique_preserve([subject for subject, _ in hits])
    names = _unique_preserve([name for _, name in hits])
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


def _cite(fact: IndicatorFact) -> str:
    return _cite_evidence(fact.evidence)


def _cite_evidence(evidence) -> str:
    if evidence is None:
        return "出处不明"
    if evidence.page is None:
        where = evidence.filename or "出处不明"
    elif evidence.page_end and evidence.page_end != evidence.page:
        where = f"{evidence.filename} 第{evidence.page}–{evidence.page_end}页"
    else:
        where = f"{evidence.filename} 第{evidence.page}页"
    if evidence.section:
        where = f"{where} {evidence.section}"
    anchor = citation_anchor(evidence.file_id)
    if anchor:
        where = f"{where} {anchor}"
    return where


def _scope(scope: str, *, inferred: bool = False) -> str:
    # 「合并／母公司」范围；能推定的写「按季报模板推定」，不写「未标明」。
    if scope in _SCOPE:
        return scope_label(scope, inferred=inferred)
    return _UNKNOWN_SCOPE


def _fact_scope_label(fact: IndicatorFact) -> str:
    """展示用范围：抽取已推定则沿用；空 scope 时按章节／列名再推定一次。"""
    scope = fact.scope or ""
    inferred = bool(getattr(fact, "scope_inferred", False))
    if not scope:
        evidence = fact.evidence
        scope, inferred = infer_scope(
            scope="",
            section=getattr(evidence, "section", "") or "",
            column=fact.column or "",
            quote=getattr(evidence, "quote", "") or "",
        )
    return scope_label(scope, inferred=inferred)


def _column(fact: IndicatorFact) -> str:
    """数值取自哪一列（本报告期 / 年初至报告期末），写进正文防止口径误读。"""
    return fact.column.strip() or "未标明取数列"


def _markdown(
    title: str, sections: list[tuple[str, list[str]]], note: tuple[str, ...] = ()
) -> str:
    """段落之间空一行，但表格的连续行必须挨着，否则 Markdown 表格会散架。

    顶层 join 会把每个元素之间插入空行，所以表格行先合并成一块再交给它。
    HTML 路径（`_html_block`）本来就是按连续 "|" 行拼 <table>，不受影响。
    """
    parts = [f"# {title}"]
    if note:
        parts.append("\n".join(f"> {line}" for line in note))
    for heading, lines in sections:
        parts.append(f"## {heading}")
        table: list[str] = []
        for line in lines:
            if line.startswith("|"):
                table.append(line)
                continue
            if table:
                parts.append("\n".join(table))
                table.clear()
            parts.append(line)
        if table:
            parts.append("\n".join(table))
    return "\n\n".join(parts).strip() + "\n"


def _html(
    title: str, sections: list[tuple[str, list[str]]], note: tuple[str, ...] = ()
) -> str:
    body = [f"<h1>{html.escape(title)}</h1>"]
    if note:
        inner = "".join(f"<p>{html.escape(line)}</p>" for line in note)
        body.append(f"<blockquote>{inner}</blockquote>")
    for heading, lines in sections:
        body.append("<section>")
        body.append(f"<h2>{html.escape(heading)}</h2>")
        body.append(_html_block(lines))
        body.append("</section>")
    inner = "\n".join(body)
    return (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        f"<title>{html.escape(title)}</title>\n</head>\n<body>\n{inner}\n</body>\n</html>\n"
    )


# 千分位长数字：包进 .num 禁止 PDF 中途断行。
_LONG_NUMBER = re.compile(r"[+\-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?")
_NUMERIC_CELL = re.compile(r"^[+\-]?[\d,]+(?:\.\d+)?(?:%|元|次|亿元|万元|百万元)?$")


def _html_block(lines: list[str]) -> str:
    chunks: list[str] = []
    buffer: list[str] = []

    def flush_table() -> None:
        if not buffer:
            return
        rows = [line for line in buffer if not _is_rule(line)]
        head, *rest = rows
        rendered = ["<table>", "<thead><tr>", *_th(_cells(head)), "</tr></thead>", "<tbody>"]
        for row in rest:
            rendered.append("<tr>")
            rendered.extend(_td(_cells(row)))
            rendered.append("</tr>")
        rendered.append("</tbody></table>")
        chunks.append("\n".join(rendered))
        buffer.clear()

    for line in lines:
        if line.startswith("|"):
            buffer.append(line)
            continue
        flush_table()
        chunks.append(f"<p>{_wrap_long_numbers(html.escape(line))}</p>")
    flush_table()
    return "\n".join(chunks)


def _is_rule(line: str) -> bool:
    return set(line.replace("|", "").replace(" ", "")) <= {"-", ":"}


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _th(cells: list[str]) -> list[str]:
    return [f"<th>{html.escape(cell)}</th>" for cell in cells]


def _td(cells: list[str]) -> list[str]:
    out: list[str] = []
    for cell in cells:
        escaped = html.escape(cell)
        if _NUMERIC_CELL.match(cell.replace(" ", "")):
            out.append(f'<td><span class="num">{escaped}</span></td>')
        else:
            out.append(f"<td>{_wrap_long_numbers(escaped)}</td>")
    return out


def _wrap_long_numbers(escaped: str) -> str:
    """已 HTML escape 的文本里，给千分位数字加 nowrap 包裹。"""
    return _LONG_NUMBER.sub(lambda match: f'<span class="num">{match.group(0)}</span>', escaped)
