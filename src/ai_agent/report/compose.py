"""按大纲写成研报。正文里的数字只来自抽取结果和计算工具。

不调用模型，因此不会补一个大纲里有、但表格里没抽到的数。
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

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
from ai_agent.report.extract import IndicatorFact, citation_anchor
from ai_agent.report.parse import OutlineItem, ReportPlan
from ai_agent.report.supplemental import SupplementalResult

_SCOPE = {"consolidated": "合并", "parent": "母公司"}
_UNKNOWN_SCOPE = "范围未标明"
# 表格里未标明的范围用破折号，不逐行重复「范围未标明」；含义在阅读说明里一次讲清。
_UNKNOWN_SCOPE_CELL = "—"

# 开头统一声明。刻意不写「均为合并报表口径／年初至报告期末累计值」：
# 库里 scope 多为空，且期末余额、时点比率也不是累计值，那样写等于编造口径。
# 文案里不出现阿拉伯数字，避免被 validate 的数字扫描当成未核对数值。
_READING_NOTE = (
    "阅读说明：本文数值均按财报原文披露值引用，不自行折算单位、不由累计值倒推单季；"
    "期间口径、合并／母公司范围与取数列逐项列在「主要指标与出处」表，不以统一口径概括。",
    "表中「范围」记为「—」表示原文未注明合并或母公司范围，不代为推断；"
    "期末时点余额与期间比率不按累计值理解。",
    "每项数值的页码、章节与文件出处集中在表格「出处」列；方括号内的短标识为"
    "知识库文件标识前缀，完整标识与原文片段见溯源附件。",
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
) -> ReportDocument:
    """按计划里的大纲逐节填写。某一节没有数据时只写缺失，不留空头结论。"""
    supplemental = views or SupplementalResult()
    series = compare_measures([_measure(fact) for fact in facts])
    gaps = _gaps(plan, facts, series)
    # 结论句已不再自带出处，出处全靠主要指标表承载。大纲若缺这一节（外部
    # 构造的 ReportPlan 可能没有），补上，避免产出一份全文无出处的报告。
    outline = list(plan.outline)
    if outline and not any(item.title == "主要指标与出处" for item in outline):
        outline.append(OutlineItem("主要指标与出处", ""))
    sections = [
        (item.title, _section(item.title, plan, facts, series, gaps, supplemental))
        for item in outline
    ]
    if not sections:
        sections = [("正文", ["没有大纲，未写正文。"])]
    # 无效请求不该套一份财报阅读说明。
    note = _READING_NOTE if outline else ()
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


def _section(
    title: str,
    plan: ReportPlan,
    facts: list[IndicatorFact],
    series: list[SeriesCompare],
    gaps: list[str],
    views: SupplementalResult,
) -> list[str]:
    current = [fact for fact in facts if fact.period == plan.period]
    if title in {"本期结论", "覆盖范围"}:
        return _lead(plan, current, series)
    if title == "主要指标与出处":
        return _indicator_lines(plan, current)
    if title == "同比与环比":
        return _change_lines(plan, series)
    if title == "单季节奏":
        return _quarter_lines(plan, views)
    if title == "较上年末变动":
        return _year_end_lines(views)
    if title == "数据观察":
        return _observation_lines(plan, current, views)
    if title == "趋势":
        return _trend_lines(series, plan)
    if title == "公司间对比":
        return _peer_lines(plan, current)
    if title in {"缺失说明", "缺失与口径"}:
        return _gap_lines(plan, current, gaps)
    return ["这一节没有对应的数据，未写。"]


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
    plan: ReportPlan, current: list[IndicatorFact], series: list[SeriesCompare]
) -> list[str]:
    if plan.kind == "industry":
        names = "、".join(plan.subjects) if plan.subjects else "没有公司"
        line = f"{plan.industry}行业 {plan.period} 覆盖：{names}。"
        return [line, *_indicator_sentences(plan, current, series)]
    if not current:
        return [f"{plan.period} 没有抽到指标。"]
    return _indicator_sentences(plan, current, series)


def _indicator_sentences(
    plan: ReportPlan, facts: list[IndicatorFact], series: list[SeriesCompare]
) -> list[str]:
    """结论句只写「是什么 + 变化多少」。

    口径、范围、取数列、页码、章节、文件出处全部交给主要指标表和阅读说明；
    净息差年化之类的特别注记挂在表格下方，不再逐条塞进句子。数值保持
    原文的值与单位，不换算、不新增。单公司报告省略主语，全称由标题承担。
    """
    rates: dict[tuple[str, str, str], str] = {}
    for item in series:
        for period, change in item.yoy.items():
            # 推导值是否放行由 validate 判定，结论句不引入门禁未放行的数字。
            if change.source == "derived" or not change.rate:
                continue
            rates[(item.subject, item.name, period)] = change.rate
    subjects = {fact.subject for fact in facts}
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in facts:
        key = (fact.subject, fact.name, fact.scope)
        if key in seen:
            continue
        seen.add(key)
        # 只有累计/单季金额会被误读成别的期间，前缀必须留在主语里。
        # 时点比率与期末余额不会和累计混淆，硬加「时点／期间比率」反而像模板；
        # 它们的口径由阅读说明和主要指标表承担。
        period_text = "" if fact.basis in (POINT, STOCK) else _period_label(fact)
        rate = rates.get((fact.subject, fact.name, fact.period), "")
        change_text = f"，同比 {rate}" if rate else ""
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
        # 范围未知时只留破折号，含义已在开头阅读说明里统一声明。
        scope = _SCOPE.get(fact.scope, _UNKNOWN_SCOPE_CELL)
        subject_cell = f"{fact.subject} | " if with_subject else ""
        lines.append(
            f"| {subject_cell}{fact.name} | {fact.value}{fact.unit} | "
            f"{_period_label(fact)}／{scope}／{_column(fact)} | {_cite(fact)} |"
        )
    # 个别指标的特别注记（如净息差未注明年化）跟在表格下面当脚注，
    # 不再重复写进每条结论句。
    notes = sorted({note for fact in rows if (note := disclosure_note(fact.name))})
    lines.extend(f"注：{note}。" for note in notes)
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


def _year_end_lines(views: SupplementalResult) -> list[str]:
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
    return lines


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
    if stocks:
        lines.append(
            f"{'、'.join(stocks)}为期末时点余额，不是累计发生额，不与前三季度累计值相减。"
        )
    lines.append(
        "以上只并列原文披露的数字关系，不构成因果归因。季报未披露单季净息差拆分"
        "所需的平均生息资产，也未披露拨备前利润（PPOP）的统一口径，两项均不推算。"
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


def _change_lines(plan: ReportPlan, series: list[SeriesCompare]) -> list[str]:
    lines: list[str] = []
    subjects = {item.subject for item in series}
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
            lines.append(_change_sentence(prefix, item.name, "同比", yoy.base_period, yoy.delta, yoy.rate, note))
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
        lines.append(f"{name}（{_scope(scope)}）")
        lines.append("| 公司 | 数值 | 出处 |")
        lines.append("| --- | --- | --- |")
        for row in rows:
            lines.append(f"| {row.subject} | {row.value}{row.unit} | {_cite(row)} |")
    return lines or ["同一指标没有两家以上的公司，不做公司间对比。"]


def _gap_lines(plan: ReportPlan, current: list[IndicatorFact], gaps: list[str]) -> list[str]:
    known = sorted({_SCOPE[scope] for scope in (f.scope for f in current) if scope in _SCOPE})
    lines = [
        f"合并／母公司范围：{'、'.join(known) if known else '原文均未注明'}"
        "（逐项见主要指标表「范围」列，未注明的不代为推断）。"
    ]
    bases = sorted({_period_label(fact) for fact in current if fact.basis})
    if bases:
        lines.append(f"期间口径：{'、'.join(bases)}（同比只用同口径同期，跨期数不同不相减）。")
    columns = sorted({_column(fact) for fact in current})
    if columns:
        lines.append(f"取数列：{'、'.join(columns)}（单季与累计数不可混用）。")
    # 个别指标的特别注记已作为脚注挂在主要指标表下方，这里不再重复。
    lines.extend(gaps)
    return lines or ["没有缺失项。"]


def _gaps(
    plan: ReportPlan, facts: list[IndicatorFact], series: list[SeriesCompare]
) -> list[str]:
    gaps = list(plan.gaps)
    have = {(fact.subject, fact.name) for fact in facts if fact.period == plan.period}
    # 单公司报告缺口句省略主语，全称由标题承担
    subjects = {fact.subject for fact in facts} | {item.subject for item in series}
    single = _sole_subject(plan, subjects)
    for subject in plan.subjects:
        for name in plan.indicators:
            if (subject, name) not in have:
                prefix = "" if single else f"{subject} "
                gaps.append(f"{prefix}没有抽到 {plan.period} {name}")
    # 「不可比」和「时点存量不跨期」是逐指标产生的同一句话，21 个指标会刷出
    # 21 行一样的缺口，把真正的缺失项淹掉。合并成一行计数。
    repeated: dict[str, int] = {}
    for item in series:
        for gap in item.gaps:
            if gap in {NOT_COMPARABLE_NOTE, STOCK_NOTE}:
                repeated[gap] = repeated.get(gap, 0) + 1
                continue
            prefix = "" if single else item.subject
            text = f"{prefix}{item.name}：{gap}"
            if text not in gaps:
                gaps.append(text)
    for note, count in repeated.items():
        gaps.append(f"{count} 项指标：{note}")
    return gaps


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


def _scope(scope: str) -> str:
    # 这里说的是「合并／母公司」范围，不是期间口径，别再用「口径」两个字
    # 和「前三季度累计」混在一起。
    return _SCOPE.get(scope, _UNKNOWN_SCOPE)


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
        chunks.append(f"<p>{html.escape(line)}</p>")
    flush_table()
    return "\n".join(chunks)


def _is_rule(line: str) -> bool:
    return set(line.replace("|", "").replace(" ", "")) <= {"-", ":"}


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _th(cells: list[str]) -> list[str]:
    return [f"<th>{html.escape(cell)}</th>" for cell in cells]


def _td(cells: list[str]) -> list[str]:
    return [f"<td>{html.escape(cell)}</td>" for cell in cells]
