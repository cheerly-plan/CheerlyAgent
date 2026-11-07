"""compose 共用原语：阅读说明、出处、范围、单元格。"""

from __future__ import annotations

from ai_agent.kb.ingest.meta import name_in_text
from ai_agent.report import sections as report_sections
from ai_agent.report.basis import label as basis_label
from ai_agent.report.extract import IndicatorFact, citation_anchor
from ai_agent.report.meta import infer_scope, scope_label
from ai_agent.report.parse import ReportPlan
from ai_agent.report.skills import get_skill, is_segment_skill
from ai_agent.report import registry

SCOPE = {"consolidated": "合并", "parent": "母公司"}
UNKNOWN_SCOPE = "—"

# 开头统一声明一次：能按模板推定的不写「找不到」；文案避免阿拉伯数字（校验器）。
READING_NOTE = (
    "阅读说明：金额默认统一为元；若由千元／万元／百万元折算，指标表会注明原单位。",
    "范围：主要财务数据等主表按季报模板推定为合并，母公司口径单独标注；"
    "期间按列名区分——本报告期为单季、年初至报告期末为累计、期末为时点余额。",
    "基期文件未入库属于数据缺口，与口径推定无关，见「缺失与口径」。"
    "方括号短标识为知识库文件前缀，完整出处见溯源附件。",
)

READING_NOTE_SEGMENT_TAIL = (
    "金额默认统一为元，主表范围按季报模板推定为合并，期间按列名区分单季／累计／时点。",
    "三级归因边界：已抽取的数据事实可写；原文经营说明须标出处后方可引用；"
    "无出处的第三方推断一律不写。完整标识与原文片段见溯源附件。",
)

NEXT_PERIOD_NOTE = (
    "景气周期依赖外部数据源，本期流水线未接入，不推断周期位置。",
    "下一期预告需明确：数据源清单、入库表结构、更新频率；接入前本报告只做公司财报赛道分组对比。",
)


def is_segment_report(plan: ReportPlan) -> bool:
    return is_segment_skill(plan.skill_id) or report_sections.outline_has(
        plan.outline, "segment_roster", "segment_peer"
    )


def sole_subject(plan: ReportPlan, subjects: set[str]) -> bool:
    """单公司报告正文省略主语，全称由标题承担。"""
    return plan.kind != "industry" and len(subjects) <= 1


def subject_prefix(plan: ReportPlan, subject: str, subjects: set[str]) -> str:
    """句首主语：单公司报告为空串，多主体报告用全称。"""
    return "" if sole_subject(plan, subjects) else subject


def segment_reading_note(plan: ReportPlan) -> tuple[str, ...]:
    industry = (plan.industry or "").strip() or "行业"
    head = f"阅读说明：本文为{industry}代表公司财报对比分析（赛道分组版）；"
    return (head + READING_NOTE_SEGMENT_TAIL[0],) + READING_NOTE_SEGMENT_TAIL[1:]


def roster_rename_note_lines(plan: ReportPlan) -> list[str]:
    """花名册「披露又名」→ 阅读说明改名备注；无又名则空。"""
    skill = get_skill(plan.skill_id) if plan.skill_id else None
    if skill is None:
        return []
    notes = skill.rename_notes()
    if not notes:
        return []
    return [f"名单备注：{'；'.join(notes)}。"]


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
    if indicator in money or (not unit and looks_like_money(fact.value)):
        return f"{fact.value}元"
    return str(fact.value)


def looks_like_money(value: str) -> bool:
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


def period_label(fact: IndicatorFact) -> str:
    """手工构造、没标口径的 fact 沿用旧文案，避免破坏纯计算调用。"""
    if not fact.basis:
        return fact.period
    return basis_label(fact.period, fact.basis, fact.coverage_months)


def cite(fact: IndicatorFact) -> str:
    return cite_evidence(fact.evidence)


def cite_evidence(evidence) -> str:
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


def scope_text(scope: str, *, inferred: bool = False) -> str:
    # 「合并／母公司」范围；能推定的写「按季报模板推定」，不写「未标明」。
    if scope in SCOPE:
        return scope_label(scope, inferred=inferred)
    return UNKNOWN_SCOPE


def fact_scope_label(fact: IndicatorFact) -> str:
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


def column_label(fact: IndicatorFact) -> str:
    """数值取自哪一列（本报告期 / 年初至报告期末），写进正文防止口径误读。"""
    return fact.column.strip() or "未标明取数列"


