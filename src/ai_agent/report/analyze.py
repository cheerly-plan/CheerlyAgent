"""经营解读：受约束 LLM，只许用已抽取事实 + 季报原文。

数字路径（抽数 / 计算 / 结论表）仍是确定性代码。本模块只编排「经营解读」
一节；原文清洗见 analyze_evidence，结构梯队见 analyze_structure，
启发式名单见 config/analysis.yaml。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import complete_chat, lightweight_llm
from ai_agent.llm.json_utils import load_llm_json
from ai_agent.prompts.report_analysis import ANALYSIS_SYSTEM, render_analysis_user
from ai_agent.report.compare import SeriesCompare, compare_measures
from ai_agent.report.extract import IndicatorFact, citation_anchor
from ai_agent.report.parse import ReportPlan
from ai_agent.report.phrases import ANALYSIS_FALLBACK
from ai_agent.report.supplemental import SupplementalResult
from ai_agent.report.validate import _allowed, _number_key
from ai_agent.report.analyze_evidence import (
    EvidenceItem,
    collect_evidence_pack,
    extract_operating_narrative,
    is_form_chrome,
    is_statement_template,
    is_usable_operating_quote,
    iter_reason_snippets,
    plain_evidence_quote,
    safe_truncate,
)
from ai_agent.report.analyze_heuristics import operating_snippet_keys
from ai_agent.report.analyze_structure import (
    company_fact_priority,
    company_structure_line,
    segment_structure_lines,
    semiconductor_structure_lines,
)

# 兼容旧 import / 测试私有名
__all__ = (
    "ANALYSIS_FALLBACK",
    "AnalysisSection",
    "EvidenceItem",
    "analyze_operations",
    "collect_evidence_pack",
)

logger = get_logger(__name__)

_FALLBACK = ANALYSIS_FALLBACK
_TOPICS = frozenset({"nim", "profit", "asset_quality", "other"})
_CLAIM_KINDS = frozenset({"restatement", "comparison", "gap"})
_NUMBER = re.compile(r"-?\d[\d,，]*(?:[ \t\u00a0]\d{3})*(?:\.\d+)?%?")

# 测试兼容别名
_company_fact_priority = company_fact_priority
_segment_structure_lines = segment_structure_lines
_semiconductor_structure_lines = semiconductor_structure_lines
_extract_operating_narrative = extract_operating_narrative
_is_usable_operating_quote = is_usable_operating_quote


@dataclass
class AnalysisSection:
    lines: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    used_llm: bool = False
    # 正文推导出的赛道均值等，并入 facts 供 validate 白名单，不进指标宽表。
    derived_facts: list[IndicatorFact] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.lines


def analyze_operations(
    plan: ReportPlan,
    facts: list[IndicatorFact],
    series: list[SeriesCompare] | None = None,
    views: SupplementalResult | None = None,
    *,
    enabled: bool | None = None,
    complete=None,
    evidence: list[EvidenceItem] | None = None,
) -> AnalysisSection:
    """生成经营解读段落。enabled 默认读配置；complete / evidence 可注入以便单测。"""
    from ai_agent.report import sections as report_sections

    if not report_sections.outline_has(plan.outline, "analysis"):
        return AnalysisSection()

    current = [fact for fact in facts if fact.period == plan.period]
    if not current:
        return AnalysisSection(lines=[_FALLBACK], gaps=["本期没有可解读的抽取指标"])

    # 赛道分组报告：用已抽取数字写梯队与结构关系，不跑 LLM 复述/乱引模板条款。
    from ai_agent.report.skills import is_segment_skill
    from ai_agent.report.industry import structure_for_plan

    if is_segment_skill(plan.skill_id):
        lines, derived = segment_structure_lines(plan, current)
        if lines:
            return AnalysisSection(lines=lines, used_llm=False, derived_facts=derived)
        wanted = structure_for_plan(plan)
        gap = (
            f"缺少可排序的结构指标（{'、'.join(wanted)}）"
            if wanted
            else "行业档案未声明 structure，赛道解读从略"
        )
        return AnalysisSection(lines=[_FALLBACK], gaps=[gap])

    supplemental = views or SupplementalResult()
    computed = series if series is not None else compare_measures(
        [fact.to_measure() for fact in facts]
    )
    pack = evidence if evidence is not None else collect_evidence_pack(plan, current)
    by_id = {item.evidence_id: item for item in pack}
    allowed = _allowed(facts, computed, plan, supplemental)
    for item in pack:
        for match in _NUMBER.finditer(item.quote):
            key = _number_key(match.group(0))
            if key:
                allowed.add(key)

    use_llm = settings.report_analysis_enabled if enabled is None else enabled
    if not use_llm:
        lines = deterministic_lines(plan, current, pack)
        if lines:
            return AnalysisSection(
                lines=lines, gaps=["经营解读未启用模型，已用数字事实与原文引述"], used_llm=False
            )
        return AnalysisSection(lines=[_FALLBACK], gaps=["经营解读未启用"])

    user = render_analysis_user(
        subject_period=_subject_period(plan),
        facts=_render_facts(plan, current, computed, supplemental),
        evidence=_render_evidence(pack),
    )
    try:
        raw = (complete or _call_llm)(ANALYSIS_SYSTEM, user)
    except Exception:
        logger.exception("经营解读 LLM 调用失败，改用原文证据回退")
        lines = deterministic_lines(plan, current, pack)
        if lines:
            return AnalysisSection(
                lines=lines, gaps=["经营解读模型调用失败，已改用原文引述"], used_llm=False
            )
        return AnalysisSection(lines=[_FALLBACK], gaps=["经营解读模型调用失败"], used_llm=False)

    parsed = load_llm_json(raw, repair=True)
    if not isinstance(parsed, dict):
        lines = deterministic_lines(plan, current, pack)
        if lines:
            return AnalysisSection(
                lines=lines, gaps=["经营解读输出无法解析，已改用原文引述"], used_llm=True
            )
        return AnalysisSection(lines=[_FALLBACK], gaps=["经营解读输出无法解析"], used_llm=True)

    lines, gaps = _postprocess(parsed, by_id=by_id, allowed=allowed)
    if not lines:
        # 模型给了段落但全被门禁丢掉：保持从略，避免换一套文案绕过校验。
        # 仅在模型无可用结构时，才退回数字事实 + 原文引述。
        raw_paras = parsed.get("paragraphs")
        if isinstance(raw_paras, list) and raw_paras:
            gaps = list(gaps) or ["经营解读无合法段落"]
            return AnalysisSection(lines=[_FALLBACK], gaps=gaps, used_llm=True)
        fallback_lines = deterministic_lines(plan, current, pack)
        if fallback_lines:
            gaps = list(gaps) or ["经营解读无合法 LLM 段落"]
            gaps.append("已改用原文证据引述")
            return AnalysisSection(lines=fallback_lines, gaps=gaps, used_llm=True)
        gaps = list(gaps) or ["经营解读无合法段落"]
        return AnalysisSection(lines=[_FALLBACK], gaps=gaps, used_llm=True)
    return AnalysisSection(lines=lines, gaps=gaps, used_llm=True)


def deterministic_lines(
    plan: ReportPlan,
    facts: list[IndicatorFact],
    pack: list[EvidenceItem],
) -> list[str]:
    """无可靠 LLM 输出时：①并列已授权数字事实；②引述经营分析原文（带出处）。

    不写第三方推断。证据不足时返回空，由调用方落从略句。
    """
    from ai_agent.report.skills import is_segment_skill

    if is_segment_skill(plan.skill_id):
        lines, _derived = segment_structure_lines(plan, facts)
        return lines

    lines: list[str] = []
    keys = operating_snippet_keys()
    ban = (
        "被合并方",
        "合并日",
        "指引",
        "预期",
        "前瞻",
        "净利润为0",
        "净利润为 0",
        "是否经审计",
    )
    narrative_keep = ("量价", "存储", "出货", "毛利润增加", "经营业绩")
    used = 0
    seen_snippets: set[str] = set()
    for item in pack:
        raw = extract_operating_narrative(plain_evidence_quote(item.quote or ""))
        if len(raw) < 20 or not any(token in raw for token in keys):
            continue
        if any(token in raw for token in ban):
            continue
        if is_statement_template(raw) and not any(
            token in raw for token in narrative_keep
        ):
            continue
        for snippet in iter_reason_snippets(raw):
            snippet = extract_operating_narrative(snippet)
            if not snippet or not is_usable_operating_quote(snippet):
                continue
            if is_form_chrome(snippet):
                continue
            key = snippet[:40]
            if key in seen_snippets:
                continue
            seen_snippets.add(key)
            quote = safe_truncate(snippet, 120)
            cite = _format_citation([item])
            lines.append(f"原文披露：{quote}{cite}。")
            used += 1
            if used >= 2:
                break
        if used >= 2:
            break
    cited = "\n".join(lines)
    structure = company_structure_line(
        plan, facts, has_narrative=bool(lines), cited_text=cited
    )
    if structure:
        lines.append(structure)
    if not lines:
        priority = company_fact_priority(plan)
        ordered = sorted(
            facts,
            key=lambda fact: (
                priority.index(fact.name) if fact.name in priority else 99,
                fact.subject,
            ),
        )
        for fact in ordered[:4]:
            anchor = citation_anchor(fact.evidence.file_id)
            source = fact.evidence.filename or "出处"
            page = f"第{fact.evidence.page}页" if fact.evidence.page is not None else ""
            cite = " ".join(part for part in (page, anchor) if part)
            extra = f"（{cite}）" if cite else f"（{source}）"
            lines.append(
                f"{fact.subject}{plan.period}{fact.name}为 {fact.value}{fact.unit}{extra}。"
            )
    if not lines:
        return []
    lines.append("以上仅并列已授权数字与原文表述，不构成无出处的因果归因。")
    return lines


_deterministic_lines = deterministic_lines


def _call_llm(system: str, user: str) -> str:
    model, client = lightweight_llm()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return complete_chat(
            messages,
            model=model,
            client=client,
            temperature=0.1,
            max_tokens=1200,
            timeout=60,
            response_format={"type": "json_object"},
        )
    except LLMError:
        return complete_chat(
            messages,
            model=model,
            client=client,
            temperature=0.1,
            max_tokens=1200,
            timeout=60,
        )


def _subject_period(plan: ReportPlan) -> str:
    names = "、".join(plan.subjects) if plan.subjects else "（未识别主体）"
    return f"{names} / {plan.period or '（未识别报告期）'}"


def _render_facts(
    plan: ReportPlan,
    facts: list[IndicatorFact],
    series: list[SeriesCompare],
    views: SupplementalResult,
) -> str:
    rates: dict[tuple[str, str], str] = {}
    for item in series:
        change = item.yoy.get(plan.period)
        if change and change.source != "derived" and change.rate:
            rates[(item.subject, item.name)] = change.rate
    lines: list[str] = []
    for fact in facts:
        rate = rates.get((fact.subject, fact.name), "")
        extra = f"，同比 {rate}" if rate else ""
        lines.append(f"- {fact.name}：{fact.value}{fact.unit}{extra}")
    for item in views.year_ends:
        lines.append(
            f"- {item.name} 较上年末：{item.change}（期末值 {item.value}{item.unit}）"
        )
    for item in views.quarters:
        if item.quarter_yoy and item.cumulative_yoy:
            lines.append(
                f"- {item.name} 单季同比 {item.quarter_yoy} / 累计同比 {item.cumulative_yoy}"
            )
    return "\n".join(lines) if lines else "（无）"


def _render_evidence(items: list[EvidenceItem]) -> str:
    if not items:
        return "（未召回到经营分析原文；只能写 gap）"
    lines: list[str] = []
    for item in items:
        page = f"第{item.page}页" if item.page is not None else "页码未知"
        quote = " ".join(item.quote.split())
        if len(quote) > 280:
            quote = quote[:279] + "…"
        lines.append(
            f"- {item.evidence_id}｜{page}｜{item.section or '（无章节）'}｜{quote}"
        )
    return "\n".join(lines)


def _postprocess(
    payload: dict,
    *,
    by_id: dict[str, EvidenceItem],
    allowed: set[str],
) -> tuple[list[str], list[str]]:
    gaps = [
        str(item).strip()
        for item in (payload.get("gaps") or [])
        if str(item).strip()
    ]
    lines: list[str] = []
    raw_paragraphs = payload.get("paragraphs") or []
    if not isinstance(raw_paragraphs, list):
        return [], gaps + ["paragraphs 不是数组"]

    for raw in raw_paragraphs[:5]:
        if not isinstance(raw, dict):
            continue
        topic = str(raw.get("topic") or "").strip()
        text = str(raw.get("text") or "").strip()
        kind = str(raw.get("claim_kind") or "").strip()
        evidence_ids = [
            str(item).strip()
            for item in (raw.get("evidence_ids") or [])
            if str(item).strip()
        ]
        if topic not in _TOPICS or kind not in _CLAIM_KINDS or not text:
            continue
        if kind == "gap":
            if not any(token in text for token in ("未披露", "不推断", "无法判断", "从略")):
                gaps.append(f"gap 段未标明未披露：{text[:40]}")
                continue
            lines.append(text if text.endswith("。") else text + "。")
            continue
        if not evidence_ids or any(eid not in by_id for eid in evidence_ids):
            gaps.append(f"段落缺少有效证据，已丢弃：{text[:40]}")
            continue
        if not _numbers_allowed(text, allowed):
            gaps.append(f"段落含未授权数字，已丢弃：{text[:40]}")
            continue
        cite = _format_citation([by_id[eid] for eid in evidence_ids if eid in by_id])
        body = text.rstrip("。．.！？；;")
        lines.append(f"{body}{cite}。")
    return lines, gaps


def _numbers_allowed(text: str, allowed: set[str]) -> bool:
    for match in _NUMBER.finditer(text):
        raw = match.group(0)
        key = _number_key(raw)
        if not key:
            continue
        if re.fullmatch(r"-?\d{1,2}", key.rstrip("%")):
            continue
        if key in allowed:
            continue
        stripped = key.lstrip("+-").rstrip("%")
        if any(item.lstrip("+-").rstrip("%") == stripped for item in allowed):
            continue
        return False
    return True


def _format_citation(items: list[EvidenceItem]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for item in items:
        anchor = citation_anchor(item.file_id) if item.file_id else ""
        page = f"第{item.page}页" if item.page is not None else ""
        label = " ".join(part for part in (page, anchor) if part)
        if not label or label in seen:
            continue
        seen.add(label)
        parts.append(label)
    if not parts:
        return ""
    return "（" + "；".join(parts) + "）"
