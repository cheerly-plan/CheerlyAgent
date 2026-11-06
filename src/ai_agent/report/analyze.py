"""经营解读：受约束 LLM，只许用已抽取事实 + 季报原文。

数字路径（抽数 / 计算 / 结论表）仍是确定性代码。本模块只生成「经营解读」
一节；输出必须过数字白名单与证据 id 校验，失败则从略，不阻断整份报告。
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
from ai_agent.report.extract import IndicatorFact, citation_anchor, _as_uuid, _files_for
from ai_agent.report.parse import ReportPlan
from ai_agent.report.supplemental import SupplementalResult
from ai_agent.report.validate import _allowed, _measure, _number_key
from ai_agent.storage.store import store

logger = get_logger(__name__)

_FALLBACK = "经营解读：本期季报未提供可核对的归因表述，本节从略。"
_TOPICS = frozenset({"nim", "profit", "asset_quality", "other"})
_CLAIM_KINDS = frozenset({"restatement", "comparison", "gap"})
_NUMBER = re.compile(r"-?\d[\d,，]*(?:[ \t\u00a0]\d{3})*(?:\.\d+)?%?")
_EVIDENCE_KEYWORDS = (
    "经营情况简要分析",
    "净息差",
    "不良贷款",
    "拨备覆盖率",
    "营业收入",
    "净利润",
)


@dataclass
class EvidenceItem:
    evidence_id: str
    file_id: str
    filename: str
    page: int | None
    section: str
    quote: str


@dataclass
class AnalysisSection:
    lines: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    used_llm: bool = False

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
    if not (settings.report_analysis_enabled if enabled is None else enabled):
        return AnalysisSection(lines=[_FALLBACK])
    if not any(item.title == "经营解读" for item in plan.outline):
        return AnalysisSection()

    current = [fact for fact in facts if fact.period == plan.period]
    if not current:
        return AnalysisSection(lines=[_FALLBACK], gaps=["本期没有可解读的抽取指标"])

    supplemental = views or SupplementalResult()
    computed = series if series is not None else compare_measures(
        [_measure(fact) for fact in facts]
    )
    pack = evidence if evidence is not None else collect_evidence_pack(plan, current)
    by_id = {item.evidence_id: item for item in pack}
    allowed = _allowed(facts, computed, plan, supplemental)
    for item in pack:
        for match in _NUMBER.finditer(item.quote):
            key = _number_key(match.group(0))
            if key:
                allowed.add(key)

    user = render_analysis_user(
        subject_period=_subject_period(plan),
        facts=_render_facts(plan, current, computed, supplemental),
        evidence=_render_evidence(pack),
    )
    try:
        raw = (complete or _call_llm)(ANALYSIS_SYSTEM, user)
    except Exception:
        logger.exception("经营解读 LLM 调用失败，本节从略")
        return AnalysisSection(lines=[_FALLBACK], gaps=["经营解读模型调用失败"], used_llm=False)

    parsed = load_llm_json(raw, repair=True)
    if not isinstance(parsed, dict):
        return AnalysisSection(lines=[_FALLBACK], gaps=["经营解读输出无法解析"], used_llm=True)

    lines, gaps = _postprocess(parsed, by_id=by_id, allowed=allowed)
    if not lines:
        gaps = list(gaps) or ["经营解读无合法段落"]
        return AnalysisSection(lines=[_FALLBACK], gaps=gaps, used_llm=True)
    return AnalysisSection(lines=lines, gaps=gaps, used_llm=True)


def collect_evidence_pack(
    plan: ReportPlan, facts: list[IndicatorFact]
) -> list[EvidenceItem]:
    """KB 召回 + 本期事实出处，供 analyze / 测试共用。"""
    items = _load_kb_evidence(plan)
    seen = {(item.file_id, item.page, item.quote[:60]) for item in items}
    index = len(items)
    for fact in facts:
        if fact.period != plan.period:
            continue
        quote = " ".join((fact.evidence.quote or "").split())
        if len(quote) < 8:
            continue
        key = (fact.evidence.file_id, fact.evidence.page, quote[:60])
        if key in seen:
            continue
        seen.add(key)
        items.append(
            EvidenceItem(
                evidence_id=f"e{index}",
                file_id=fact.evidence.file_id,
                filename=fact.evidence.filename,
                page=fact.evidence.page,
                section=fact.evidence.section,
                quote=quote[:500],
            )
        )
        index += 1
        if len(items) >= 12:
            break
    return items


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
        # 本地小模型常不支持 response_format，降级为纯 prompt 约束
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


def _load_kb_evidence(plan: ReportPlan) -> list[EvidenceItem]:
    try:
        files = _files_for(plan.subjects, plan.period)
    except Exception:
        logger.exception("经营解读定位本期文件失败")
        return []
    file_ids = [_as_uuid(row.get("file_id")) for row in files]
    file_ids = [item for item in file_ids if item is not None]
    if not file_ids:
        return []
    try:
        rows = store.search_kb_chunks_keyword(
            list(_EVIDENCE_KEYWORDS),
            file_ids=file_ids,
            top_k=8,
            kinds=["text"],
        )
    except Exception:
        logger.exception("经营解读证据召回失败")
        return []
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        text = " ".join(str(row.get("text") or "").split())
        if len(text) < 20:
            continue
        file_id = str(row.get("file_id") or "")
        page = row.get("page")
        key = f"{file_id}:{page}:{text[:80]}"
        if key in seen:
            continue
        seen.add(key)
        page_int: int | None
        if isinstance(page, int):
            page_int = page
        elif str(page).isdigit():
            page_int = int(page)
        else:
            page_int = None
        items.append(
            EvidenceItem(
                evidence_id=f"e{index}",
                file_id=file_id,
                filename=str(row.get("filename") or ""),
                page=page_int,
                section=str(row.get("section") or ""),
                quote=text[:500],
            )
        )
        if len(items) >= 8:
            break
    return items


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
        # 锚点必须落在句号前，否则按句扫描归因时会把出处拆到下一句。
        body = text.rstrip("。．.！？；;")
        lines.append(f"{body}{cite}。")
    return lines, gaps


def _numbers_allowed(text: str, allowed: set[str]) -> bool:
    for match in _NUMBER.finditer(text):
        raw = match.group(0)
        key = _number_key(raw)
        if not key:
            continue
        # 章节号 / 短序号不拦
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
