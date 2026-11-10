"""深度分析前：近 N 年年报自动补齐 + 主指标抽数写入 deep_facts。

与 ``ensure_filings``（本期/同比季报）职责分离：本模块只服务深度台账纵向数据，
不解析附注、不拉审计/问询。失败只记 notes，不阻断深度骨架。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb import cninfo
from ai_agent.kb.fetch_reports import fetch_missing_reports, wait_until_ready
from ai_agent.report.deep.sources import (
    DEFAULT_YEARS,
    DOC_CHAPTERS,
    classify_doc,
    default_facts_dir,
    window_years,
)
from ai_agent.report.extract import IndicatorFact, _files_for, extract_indicators
from ai_agent.report.parse import ReportPlan
from ai_agent.report.skills import get_skill
from ai_agent.report.progress import (
    msg_ensure_deep_annuals,
    msg_extract_deep_facts,
    msg_deep_facts_ready,
)
from ai_agent.tools.progress import emit_progress

logger = get_logger(__name__)

_SAFE_NAME = re.compile(r'[\\/:*?"<>|\s]+')
_DEFAULT_INDICATORS = (
    "营业收入",
    "营业成本",
    "毛利率",
    "归属于母公司股东的净利润",
    "经营活动产生的现金流量净额",
    "资产总计",
    "负债合计",
    "所有者权益合计",
)


@dataclass
class EnsureDeepAnnualsResult:
    notes: list[str] = field(default_factory=list)
    facts_path: Path | None = None
    fact_count: int = 0
    fetched_file_ids: list[str] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)
    skipped_extract: int = 0


FetchFn = Callable[..., object]
ExtractFn = Callable[[list[str], str, list[str]], object]
FilesForFn = Callable[[list[str], str], list[dict]]
WaitFn = Callable[[Sequence[str]], list[str]]


def annual_periods(period: str, years: int = DEFAULT_YEARS) -> list[str]:
    """以报告期年份为截止，生成近 N 个年报期（``YYYY``，最近年在前）。

    报告期无法识别年份时返回空列表（不做「当前年」兜底），避免误拉无关年报。
    """
    start, end, assumed = window_years(period, years, fallback_now=False)
    if assumed or not end:
        return []
    return [str(year) for year in range(end, start - 1, -1)]


def deep_full_size_limit_bytes() -> int:
    return max(int(float(settings.cninfo_deep_full_max_mb) * 1024 * 1024), 1)


def ensure_deep_annuals(
    plan: ReportPlan,
    *,
    years: int = DEFAULT_YEARS,
    facts_dir: Path | str | None = None,
    request_text: str = "",
    indicators: Sequence[str] | None = None,
    fetch_fn: FetchFn | None = None,
    extract_fn: ExtractFn | None = None,
    files_for_fn: FilesForFn | None = None,
    wait_fn: WaitFn | None = None,
    enable_fetch: bool | None = None,
) -> EnsureDeepAnnualsResult:
    """拉近 N 年年报（缺才拉）→ 等 ready → 抽主指标 → 写入 deep_facts JSON。"""
    result = EnsureDeepAnnualsResult()
    periods = annual_periods(str(getattr(plan, "period", "") or ""), years)
    result.periods = list(periods)
    if not periods:
        result.notes.append("深度年报补齐跳过：需求未识别报告期年份")
        return result

    out_dir = Path(facts_dir) if facts_dir is not None else default_facts_dir()
    pairs = _subject_code_pairs(plan, request_text)
    if not pairs:
        result.notes.append("深度年报补齐跳过：未能解析股票代码")
        return result

    wanted = _indicator_names(plan, indicators)
    do_fetch = (
        settings.enable_cninfo_fetch if enable_fetch is None else bool(enable_fetch)
    )
    fetch = fetch_fn or fetch_missing_reports
    extract = extract_fn or extract_indicators
    files_for = files_for_fn or _files_for
    wait = wait_fn or wait_until_ready
    max_full = deep_full_size_limit_bytes()
    limit = max(int(settings.cninfo_auto_fetch_max), 1)

    emit_progress(msg_ensure_deep_annuals(years=years))
    wait_ids: list[str] = []
    fetch_slots = 0
    summary_hits = 0
    hit_fetch_cap = False

    # 最近年优先；已有文件一律纳入等待，仅「缺文件去拉」受名额限制。
    for period in periods:
        for subject, code in pairs:
            existing = (
                files_for([subject], period)
                or files_for([code], period)
                or _files_for_stock_name(files_for, code, period)
            )
            if existing:
                for row in existing:
                    file_id = str(row.get("file_id") or "")
                    if file_id:
                        wait_ids.append(file_id)
                continue
            if not do_fetch:
                result.notes.append(
                    f"深度年报未入库且自动拉取已关闭：{subject} {period}"
                )
                continue
            if fetch_slots >= limit:
                hit_fetch_cap = True
                continue
            fetch_slots += 1
            try:
                fetched = fetch(
                    code,
                    period,
                    max_files=1,
                    max_full_bytes=max_full,
                )
            except Exception as exc:
                logger.exception("深度年报补齐失败 subject=%s period=%s", subject, period)
                result.notes.append(f"深度年报补齐 {subject} {period} 失败：{exc}")
                continue
            errors = list(getattr(fetched, "errors", None) or [])
            skipped = list(getattr(fetched, "skipped", None) or [])
            files = list(getattr(fetched, "files", None) or [])
            if errors:
                result.notes.extend(
                    f"深度年报补齐 {subject} {period}：{item}" for item in errors
                )
            if skipped and not files:
                result.notes.extend(
                    f"深度年报补齐 {subject} {period}：{item}" for item in skipped
                )
            for item in files:
                file_id = str(getattr(item, "file_id", "") or "")
                title = str(getattr(item, "title", "") or "")
                filename = str(getattr(item, "filename", "") or "")
                if file_id:
                    result.fetched_file_ids.append(file_id)
                    wait_ids.append(file_id)
                if "摘要" in title or "摘要" in filename:
                    summary_hits += 1
                    result.notes.append(
                        f"{subject} {period}《{filename or title}》为摘要降级，"
                        "附注级仍缺，仅主表指标可入深度台账"
                    )
                elif file_id:
                    result.notes.append(
                        f"已从巨潮补齐年报 {subject} {period}《{filename}》"
                        f"（file_id={file_id}）"
                    )
    if hit_fetch_cap:
        result.notes.append(
            f"深度年报补齐达到单次上限 {limit}，其余缺失期次未拉取；"
            "库中已有文件仍会抽数"
        )

    # 去重保序后等待入库
    seen: set[str] = set()
    ordered: list[str] = []
    for file_id in wait_ids:
        if not file_id or file_id in seen:
            continue
        seen.add(file_id)
        ordered.append(file_id)
    if ordered:
        not_ready = wait(ordered)
        if not_ready:
            result.notes.append(
                "深度年报入库等待超时，下列 file_id 尚未 ready，抽数可能仍缺："
                + "、".join(not_ready[:8])
                + (f" 等共 {len(not_ready)} 份" if len(not_ready) > 8 else "")
            )

    emit_progress(msg_extract_deep_facts())
    rows: list[dict] = []
    skipped = 0
    for period in periods:
        for subject, code in pairs:
            names = [subject, code]
            try:
                stock = cninfo.lookup_stock(code)
            except Exception:
                stock = None
            if stock and stock.name and stock.name not in names:
                names.append(stock.name)
            try:
                extracted = extract(names, period, list(wanted))
            except Exception as exc:
                logger.exception("深度年报抽数失败 subject=%s period=%s", subject, period)
                result.notes.append(f"深度年报抽数 {subject} {period} 失败：{exc}")
                continue
            facts = list(getattr(extracted, "facts", None) or [])
            for fact in facts:
                row = indicator_fact_to_deep_row(fact)
                if row is None:
                    skipped += 1
                    continue
                rows.append(row)

    result.skipped_extract = skipped
    if skipped:
        result.notes.append(
            f"深度年报抽数跳过 {skipped} 条（缺页码/原文片段，守零幻觉门禁）"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    label = _stem_label(plan, pairs)
    window = f"{periods[-1]}-{periods[0]}" if periods else "unknown"
    path = out_dir / f"auto_annual_{label}_{window}.json"
    payload = {
        "meta": {
            "schema": "deep-facts/v1",
            "source": "ensure_deep_annuals",
            "years": years,
            "periods": periods,
            "fact_count": len(rows),
            "note": (
                "近 N 年年报主指标自动抽取（零 LLM）；"
                "附注/审计/问询仍需人工或后续通道"
            ),
        },
        "facts": rows,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result.facts_path = path
    result.fact_count = len(rows)
    result.notes.append(
        f"深度年报事实清单已写入 {path.name}（{len(rows)} 条；"
        f"窗口 {window}；摘要降级 {summary_hits} 份）"
    )
    emit_progress(msg_deep_facts_ready(fact_count=len(rows)))
    return result


def indicator_fact_to_deep_row(fact: IndicatorFact) -> dict | None:
    """``IndicatorFact`` → deep-facts 行；缺页码或原文片段则丢弃。"""
    evidence = fact.evidence
    page = evidence.page
    quote = (evidence.quote or "").strip()
    if not page or not quote:
        return None
    value = (fact.value or "").strip()
    if not value:
        return None
    unit = (fact.unit or "").strip()
    if unit and unit not in value:
        value = f"{value}{unit}"
    file_id = (evidence.file_id or "").strip()
    filename = (evidence.filename or "").strip() or "未知文件"
    source_file = f"{filename}（{file_id}）" if file_id else filename
    section = (evidence.section or "").strip()
    location = f"第{page}页"
    if evidence.page_end and evidence.page_end != page:
        location = f"第{page}-{evidence.page_end}页"
    if section:
        location = f"{location} {section}"
    doc_type = classify_doc(filename)
    if doc_type == "other":
        doc_type = "annual"
    return {
        "item": (fact.name or "")[:120],
        "value": value[:120],
        "source_file": source_file[:200],
        "doc_type": doc_type,
        "level": "L1",
        "location": location[:120],
        "quote": quote[:400],
        "subject": (fact.subject or "")[:80],
        "period": (fact.period or "")[:40],
        "acquisition": "知识库年报原文／extract_indicators 确定性抽取（零 LLM）",
        "chapters": list(DOC_CHAPTERS.get(doc_type, DOC_CHAPTERS["annual"])),
    }


def _files_for_stock_name(
    files_for: FilesForFn, code: str, period: str
) -> list[dict]:
    try:
        stock = cninfo.lookup_stock(code)
    except Exception:
        return []
    if not stock or not stock.name:
        return []
    return files_for([stock.name], period)


def _subject_code_pairs(
    plan: ReportPlan, request_text: str
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    skill = get_skill(plan.skill_id) if plan.skill_id else None
    if skill and skill.companies:
        for name, code, _segment, _aka in skill.companies:
            if plan.subjects and name not in plan.subjects:
                if not any(
                    name in subject or subject in name for subject in plan.subjects
                ):
                    continue
            pairs.append((name, code))
    if not pairs:
        for subject in plan.subjects:
            stock = cninfo.resolve_code_by_name(subject)
            if stock is None:
                continue
            pairs.append((subject, stock.code))
    if not pairs and request_text:
        for stock in cninfo.resolve_codes_in_text(request_text):
            pairs.append((stock.name or stock.code, stock.code))
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for subject, code in pairs:
        key = (subject, code)
        if key in seen:
            continue
        seen.add(key)
        unique.append((subject, code))
    return unique


def _indicator_names(
    plan: ReportPlan, indicators: Sequence[str] | None
) -> list[str]:
    if indicators:
        return [item.strip() for item in indicators if item and str(item).strip()]
    planned = [
        item.strip()
        for item in (getattr(plan, "indicators", None) or [])
        if item and str(item).strip()
    ]
    if planned:
        return planned
    return list(_DEFAULT_INDICATORS)


def _stem_label(plan: ReportPlan, pairs: list[tuple[str, str]]) -> str:
    raw = ""
    if getattr(plan, "kind", "") == "industry":
        raw = str(getattr(plan, "industry", "") or "")
    if not raw and pairs:
        raw = pairs[0][0]
    if not raw and plan.subjects:
        raw = plan.subjects[0]
    cleaned = _SAFE_NAME.sub("_", (raw or "target").strip())[:40].strip("_")
    return cleaned or "target"
