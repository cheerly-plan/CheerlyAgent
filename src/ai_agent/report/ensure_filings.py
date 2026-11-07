"""研报生成前：缺原文则从巨潮补齐，再继续抽数。

对话里不提供「补文件」工具；用户只说生成研报，缺本期/同比基期时由流水线补。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb import cninfo
from ai_agent.kb.fetch_reports import fetch_missing_reports, wait_until_ready
from ai_agent.report.compare import cumulative_base_periods, yoy_period
from ai_agent.report.extract import _files_for
from ai_agent.report.parse import ReportPlan
from ai_agent.report.skills import get_skill
from ai_agent.storage.store import store

logger = get_logger(__name__)


@dataclass
class EnsureFilingsResult:
    notes: list[str] = field(default_factory=list)
    fetched_file_ids: list[str] = field(default_factory=list)
    attempted: bool = False

    @property
    def fetched_any(self) -> bool:
        return bool(self.fetched_file_ids)


def ensure_filings_for_report(
    plan: ReportPlan,
    request_text: str = "",
) -> EnsureFilingsResult:
    """按计划缺的报告期拉巨潮 PDF，等到 ready（或超时）后返回说明。

    也会等待「库里已有、但仍在 parsing／embedding」的本期／基期文件，
    避免未解析完就抽数出稿。
    """
    result = EnsureFilingsResult()
    if not plan.period:
        return result

    # 不依赖巨潮开关：只要库里已有未就绪文件，就必须等。
    wait_ids = list(_pending_file_ids(plan))

    if settings.enable_cninfo_fetch:
        result.attempted = True
        targets = _missing_targets(plan, request_text)
        # 行业报告一次可能很多家，限制补齐规模，避免打爆巨潮
        limit = max(int(settings.cninfo_auto_fetch_max), 1)
        targets = targets[:limit]

        for subject, code, period in targets:
            try:
                fetched = fetch_missing_reports(code, period, max_files=1)
            except Exception as exc:
                logger.exception("自动补齐失败 subject=%s period=%s", subject, period)
                result.notes.append(f"自动补齐 {subject} {period} 失败：{exc}")
                continue
            if fetched.errors:
                result.notes.extend(
                    f"自动补齐 {subject} {period}：{item}" for item in fetched.errors
                )
            if fetched.skipped and not fetched.files:
                result.notes.extend(
                    f"自动补齐 {subject} {period}：{item}" for item in fetched.skipped
                )
            for item in fetched.files:
                result.fetched_file_ids.append(item.file_id)
                wait_ids.append(item.file_id)
                if item.duplicate_of:
                    result.notes.append(
                        f"沿用已有 {subject} {period}《{item.filename}》"
                        f"（status={item.status or '?'}）"
                    )
                else:
                    result.notes.append(
                        f"已从巨潮补齐 {subject} {period}《{item.filename}》"
                        f"（file_id={item.file_id}）"
                    )

    # 去重保序
    seen: set[str] = set()
    ordered_wait: list[str] = []
    for file_id in wait_ids:
        if not file_id or file_id in seen:
            continue
        seen.add(file_id)
        ordered_wait.append(file_id)

    if ordered_wait:
        not_ready = wait_until_ready(ordered_wait)
        if not_ready:
            # 用文件名+状态写进缺口，比裸 UUID 更好转述给用户
            labels: list[str] = []
            for raw in not_ready:
                try:
                    row = store.get_kb_file(UUID(str(raw)))
                except Exception:
                    row = None
                name = str((row or {}).get("filename") or raw)
                status = str((row or {}).get("status") or "?")
                labels.append(f"{name}({status})")
            result.notes.append(
                "入库等待超时，下列文件尚未 ready，抽数可能仍缺原文："
                + "、".join(labels[:8])
                + (f" 等共 {len(labels)} 份" if len(labels) > 8 else "")
            )
    return result


def _pending_file_ids(plan: ReportPlan) -> list[str]:
    """计划需要的报告期里，库中已有但尚未 ready 的文件。"""
    periods = set(_needed_periods(plan))
    if not periods:
        return []
    names = _plan_name_keys(plan)
    if not names:
        return []
    pending: list[str] = []
    for row in store.list_kb_files():
        status = str(row.get("status") or "")
        if status not in {"uploaded", "parsing", "embedding"}:
            continue
        if str(row.get("period") or "") not in periods:
            continue
        blob = " ".join(
            [
                str(row.get("subject") or ""),
                " ".join(str(item) for item in (row.get("subject_aliases") or [])),
                str(row.get("filename") or ""),
            ]
        )
        from ai_agent.kb.ingest.meta import name_in_text

        if any(name_in_text(blob, name) for name in names):
            file_id = str(row.get("file_id") or "")
            if file_id:
                pending.append(file_id)
    return pending


def _plan_name_keys(plan: ReportPlan) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in plan.subjects:
        key = (item or "").strip()
        if key and key not in seen:
            seen.add(key)
            names.append(key)
    skill = get_skill(plan.skill_id) if plan.skill_id else None
    if skill and skill.companies:
        for name, code, _segment in skill.companies:
            for key in (name, code):
                text = (key or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    names.append(text)
    return names


def _missing_targets(
    plan: ReportPlan, request_text: str
) -> list[tuple[str, str, str]]:
    """返回 (subject_label, stock_code, period) 待补列表。"""
    periods = _needed_periods(plan)
    if not periods:
        return []

    pairs: list[tuple[str, str]] = []  # subject_label, code
    skill = get_skill(plan.skill_id) if plan.skill_id else None
    if skill and skill.companies:
        for name, code, _segment in skill.companies:
            if plan.subjects and name not in plan.subjects:
                # 需求若已收窄主体，只补名单交集
                if not any(name in subject or subject in name for subject in plan.subjects):
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

    # 去重保序
    seen_pair: set[tuple[str, str]] = set()
    unique_pairs: list[tuple[str, str]] = []
    for subject, code in pairs:
        key = (subject, code)
        if key in seen_pair:
            continue
        seen_pair.add(key)
        unique_pairs.append((subject, code))

    targets: list[tuple[str, str, str]] = []
    seen_target: set[tuple[str, str]] = set()
    # 先补本期、再补基期，避免名额被同比期占满而本期一家都没有
    ordered_periods = list(periods)
    if plan.period in ordered_periods:
        ordered_periods = [plan.period] + [
            item for item in ordered_periods if item != plan.period
        ]
    for period in ordered_periods:
        for subject, code in unique_pairs:
            if _files_for([subject], period) or _files_for([code], period):
                continue
            try:
                stock = cninfo.lookup_stock(code)
            except Exception:
                stock = None
            if stock and stock.name and _files_for([stock.name], period):
                continue
            key = (code, period)
            if key in seen_target:
                continue
            seen_target.add(key)
            targets.append((subject, code, period))
    return targets


def _needed_periods(plan: ReportPlan) -> list[str]:
    if not plan.period:
        return []
    periods = [plan.period]
    if "同比" in plan.dimensions:
        yoy = yoy_period(plan.period)
        if yoy:
            periods.append(yoy)
    if "环比" in plan.dimensions:
        for item in cumulative_base_periods(plan.period):
            if item not in periods:
                periods.append(item)
    return periods
