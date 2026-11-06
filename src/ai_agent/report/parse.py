"""把研报需求拆成大纲、文档、时间和对比维度。

公司名只认知识库里已经有的主体，报告期只认需求里写明的那一期。
对不上就不编一家公司、也不猜一个年份。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai_agent.core.logging import get_logger
from ai_agent.kb.meta import name_in_text
from ai_agent.memory.store import store
from ai_agent.report.compare import cumulative_base_periods, qoq_period, yoy_period
from ai_agent.report.industry import indicator_aliases, industry_of
from ai_agent.report.skills import ReportSkill, get_skill, resolve_skill, skill_ids

logger = get_logger(__name__)

_QUARTER_DIGIT = {"一": "1", "二": "2", "三": "3", "四": "4", "1": "1", "2": "2", "3": "3", "4": "4"}
_PERIOD_CODE = re.compile(r"((?:19|20)\d{2})\s*[Qq]([1-4])")
_PERIOD_QUARTER = re.compile(r"((?:19|20)\d{2})\s*年?\s*第?([一二三四1-4])\s*季度")
_PERIOD_HALF = re.compile(r"((?:19|20)\d{2})\s*年?\s*(?:半年度|半年报|中期)")
_PERIOD_YEAR = re.compile(r"((?:19|20)\d{2})\s*年(?:度报告|年报|度)?")
_INDUSTRY = re.compile(r"([0-9A-Za-z\u4e00-\u9fff]{2,12})行业")
_REPORT = re.compile(r"研报|财报|深度分析|分析报告")
_PDF = re.compile(r"PDF|pdf")
# 长名称放前面，避免「净利润」从「扣非净利润」里再认一次
_INDICATOR_KEYS = (
    ("扣非净利润", ("扣非净利润", "扣非")),
    ("归母净利润", ("归母净利润", "归母")),
    ("毛利率", ("毛利率",)),
    ("净利润", ("净利润",)),
    ("营业收入", ("营业收入", "营收")),
)
# 默认指标集在 report/industry.py 按主体所属行业选择，这里只处理「需求点名」。
_INDUSTRY_STOP = {"所属", "本公司", "该", "全", "本"}


@dataclass
class OutlineItem:
    title: str
    brief: str


@dataclass
class SourceFile:
    file_id: str
    filename: str
    subject: str
    period: str
    role: str


@dataclass
class ReportPlan:
    title: str
    kind: str
    deliverable: str
    subjects: list[str] = field(default_factory=list)
    industry: str = ""
    period: str = ""
    dimensions: list[str] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)
    outline: list[OutlineItem] = field(default_factory=list)
    files: list[SourceFile] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    # 命中的报告类型（report/skills.py 注册表）。显式指定未知类型时
    # 落回路由结果并把原因写进 gaps，不静默忽略。
    skill_id: str = ""


def parse_request(text: str, skill_id: str = "") -> ReportPlan:
    """解析一句研报需求。知识库读失败时计划仍返回，缺口写在 gaps。

    skill_id 由调用方显式指定报告类型（聊天侧/端点的 report_type 参数）：
    命中注册表就用它的大纲与指标集；没命中则照常路由并写一条缺口。
    """
    request = (text or "").strip()
    plan = ReportPlan(
        title="",
        kind="unknown",
        deliverable="pdf" if _PDF.search(request) else "markdown",
    )
    if not request or not _REPORT.search(request):
        plan.gaps.append("没有看到研报或财报分析需求")
        return plan

    periods = _periods_in(request)
    if not periods:
        plan.gaps.append("没有写明报告期")
    else:
        plan.period = periods[-1]
        if len(periods) > 1:
            plan.gaps.append(f"写了多个报告期，主报告期取 {plan.period}")

    try:
        rows = [row for row in store.list_kb_files() if row.get("status") == "ready"]
    except Exception:
        logger.exception("解析研报需求时读取文件失败")
        plan.gaps.append("知识库文件列表读取失败")
        rows = []

    plan.subjects = _mentioned_subjects(request, rows)
    plan.industry = _industry_of(request)
    if plan.subjects:
        plan.kind = "company"
    elif plan.industry:
        plan.kind = "industry"
        plan.subjects = _industry_subjects(plan.industry, rows, plan.period)
        if not plan.subjects:
            plan.gaps.append(f"知识库里没有标成{plan.industry}行业的公司，无法确定要调取哪些季报")
    else:
        plan.gaps.append("没有点到知识库里的公司，也没有写行业")

    # skill 路由：显式指定优先；没指定就按 kind + 主体行业路由。
    skill = _resolve_plan_skill(plan, skill_id)
    plan.skill_id = skill.skill_id
    plan.indicators = _indicators_in(request, plan.subjects, skill)
    plan.dimensions = _dimensions(plan)
    plan.files = _planned_files(plan, rows)
    plan.outline = _outline(plan, skill)
    plan.title = _title(plan)
    return plan


def _resolve_plan_skill(plan: ReportPlan, skill_id: str) -> ReportSkill:
    """skill 层唯一入口：先看显式指定，再按 kind + 行业路由。"""
    if skill_id:
        chosen = get_skill(skill_id)
        if chosen is not None:
            # 显式指定允许覆盖路由出的 kind（比如把公司请求强制走行业模板），
            # 但主体与报告期仍来自需求文本，skill 不改事实、只改写法。
            plan.kind = chosen.kind
            return chosen
        plan.gaps.append(
            f"未知报告类型 {skill_id}（可选：{'、'.join(skill_ids())}），已按需求自动路由"
        )
    return resolve_skill(plan.kind, industry_of(plan.subjects))


def _periods_in(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def add(period: str) -> None:
        if period and period not in seen:
            seen.add(period)
            found.append(period)

    for match in _PERIOD_CODE.finditer(text):
        add(f"{match.group(1)}Q{match.group(2)}")
    for match in _PERIOD_QUARTER.finditer(text):
        add(f"{match.group(1)}Q{_QUARTER_DIGIT[match.group(2)]}")
    for match in _PERIOD_HALF.finditer(text):
        add(f"{match.group(1)}H1")
    if not found:
        for match in _PERIOD_YEAR.finditer(text):
            add(match.group(1))
    return sorted(found, key=_period_sort)


def _period_sort(period: str) -> tuple[int, int]:
    from ai_agent.kb.meta import period_rank

    return period_rank(period)


def _mentioned_subjects(text: str, rows: list[dict]) -> list[str]:
    found: list[str] = []
    for row in rows:
        subject = str(row.get("subject") or "").strip()
        if not subject or subject in found:
            continue
        names = [subject, *list(row.get("subject_aliases") or []), str(row.get("filename") or "")]
        if any(name and name_in_text(text, str(name)) for name in names):
            found.append(subject)
    return found


def _industry_of(text: str) -> str:
    match = _INDUSTRY.search(text)
    if not match:
        return ""
    name = match.group(1)
    if name in _INDUSTRY_STOP:
        return ""
    return name


def _industry_subjects(industry: str, rows: list[dict], period: str) -> list[str]:
    found: list[str] = []
    for row in rows:
        if period and (row.get("period") or "") != period:
            continue
        subject = str(row.get("subject") or "").strip()
        if not subject or subject in found:
            continue
        blob = " ".join(
            [
                subject,
                " ".join(str(item) for item in (row.get("subject_aliases") or [])),
                str(row.get("filename") or ""),
                " ".join(str(item) for item in (row.get("tags") or [])),
                str(row.get("description") or ""),
            ]
        )
        if industry in blob:
            found.append(subject)
    return found


def _indicators_in(text: str, subjects: list[str], skill: ReportSkill) -> list[str]:
    found: list[str] = []
    occupied: list[tuple[int, int]] = []
    for name, keys in _INDICATOR_KEYS:
        for key in keys:
            start = text.find(key)
            if start < 0:
                continue
            span = (start, start + len(key))
            if any(old[0] <= span[0] and span[1] <= old[1] for old in occupied):
                continue
            occupied.append(span)
            found.append(name)
            break
    # 银行/保险这类专用指标用简称归一，点名了就按点名的来
    for alias, canonical in indicator_aliases():
        if alias in text and canonical not in found:
            found.append(canonical)
    # 没点名指标就落回 skill 携带的默认指标集（skill 层的「写什么」）
    return found or list(skill.indicators)


def _dimensions(plan: ReportPlan) -> list[str]:
    if not plan.period:
        return []
    items = ["同比"]
    if qoq_period(plan.period):
        items.append("环比")
    if plan.kind == "industry" or len(plan.subjects) > 1:
        items.append("公司间对比")
    return items


def _planned_files(plan: ReportPlan, rows: list[dict]) -> list[SourceFile]:
    if not plan.period or not plan.subjects:
        return []
    wanted = [(plan.period, "本期")]
    yoy = yoy_period(plan.period)
    if "同比" in plan.dimensions and yoy:
        wanted.append((yoy, "同比基期"))
    if "环比" in plan.dimensions:
        # A 股没有独立二季报：qoq_period("2026Q3") 指出的 "2026Q2" 永远
        # 找不到文件。改为调取推导单季所需的 Q1 / H1 累计报告，
        # 由 compare 用「相邻累计相减」还原单季后再算真环比。
        bases = cumulative_base_periods(plan.period)
        if bases:
            wanted.extend((item, "累计基期") for item in bases)
        else:
            qoq = qoq_period(plan.period)
            if qoq:
                wanted.append((qoq, "环比基期"))
    files: list[SourceFile] = []
    for period, role in wanted:
        matched = _files_for(plan.subjects, period, rows)
        if not matched:
            if role == "本期":
                plan.gaps.append(f"没有找到 {period} 的季报")
            elif role == "累计基期":
                plan.gaps.append(
                    f"A 股无独立二季报；缺 {period} 的累计报告，"
                    f"{plan.period} 单季环比无法推导"
                )
            else:
                # 说清楚缺的是「跨期绝对值」：本期季报表格里可能已披露
                # 同比增减，那一列仍可用，别说成「同比无法调取」。
                plan.gaps.append(f"没有找到 {period} 的季报，{role}的绝对值无法调取，无法跨期重算")
            continue
        for row in matched:
            files.append(
                SourceFile(
                    file_id=str(row.get("file_id") or ""),
                    filename=str(row.get("filename") or ""),
                    subject=str(row.get("subject") or ""),
                    period=period,
                    role=role,
                )
            )
    return files


def _files_for(subjects: list[str], period: str, rows: list[dict]) -> list[dict]:
    chosen: list[dict] = []
    for row in rows:
        if (row.get("period") or "") != period:
            continue
        blob = " ".join(
            [
                str(row.get("subject") or ""),
                " ".join(str(item) for item in (row.get("subject_aliases") or [])),
                str(row.get("filename") or ""),
            ]
        )
        if any(name_in_text(blob, subject) for subject in subjects):
            chosen.append(row)
    return chosen


def _outline(plan: ReportPlan, skill: ReportSkill) -> list[OutlineItem]:
    """大纲由 skill 注册表提供；parse 只负责把占位填成实际指标集。"""
    return [OutlineItem(title, brief) for title, brief in skill.build_outline(plan.indicators)]


def _title(plan: ReportPlan) -> str:
    if plan.kind == "industry" and plan.industry and plan.period:
        return f"{plan.industry}行业 {plan.period} 深度分析"
    if plan.subjects and plan.period:
        names = "、".join(plan.subjects)
        return f"{names} {plan.period} 财报深度分析"
    return ""
