"""把研报需求拆成大纲、文档、时间和对比维度。

公司名只认知识库里已经有的主体，报告期只认需求里写明的那一期。
对不上就不编一家公司、也不猜一个年份。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai_agent.core.logging import get_logger
from ai_agent.kb.ingest.meta import fragment_subjects, is_plausible_subject, name_in_text
from ai_agent.storage.store import store
from ai_agent.report.compare import cumulative_base_periods, qoq_period, yoy_period
from ai_agent.report.industry import (
    GENERIC,
    company_type_of,
    default_indicators,
    indicator_aliases,
    industry_of,
)
from ai_agent.report.skills import ReportSkill, get_skill, resolve_skill, skill_ids
from ai_agent.report import registry

logger = get_logger(__name__)

_QUARTER_DIGIT = {"一": "1", "二": "2", "三": "3", "四": "4", "1": "1", "2": "2", "3": "3", "4": "4"}
_PERIOD_CODE = re.compile(r"((?:19|20)\d{2})\s*[Qq]([1-4])")
_PERIOD_QUARTER = re.compile(r"((?:19|20)\d{2})\s*年?\s*第?([一二三四1-4])\s*季度")
_PERIOD_HALF = re.compile(r"((?:19|20)\d{2})\s*年?\s*(?:半年度|半年报|中期)")
_PERIOD_YEAR = re.compile(r"((?:19|20)\d{2})\s*年(?:度报告|年报|度)?")
_INDUSTRY = re.compile(r"([0-9A-Za-z\u4e00-\u9fff]{2,12})行业")
_REPORT = re.compile(r"研报|财报|深度分析|分析报告")
# 默认出 PDF；只有明确只要 Markdown / 不要 PDF 才降级（聊天侧常不写「PDF」二字）
_MARKDOWN_ONLY = re.compile(
    r"(只要|仅要|仅出|只要出|仅生成|只要生成)\s*(Markdown|markdown|MD\b|md\b)"
    r"|不要\s*PDF|无需\s*PDF|跳过\s*PDF|不需要\s*PDF"
    r"|Markdown\s*即可|只要\s*正文",
    re.I,
)
# 长名称放前面，避免「净利润」从「扣非净利润」里再认一次
_INDICATOR_KEYS = registry.indicator_keys()
# 默认指标集在 report/industry.py 按主体所属行业选择，这里只处理「需求点名」。
_INDUSTRY_STOP = {"所属", "本公司", "该", "全", "本"}
# 需求里的动词前缀：「生成储能行业」会被 _INDUSTRY 贪婪匹配成「生成储能」，
# 长词在前，逐个剥掉再判行业名。只处理动词，不猜行业。
_INDUSTRY_VERB_PREFIXES = tuple(
    sorted(
        {
            "帮我生成",
            "帮我写",
            "帮我做",
            "帮我出",
            "给我生成",
            "给我出",
            "给我写",
            "给我做",
            "给我来",
            "请生成",
            "麻烦生成",
            "生成一份",
            "生成一个",
            "生成",
            "写一份",
            "写一个",
            "写篇",
            "写",
            "做一份",
            "做一个",
            "做",
            "出一份",
            "出一个",
            "出",
            "来一份",
            "来个",
            # 量词前缀：「帮我写一份保险行业」会被捕获成「一份保险」
            "一份",
            "一个",
            "一篇",
            "一张",
            "份",
            "篇",
            "分析一下",
            "分析",
            "研究一下",
            "研究",
            "看一下",
            "看看",
            "看下",
            "我要",
            "我想要",
            "请",
            "麻烦",
        },
        key=len,
        reverse=True,
    )
)


@dataclass
class OutlineItem:
    title: str
    brief: str
    # SKILL.md 里可选的 `section: <id>`；为空时 compose 按标题别名回退。
    section_id: str = ""


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
    # 细粒度公司类型（fabless/foundry/bank…），决定指标模板；与 industry 粗路由分离。
    company_type: str = ""
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
        deliverable="markdown" if _MARKDOWN_ONLY.search(request) else "pdf",
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
    # 「机器人行业」里的行业词会命中新松证券简称／又名，不能因此改成公司单报。
    if plan.industry and f"{plan.industry}行业" in request:
        plan.kind = "industry"
        plan.subjects = []
    elif plan.subjects:
        plan.kind = "company"
    elif plan.industry:
        # 主体等 skill 路由后再定：有花名册用花名册，没有才靠 KB tags。
        plan.kind = "industry"
        plan.subjects = []
    else:
        plan.gaps.append("没有点到知识库里的公司，也没有写行业")

    # skill 路由：显式指定优先；没指定就按 kind + 行业路由。
    skill = _resolve_plan_skill(plan, skill_id)
    plan.skill_id = skill.skill_id
    # 未注册行业别静默走通用模板：写一条缺口，编辑能看懂该补什么。
    if (
        not skill_id
        and plan.kind == "industry"
        and plan.industry
        and plan.industry not in skill.industries
    ):
        plan.gaps.append(
            f"「{plan.industry}」没有专属报告模板，已按通用行业模板生成；"
            "要定制大纲或指标集请新增 SKILL.md"
        )
    # 行业主体：skill.companies 是宇宙唯一来源；KB 只决定有没有原文／要不要补齐。
    # 禁止用行业词扫公司名／文件名——「机器人」会撞上新松简称，把花名册挡住。
    if plan.kind == "industry":
        if skill.companies:
            plan.subjects = [
                name for name, _code, _segment, _aka in skill.companies
            ]
        else:
            plan.subjects = _industry_subjects(
                plan.industry, rows, plan.period
            )
            if not plan.subjects:
                plan.gaps.append(
                    f"知识库里没有标成{plan.industry}行业的公司，无法确定要调取哪些季报"
                )
    if plan.kind == "company":
        plan.company_type = company_type_of(plan.subjects)
        if plan.subjects and plan.company_type == GENERIC:
            # 判不出品类时按通用集抽数，但要让读者知道这是通用口径，不是专属模板。
            plan.gaps.append(
                "未识别公司品类，已按通用指标集生成；"
                "可在 indicators.yaml 的 classification.name_hints 补关键字，"
                "或把公司写进行业花名册"
            )
    elif skill.industries:
        plan.company_type = skill.industries[0]
    else:
        plan.company_type = plan.industry or ""
    plan.indicators = _indicators_in(request, plan.subjects, skill, plan.company_type)
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
            if chosen.kind == "industry" and chosen.industries and not plan.industry:
                plan.industry = chosen.industries[0]
            return chosen
        plan.gaps.append(
            f"未知报告类型 {skill_id}（可选：{'、'.join(skill_ids())}），已按需求自动路由"
        )
    # 行业报告用需求里的行业名（如「半导体」）；公司报告用主体推断的 bank/generic。
    if plan.kind == "industry":
        return resolve_skill(plan.kind, plan.industry)
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
    from ai_agent.kb.ingest.meta import period_rank

    return period_rank(period)


def _mentioned_subjects(text: str, rows: list[dict]) -> list[str]:
    """点名公司：全称／别名整段命中，或主体核心名里的独有片段（如「中科飞测」）。

    库里误标成「股份有限公司」「公开发行证券」的脏 subject 直接跳过，
    避免需求里写了真全称时被套话主体一起带上。
    """
    found: list[str] = []
    catalog: list[tuple[str, list[str]]] = []
    for row in rows:
        subject = str(row.get("subject") or "").strip()
        if not subject or not is_plausible_subject(subject):
            continue
        aliases = [str(item) for item in (row.get("subject_aliases") or []) if item]
        catalog.append((subject, aliases))
        if subject in found:
            continue
        names = [subject, *aliases, str(row.get("filename") or "")]
        if any(name and name_in_text(text, str(name)) for name in names):
            found.append(subject)
    for subject in fragment_subjects(text, catalog):
        if subject not in found and is_plausible_subject(subject):
            found.append(subject)
    return found


def _strip_request_verb(name: str) -> str:
    """剥掉「生成／写一份／分析一下」这类需求动词，只留行业名本体。"""
    text = name
    changed = True
    while changed:
        changed = False
        for prefix in _INDUSTRY_VERB_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                text = text[len(prefix) :]
                changed = True
                break
    return text


def _industry_of(text: str) -> str:
    # 先命中 skill 登记过的行业名，避免「生成半导体行业」被正则吃成「生成半导体」。
    from ai_agent.report.skills import _load_skills

    known = sorted(
        {
            item
            for skill in _load_skills()
            for item in skill.industries
            if item
        },
        key=len,
        reverse=True,
    )
    for item in known:
        if f"{item}行业" in text or item in text:
            # 单独的「银行」过短且易误伤公司名；仅接受「X行业」或已知多字行业词
            if len(item) >= 2 and f"{item}行业" in text:
                return item
            if len(item) >= 3 and item in text:
                return item
    match = _INDUSTRY.search(text)
    if not match:
        return ""
    name = _strip_request_verb(match.group(1))
    if len(name) < 2 or name in _INDUSTRY_STOP:
        return ""
    # 已知行业是捕获串后缀时收回（「生成半导体」→「半导体」）
    for item in known:
        if name.endswith(item) and item:
            return item
    return name


def _industry_subjects(industry: str, rows: list[dict], period: str) -> list[str]:
    """通用行业模板兜底：只认 tags 里的行业标注，不扫公司名／文件名。

    有专属 skill 花名册时不会走到这里；子串扫 subject／filename 会把
    「机器人」误判成新松独家成员。
    """
    found: list[str] = []
    token = (industry or "").strip()
    if not token:
        return found
    for row in rows:
        if period and (row.get("period") or "") != period:
            continue
        subject = str(row.get("subject") or "").strip()
        if not subject or subject in found:
            continue
        tags = [str(item) for item in (row.get("tags") or [])]
        if token in tags:
            found.append(subject)
    return found


def _indicators_in(
    text: str, subjects: list[str], skill: ReportSkill, company_type: str = ""
) -> list[str]:
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
    if found:
        return found
    # 公司单报：按细类型模板取数，不吃 skill 里过时的「通吃」集。
    # 行业报告仍用 skill.indicator_set（赛道横比共用集）。
    if skill.kind == "company" and company_type:
        return default_indicators(company_type)
    return list(skill.indicators)


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
    return [
        OutlineItem(title, brief, section_id)
        for title, brief, section_id in skill.build_outline(plan.indicators)
    ]


def _title(plan: ReportPlan) -> str:
    if plan.kind == "industry" and plan.industry and plan.period:
        from ai_agent.report.skills import is_segment_skill

        if is_segment_skill(plan.skill_id):
            return f"{plan.industry}行业 {plan.period} 财报对比分析（赛道分组版）"
        return f"{plan.industry}行业 {plan.period} 深度分析"
    if plan.subjects and plan.period:
        names = "、".join(plan.subjects)
        return f"{names} {plan.period} 财报深度分析"
    return ""
