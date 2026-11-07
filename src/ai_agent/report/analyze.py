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
from ai_agent.report.extract import (
    Evidence,
    IndicatorFact,
    citation_anchor,
    _as_uuid,
    _files_for,
)
from ai_agent.report.industry import notes_for_plan
from ai_agent.report.parse import ReportPlan
from ai_agent.report.supplemental import SupplementalResult
from ai_agent.report.validate import _allowed, _measure, _number_key
from ai_agent.storage.store import store

logger = get_logger(__name__)

# compose._ANALYSIS_FALLBACK 须与此同文（compose 不反向 import 本模块）。
ANALYSIS_FALLBACK = "经营解读：本期季报未提供可核对的归因表述，本节从略。"
_FALLBACK = ANALYSIS_FALLBACK
_TOPICS = frozenset({"nim", "profit", "asset_quality", "other"})
_CLAIM_KINDS = frozenset({"restatement", "comparison", "gap"})
_NUMBER = re.compile(r"-?\d[\d,，]*(?:[ \t\u00a0]\d{3})*(?:\.\d+)?%?")
_EVIDENCE_KEYWORDS = (
    "经营情况简要分析",
    "经营情况讨论与分析",
    "主要会计数据、财务指标发生变动的情况、原因",
    "净息差",
    "不良贷款",
    "拨备覆盖率",
    "营业收入",
    "净利润",
    "毛利率",
    "研发费用",
    "研发投入",
    "产能利用率",
    "量价齐升",
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

    if is_segment_skill(plan.skill_id):
        lines, derived = _segment_structure_lines(plan, current)
        if lines:
            return AnalysisSection(lines=lines, used_llm=False, derived_facts=derived)
        return AnalysisSection(lines=[_FALLBACK], gaps=["缺少可排序的毛利率或研发费用率"])

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

    use_llm = settings.report_analysis_enabled if enabled is None else enabled
    if not use_llm:
        lines = _deterministic_lines(plan, current, pack)
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
        lines = _deterministic_lines(plan, current, pack)
        if lines:
            return AnalysisSection(
                lines=lines, gaps=["经营解读模型调用失败，已改用原文引述"], used_llm=False
            )
        return AnalysisSection(lines=[_FALLBACK], gaps=["经营解读模型调用失败"], used_llm=False)

    parsed = load_llm_json(raw, repair=True)
    if not isinstance(parsed, dict):
        lines = _deterministic_lines(plan, current, pack)
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
        fallback_lines = _deterministic_lines(plan, current, pack)
        if fallback_lines:
            gaps = list(gaps) or ["经营解读无合法 LLM 段落"]
            gaps.append("已改用原文证据引述")
            return AnalysisSection(lines=fallback_lines, gaps=gaps, used_llm=True)
        gaps = list(gaps) or ["经营解读无合法段落"]
        return AnalysisSection(lines=[_FALLBACK], gaps=gaps, used_llm=True)
    return AnalysisSection(lines=lines, gaps=gaps, used_llm=True)


def _segment_structure_lines(
    plan: ReportPlan, facts: list[IndicatorFact]
) -> tuple[list[str], list[IndicatorFact]]:
    """用已授权毛利率/研发费用率写梯队与结构关系（①级数据关系，非第三方因果）。"""
    from ai_agent.report.compare import parse_number
    from ai_agent.report.skills import get_skill

    skill = get_skill(plan.skill_id) if plan.skill_id else None
    segment_of = skill.segment_of if skill else (lambda _name: "")
    # 行业专属的定性 gloss 来自 industries.yaml；未配置则用通用句。
    notes = notes_for_plan(plan)

    def _ranked(indicator: str) -> list[tuple[str, str, str, object]]:
        rows: list[tuple[str, str, str, object]] = []
        for fact in facts:
            if fact.name != indicator:
                continue
            parsed = parse_number(fact.value)
            if parsed is None:
                continue
            amount, _ = parsed
            cite = citation_anchor(fact.evidence.file_id)
            rows.append((fact.subject, fact.value, cite, amount))
        rows.sort(key=lambda item: item[3], reverse=True)
        return rows

    margins = _ranked("毛利率")
    rd_rates = _ranked("研发费用率")
    if len(margins) < 2 and len(rd_rates) < 2:
        return [], []

    lines: list[str] = []
    derived: list[IndicatorFact] = []
    if margins:
        ladder = " > ".join(
            f"{name} {value}%{f' {cite}' if cite else ''}"
            for name, value, cite, _ in margins
        )
        lines.append(f"毛利率梯队（从高到低）：{ladder}。")
    if rd_rates:
        ladder = " > ".join(
            f"{name} {value}%{f' {cite}' if cite else ''}"
            for name, value, cite, _ in rd_rates
        )
        lines.append(f"研发费用率梯队（从高到低）：{ladder}。")

    # 按赛道聚合格率均值，只陈述高低关系，不写外部因果
    def _segment_avg(rows: list[tuple[str, str, str, object]]) -> list[tuple[str, object]]:
        buckets: dict[str, list[object]] = {}
        for name, _value, _cite, amount in rows:
            seg = segment_of(name) or "未分组"
            buckets.setdefault(seg, []).append(amount)
        out: list[tuple[str, object]] = []
        for seg, amounts in buckets.items():
            if not amounts:
                continue
            avg = sum(amounts) / len(amounts)  # type: ignore[arg-type]
            out.append((seg, avg))
        out.sort(key=lambda item: item[1], reverse=True)
        return out

    margin_seg = _segment_avg(margins)
    rd_seg = _segment_avg(rd_rates)
    if len(margin_seg) >= 2:
        detail = " / ".join(f"{seg}≈{_fmt_avg(avg)}%" for seg, avg in margin_seg)
        order = " > ".join(seg for seg, _ in margin_seg)
        gloss = notes.margin_gloss or "此为已抽取数字的横向关系，不构成外部归因。"
        lines.append(f"按赛道看毛利率均值排序为 {order}（{detail}）：{gloss}")
        derived.extend(
            _segment_avg_fact(plan.period, "毛利率", seg, avg) for seg, avg in margin_seg
        )
    if len(rd_seg) >= 2:
        detail = " / ".join(f"{seg}≈{_fmt_avg(avg)}%" for seg, avg in rd_seg)
        order = " > ".join(seg for seg, _ in rd_seg)
        gloss = notes.rnd_gloss or "同样只陈述本期数字结构，不引入无出处因果。"
        lines.append(f"按赛道看研发费用率均值排序为 {order}（{detail}）：{gloss}")
        derived.extend(
            _segment_avg_fact(plan.period, "研发费用率", seg, avg) for seg, avg in rd_seg
        )
    lines.append("以上判断仅基于本期已抽取或结构化补齐的比率，不引入无出处的订单／景气推断。")
    return lines, derived


# 兼容旧测试名。
_semiconductor_structure_lines = _segment_structure_lines


def _fmt_avg(amount: object) -> str:
    """赛道均值展示到一位小数，便于正文横比阅读。"""
    try:
        return f"{float(amount):.1f}"
    except (TypeError, ValueError):
        return str(amount)


def _segment_avg_fact(
    period: str, indicator: str, segment: str, avg: object
) -> IndicatorFact:
    value = _fmt_avg(avg)
    quote = f"{segment}赛道{indicator}均值≈{value}%（成员本期{indicator}算术平均）"
    return IndicatorFact(
        subject=f"{segment}赛道均值",
        period=period,
        name=f"{indicator}均值",
        value=value,
        unit="%",
        column="赛道均值推算",
        scope="derived",
        evidence=Evidence(
            file_id=f"segment-avg-{segment}-{indicator}",
            filename="derived.segment_average",
            page=None,
            page_end=None,
            section="经营解读",
            quote=quote,
        ),
    )


def _deterministic_lines(
    plan: ReportPlan,
    facts: list[IndicatorFact],
    pack: list[EvidenceItem],
) -> list[str]:
    """无可靠 LLM 输出时：①并列已授权数字事实；②引述经营分析原文（带出处）。

    不写第三方推断。证据不足时返回空，由调用方落从略句。
    """
    from ai_agent.report.skills import is_segment_skill

    if is_segment_skill(plan.skill_id):
        lines, _derived = _segment_structure_lines(plan, facts)
        return lines

    lines: list[str] = []
    # ① 有经营原文明文时，不再用「长主语 + 已出现数字」复述；先写原文，再补一句
    # 可指回原文的结构关系（比率须已在 facts 中）。
    # 变动原因表常把「量价齐升」写在「主要原因」单元格——先抽叙述再判噪音，
    # 禁止因表头词整段丢弃有效原文。
    keys = ("量价", "存储", "出货", "毛利润增加", "经营业绩", "产能", "研发")
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
    used = 0
    seen_snippets: set[str] = set()
    for item in pack:
        raw = _extract_operating_narrative(_plain_evidence_quote(item.quote or ""))
        if len(raw) < 20 or not any(token in raw for token in keys):
            continue
        if any(token in raw for token in ban):
            continue
        # 纯报表模板残片（无经营叙述词）跳过；含「量价／存储」的变动原因表保留。
        if _is_statement_template(raw) and not any(
            token in raw for token in ("量价", "存储", "出货", "毛利润增加", "经营业绩")
        ):
            continue
        for snippet in _iter_reason_snippets(raw):
            snippet = _extract_operating_narrative(snippet)
            if not snippet or not _is_usable_operating_quote(snippet):
                continue
            if _is_form_chrome(snippet):
                continue
            key = snippet[:40]
            if key in seen_snippets:
                continue
            seen_snippets.add(key)
            quote = _safe_truncate(snippet, 120)
            cite = _format_citation([item])
            lines.append(f"原文披露：{quote}{cite}。")
            used += 1
            if used >= 2:
                break
        if used >= 2:
            break
    cited = "\n".join(lines)
    structure = _company_structure_line(
        plan, facts, has_narrative=bool(lines), cited_text=cited
    )
    if structure:
        lines.append(structure)
    if not lines:
        # 无原文可引时才退回数字并列，避免空节。
        priority = ("毛利率", "营业收入", "扣非净利润", "研发费用率")
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


def _company_structure_line(
    plan: ReportPlan,
    facts: list[IndicatorFact],
    *,
    has_narrative: bool,
    cited_text: str = "",
) -> str:
    """有「量价齐升／毛利润增加」类原文时，用已授权毛利率做一句结构并列（非外部归因）。"""
    if not has_narrative:
        return ""
    margin = next((f for f in facts if f.name == "毛利率" and f.period == plan.period), None)
    if margin is None:
        return ""
    # 结构句只点名正文里真实出现过的原文用语，避免「引用了量价齐升但原文没了」。
    labels = [
        label
        for label, token in (
            ("量价齐升", "量价"),
            ("毛利润增加", "毛利润增加"),
        )
        if token in (cited_text or "")
    ]
    if not labels:
        return ""
    label_text = "／".join(labels)
    anchor = citation_anchor(margin.evidence.file_id)
    source = margin.evidence.filename or ""
    # akshare 财务摘要列是报告期末累计口径，须写明，避免与单季营收混读。
    if source.startswith("akshare."):
        scope = "前三季度累计口径" if plan.period.endswith("Q3") else "结构化摘要累计口径"
        cite = f"（{scope} {anchor}）" if anchor else f"（{scope}）"
    else:
        page = f"第{margin.evidence.page}页" if margin.evidence.page is not None else ""
        cite = f"（{page} {anchor}）" if page or anchor else ""
    return (
        f"与原文「{label_text}」并列：本期毛利率为 {margin.value}%"
        f"{cite}——只陈述已授权比率与原文用语的共现，不构成外部归因。"
    )


def _is_statement_template(text: str) -> bool:
    return any(
        token in text
        for token in ("合并利润表", "合并资产负债表", "编制单位", "审计类型")
    )


def _is_form_chrome(text: str) -> bool:
    """抽后的短句若仍是表头／勾选框，才丢弃。"""
    if "适用" in text and "不适用" in text:
        return True
    if "变动比例" in text and "主要原因" in text:
        return True
    return _is_statement_template(text)


def _iter_reason_snippets(quote: str) -> list[str]:
    """同一证据块可拆出多条叙述：量价齐升与毛利润增加不要互相抢占。"""
    # 主题分组：每组至多取一条；组内取文中最早命中，避免「量价」截掉「报告期内」开头。
    theme_groups = (
        ("报告期内", "量价", "存储", "出货", "经营业绩"),
        ("毛利润增加", "主要是", "主要系"),
    )
    snippets: list[str] = []
    occupied: list[tuple[int, int]] = []
    for group in theme_groups:
        hits = sorted(
            ((quote.find(marker), marker) for marker in group if marker in quote),
            key=lambda item: item[0],
        )
        for idx, _marker in hits:
            if idx < 0:
                continue
            if any(start <= idx < end for start, end in occupied):
                continue
            tail = quote[idx:]
            caused = tail.find("所致")
            if caused >= 0 and caused < 80:
                piece = tail[: caused + 2]
            else:
                cut = re.search(
                    r"\s+(?:营业收入|利润总额|归属于上市公司|基本每股|稀释每股|加权平均)",
                    tail,
                )
                piece = tail[: cut.start()] if cut else tail
                stop = piece.find("。")
                if stop >= 0:
                    piece = piece[: stop + 1]
            cleaned = _keep_operating_fragments(
                _strip_quote_numbers(piece.strip(" ；;|"))
            )
            if not cleaned:
                continue
            snippets.append(cleaned)
            occupied.append((idx, idx + max(len(piece), 1)))
            break
        if len(snippets) >= 2:
            break
    if snippets:
        return snippets
    fallback = _keep_operating_fragments(_reason_only(quote))
    return [fallback] if fallback else []


_TIP_PREFIX = re.compile(
    r"需提醒投资者关注的关于公司报告期经营情况的其他重要信息[、,，]?"
)
_SHAREHOLDER_NOISE = (
    "转融通",
    "出借股份",
    "无限售流通股",
    "前十名股东",
    "前 名股东",
    "前名股东",
)
_OPERATING_MARKERS = (
    "报告期内",
    "公司前三季度",
    "前三季度",
    "本报告期",
    "本期公司",
    "围绕重点",
    "持续优化",
    "经营业绩",
    "量价",
    "毛利润增加",
    "主要是",
    "主要系",
)
_OPERATING_CONTENT = (
    "产能",
    "客户",
    "技术",
    "经营",
    "毛利",
    "量价",
    "研发",
    "应用板块",
    "出货",
    "存储",
)
# 利润表模板科目；成串出现 = 科目清单，不是经营解读。
_PL_ACCOUNT_NAMES = (
    "营业成本",
    "利息支出",
    "手续费及佣金支出",
    "退保金",
    "赔付支出",
    "提取保险责任准备金净额",
    "提取保险责任准备金",
    "保单红利支出",
    "分保费用",
    "税金及附加",
    "销售费用",
    "管理费用",
    "研发费用",
    "财务费用",
    "资产减值损失",
    "信用减值损失",
    "其他收益",
    "投资收益",
    "公允价值变动收益",
    "营业外收入",
    "营业外支出",
    "所得税费用",
)
# 封测／制造公司正文不该出现的跨行业科目（保险／银行模板残留）。
_CROSS_INDUSTRY_ACCOUNTS = (
    "退保金",
    "赔付支出",
    "保单红利",
    "提取保险责任准备金",
    "分保费用",
    "净息差",
    "不良贷款",
    "拨备覆盖率",
    "利息净收入",
)
_NARRATIVE_CUES = (
    "优化",
    "布局",
    "增长",
    "下降",
    "提升",
    "推进",
    "持续",
    "实现",
    "所致",
    "主要是",
    "主要系",
    "围绕",
    "保持",
    "量价",
    "出货",
    "发力",
    "向好",
)
_HOLLOW_NUMBER_SLOT = re.compile(
    r"(＿|_{2,}|人民币\s+亿?元|为\s+元|同比\s*(?:增长|下降|增减)?\s*[。.%％]?)"
)


def _plain_evidence_quote(raw: str) -> str:
    """去掉 HTML 标签，压成单行，并清洗 PDF 断字／勾选框噪音。"""
    from ai_agent.report.extract import sanitize_evidence_quote

    text = re.sub(r"<[^>]+>", " ", raw or "")
    return sanitize_evidence_quote(text)


def _is_account_catalog(text: str) -> bool:
    """整段基本是利润表科目名罗列（长电第 2 条引文这类）。"""
    hits = sum(1 for name in _PL_ACCOUNT_NAMES if name in (text or ""))
    if hits >= 4:
        return True
    return "其中" in (text or "") and hits >= 2


def _has_hollow_number_slots(text: str) -> bool:
    """数字被抽空后留下「人民币 亿元／同比 。」半截句。"""
    return bool(_HOLLOW_NUMBER_SLOT.search(text or ""))


def _looks_like_operating_sentence(text: str) -> bool:
    if any(mark in text for mark in ("。", "；", "所致")):
        return True
    return any(cue in text for cue in _NARRATIVE_CUES)


def _is_usable_operating_quote(text: str) -> bool:
    """单片段是否可引用：成句、非科目清单、无跨行业残留、无数字空壳。"""
    cleaned = (text or "").strip()
    if len(cleaned) < 20:
        return False
    if any(token in cleaned for token in _CROSS_INDUSTRY_ACCOUNTS):
        return False
    if _is_account_catalog(cleaned):
        return False
    if _has_hollow_number_slots(cleaned):
        return False
    if not _looks_like_operating_sentence(cleaned):
        return False
    return True


def _excise_bad_clauses(text: str) -> str:
    """按逗号切开，丢掉空缺／科目罗列／跨行业子句，保留其余。"""
    parts = re.split(r"([，,])", text or "")
    kept: list[str] = []
    idx = 0
    while idx < len(parts):
        chunk = parts[idx]
        delim = ""
        if idx + 1 < len(parts) and parts[idx + 1] in {"，", ","}:
            delim = parts[idx + 1]
            idx += 2
        else:
            idx += 1
        piece = chunk.strip()
        if not piece:
            continue
        if _has_hollow_number_slots(piece):
            continue
        if _is_account_catalog(piece):
            continue
        if any(token in piece for token in _CROSS_INDUSTRY_ACCOUNTS):
            continue
        kept.append(chunk + delim)
    return "".join(kept).strip(" ，,、；;")


def _keep_operating_fragments(text: str) -> str:
    """片段级切除：空缺句／科目清单丢掉，经营成句保留（不因半段瑕疵整段扔）。"""
    kept: list[str] = []
    for sentence in re.split(r"(?<=[。；])\s*", text or ""):
        piece = sentence.strip()
        if not piece:
            continue
        if _is_account_catalog(piece) and not any(
            cue in piece for cue in ("围绕", "优化", "布局", "量价", "产能落地")
        ):
            continue
        if any(token in piece for token in _CROSS_INDUSTRY_ACCOUNTS):
            # 整句都是保险／银行残留：丢；若只是尾巴污染，先切除再判。
            if not any(cue in piece for cue in _NARRATIVE_CUES):
                continue
        if _has_hollow_number_slots(piece) or any(
            token in piece for token in _CROSS_INDUSTRY_ACCOUNTS
        ):
            piece = _excise_bad_clauses(piece)
        piece = piece.strip(" ，,、；;")
        if not piece:
            continue
        if not _is_usable_operating_quote(piece):
            # 切除后可能变短，但仍有明确经营线索则保留。
            if len(piece) < 16 or not _looks_like_operating_sentence(piece):
                continue
            if _has_hollow_number_slots(piece) or _is_account_catalog(piece):
                continue
        if not piece.endswith(("。", "；", "！", "？")):
            piece += "。"
        kept.append(piece)
    return "".join(kept)


def _extract_operating_narrative(text: str) -> str:
    """从拼接块里精准切出经营原文：垃圾片段丢、有效片段留。

    长电常见形态：股东转融通前缀 + 经营句 + 「净利润为人民币 亿元」空缺尾巴。
    不能因尾巴有空缺整段扔掉（否则经营解读退化成指标复读）。
    """
    cleaned = _TIP_PREFIX.sub("", text or "")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return ""
    start = -1
    for marker in _OPERATING_MARKERS:
        idx = cleaned.find(marker)
        if idx < 0:
            continue
        if start < 0 or idx < start:
            start = idx
    if start >= 0:
        cleaned = cleaned[start:]
    elif any(token in cleaned for token in _SHAREHOLDER_NOISE):
        return ""
    if any(token in cleaned for token in _SHAREHOLDER_NOISE):
        cut = min(
            (cleaned.find(token) for token in _SHAREHOLDER_NOISE if token in cleaned),
            default=-1,
        )
        if 0 <= cut < 40:
            return ""
        if cut >= 40:
            cleaned = cleaned[:cut].rstrip(" ，,、；;")
    if cleaned.startswith("前") and "股东" in cleaned[:24]:
        return ""
    return _keep_operating_fragments(cleaned)


def _reason_only(quote: str) -> str:
    """变动原因表常带比例数字与其它科目行；只保留原因叙述句。

    定量数字必须来自已校验 facts；原文引述只留定性原因，避免断字残片进正文。
    内容词（量价／存储）优先于「主要是」，防止长原文被短原因句抢占。
    """
    markers = ("量价", "存储", "出货", "报告期内", "主要是", "主要系")
    for marker in markers:
        idx = quote.find(marker)
        if idx < 0:
            continue
        tail = quote[idx:]
        # 「…所致」后常接其它科目变动行，先截到所致。
        caused = tail.find("所致")
        if caused >= 0 and caused < 80:
            tail = tail[: caused + 2]
        else:
            # 切断后续科目行（营业收入_本报告期 197.11 …）
            cut = re.search(
                r"\s+(?:营业收入|利润总额|归属于上市公司|基本每股|稀释每股|加权平均)",
                tail,
            )
            if cut:
                tail = tail[: cut.start()]
            stop = tail.find("。")
            if stop >= 0:
                tail = tail[: stop + 1]
        return _strip_quote_numbers(tail.strip(" ；;|"))
    return _strip_quote_numbers(quote)


def _strip_quote_numbers(text: str) -> str:
    cleaned = re.sub(r"-?\d[\d,，]*(?:\.\d+)?%?", " ", text or "")
    return " ".join(cleaned.split())


def _safe_truncate(text: str, limit: int) -> str:
    """截断正文引述时不从数字中间切开，避免校验把半截数字当成未核对。"""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if cut and cut[-1].isdigit() and limit < len(text) and text[limit].isdigit():
        cut = re.sub(r"[\d,，.]+$", "", cut).rstrip(" |,，、；;")
    return cut.rstrip() + "…"


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
        # akshare / 推算出处只作数值补齐，不当作「季报原文」引述进经营解读。
        filename = fact.evidence.filename or ""
        if filename.startswith(("akshare.", "derived.")):
            continue
        quote = _extract_operating_narrative(
            _plain_evidence_quote(fact.evidence.quote or "")
        )
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
    rows: list[dict] = []
    for kinds in (["text"], ["table"]):
        try:
            rows.extend(
                store.search_kb_chunks_keyword(
                    list(_EVIDENCE_KEYWORDS),
                    file_ids=file_ids,
                    top_k=8,
                    kinds=kinds,
                )
            )
        except Exception:
            logger.exception("经营解读证据召回失败 kinds=%s", kinds)
    ban = ("被合并方", "合并日", "净利润为：0", "净利润为0")
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        text = _extract_operating_narrative(
            _plain_evidence_quote(str(row.get("text") or ""))
        )
        if len(text) < 20 or any(token in text for token in ban):
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
