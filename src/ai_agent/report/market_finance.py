"""结构化财务摘要（akshare），用于补齐 PDF 抽不到的比率类指标。

边界：
* 默认只补「缺口」——知识库已抽到的同公司同指标不覆盖，保留 PDF 出处。
* 半导体赛道报告的「毛利率」例外：统一用财务摘要累计口径覆盖正文单季叙述，
  避免与同表其它公司混口径排序。
* 每条事实带可核对 quote（指标名、报告期列、原值）；数字路径仍走 validate。
* 研发费用率由利润表 RESEARCH_EXPENSE / TOTAL_OPERATE_INCOME 计算，quote 写明两端。
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

from ai_agent.core.logging import get_logger
from ai_agent.report import registry
from ai_agent.report.basis import CUMULATIVE, basis_of, cumulative_months
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.parse import ReportPlan
from ai_agent.report.skills import get_skill

logger = get_logger(__name__)

_SOURCE = "akshare.stock_financial_abstract"
_SOURCE_RD = "akshare.stock_profit_sheet_by_report_em"

# 抽象表指标名 → 研报指标名
_ABSTRACT_MAP: tuple[tuple[str, str, str], ...] = registry.abstract_map()

_AMOUNT_INDICATORS = registry.names_with("amount_abstract")
# 半导体行业横比时，毛利率一律用财务摘要累计列，覆盖季报正文「本季」叙述。
_SEMICONDUCTOR_PREFER_MARKET = registry.names_with("prefer_market")
# 利润表推算的那个比率指标名由配置给定（新增同类指标不用改本模块）。
_RND_RATIO_INDICATOR = registry.ordering_str("rnd_ratio_indicator")


def period_column(period: str) -> str | None:
    """2025Q3 → 20250930；不认识的报告期返回 None。"""
    text = (period or "").strip().upper()
    match = re.fullmatch(r"(20\d{2})Q([1-4])", text)
    if not match:
        return None
    year = int(match.group(1))
    quarter = int(match.group(2))
    month = {1: 3, 2: 6, 3: 9, 4: 12}[quarter]
    day = {3: 31, 6: 30, 9: 30, 12: 31}[month]
    return f"{year}{month:02d}{day:02d}"


def market_code(stock_code: str) -> str:
    """6 位代码 → 东方财富利润表前缀 SH/SZ。"""
    code = re.sub(r"\D", "", stock_code or "")
    if len(code) != 6:
        return ""
    if code.startswith(("5", "6", "9")):
        return f"SH{code}"
    return f"SZ{code}"


def supplement_facts_from_market(
    plan: ReportPlan,
    facts: list[IndicatorFact],
    missing: list[str],
) -> tuple[list[IndicatorFact], list[str]]:
    """用 akshare 补计划指标里仍缺的项。

    优先 skill 宇宙名单（含代码）；公司单报没有名单时，按 plan.subjects
    走巨潮名称解析拿证券代码。
    """
    skill = get_skill(plan.skill_id) if plan.skill_id else None
    if not plan.period:
        return facts, missing
    roster = _market_roster(plan, skill)
    if not roster:
        return facts, missing
    wanted = list(plan.indicators) or list(
        registry.ordering_list("market_default_indicators")
    )
    col = period_column(plan.period)
    if not col:
        return facts, missing
    prefer_market = (
        registry.names_with(
            "prefer_market", fallback=_SEMICONDUCTOR_PREFER_MARKET
        )
        if plan.skill_id == "semiconductor_industry_report"
        else frozenset()
    )

    have = {
        (_canon_subject(fact.subject), fact.name)
        for fact in facts
        if fact.period == plan.period
    }
    added: list[IndicatorFact] = []
    replaced: set[tuple[str, str]] = set()
    for name, code in roster:
        for indicator in wanted:
            key = (_canon_subject(name), indicator)
            already = any(
                _canon_subject(subject) == key[0] and ind == indicator
                for subject, ind in have
            )
            if already and indicator not in prefer_market:
                continue
            if already and _already_from_market(facts, name, indicator, plan.period):
                continue
            fact = _fetch_one(name, code, plan.period, col, indicator)
            if fact is None:
                continue
            if already:
                replaced.add(key)
            added.append(fact)
            have.add(key)

    if not added:
        return facts, missing

    kept = [
        fact
        for fact in facts
        if not (
            fact.period == plan.period
            and (_canon_subject(fact.subject), fact.name) in replaced
        )
    ]
    filled_names = {(fact.subject, fact.name) for fact in added}
    pruned = [
        gap
        for gap in missing
        if not any(
            subject in gap and name in gap for subject, name in filled_names
        )
    ]
    for fact in added:
        action = "覆盖" if (_canon_subject(fact.subject), fact.name) in replaced else "补齐"
        logger.info(
            "akshare %s %s %s %s=%s%s",
            action,
            fact.subject,
            fact.period,
            fact.name,
            fact.value,
            fact.unit,
        )
    return [*kept, *added], pruned


def _market_roster(
    plan: ReportPlan, skill
) -> list[tuple[str, str]]:
    """返回 (展示名, 6 位代码)。行业 skill 用宇宙名单；公司单报用巨潮解析。"""
    if skill is not None and skill.companies:
        rows: list[tuple[str, str]] = []
        for name, code, _segment, _aka in skill.companies:
            if plan.subjects and not any(
                name == subject or name in subject or subject in name
                for subject in plan.subjects
            ):
                continue
            rows.append((name, code))
        return rows
    if not plan.subjects:
        return []
    try:
        from ai_agent.kb import cninfo
    except Exception:
        return []
    rows = []
    for subject in plan.subjects:
        stock = cninfo.resolve_code_by_name(subject)
        if stock is None or not getattr(stock, "code", ""):
            continue
        rows.append((subject, str(stock.code)))
    return rows


def _already_from_market(
    facts: list[IndicatorFact], subject: str, indicator: str, period: str
) -> bool:
    """已有同指标且出处是 akshare 时不再重复拉。"""
    for fact in facts:
        if fact.period != period or fact.name != indicator:
            continue
        if _canon_subject(fact.subject) != _canon_subject(subject):
            continue
        filename = fact.evidence.filename or ""
        return filename.startswith("akshare.")
    return False


def _canon_subject(name: str) -> str:
    text = (name or "").strip()
    for alias in ("股份有限公司", "有限公司", "集团"):
        text = text.replace(alias, "")
    return text


def _fetch_one(
    subject: str,
    code: str,
    period: str,
    col: str,
    indicator: str,
) -> IndicatorFact | None:
    try:
        if indicator == _RND_RATIO_INDICATOR:
            return _rd_rate_fact(subject, code, period, col)
        return _abstract_fact(subject, code, period, col, indicator)
    except Exception:
        logger.exception("akshare 拉取失败 code=%s indicator=%s", code, indicator)
        return None


@lru_cache(maxsize=32)
def _load_abstract(code: str):
    import akshare as ak

    return ak.stock_financial_abstract(symbol=code)


@lru_cache(maxsize=32)
def _load_profit(em_code: str):
    import akshare as ak

    return ak.stock_profit_sheet_by_report_em(symbol=em_code)


def _abstract_fact(
    subject: str,
    code: str,
    period: str,
    col: str,
    indicator: str,
) -> IndicatorFact | None:
    # 已登记指标以注册表为准；未登记的回退内置映射。
    entry = registry.abstract_entry(indicator) or next(
        (item for item in _ABSTRACT_MAP if item[1] == indicator), None
    )
    if entry is None:
        return None
    source_name, _dst, unit = entry
    df = _load_abstract(code)
    if col not in df.columns or "指标" not in df.columns:
        return None
    rows = df[df["指标"].astype(str) == source_name]
    if rows.empty:
        rows = df[df["指标"].astype(str).str.contains(source_name, na=False)]
    if rows.empty:
        return None
    raw = rows.iloc[0][col]
    formatted = _format_value(raw, unit=unit, amount=indicator in _AMOUNT_INDICATORS)
    if formatted is None:
        return None
    value, display_unit = formatted
    # 财务摘要报告期列是期末累计口径（Q3=前三季度），毛利率等同列比率也是累计。
    is_amount = indicator in _AMOUNT_INDICATORS
    is_ytd_ratio = registry.flag(
        indicator, "ytd_ratio", default=False
    )
    if is_ytd_ratio:
        basis, months = CUMULATIVE, cumulative_months(period)
        cumulative = True
    else:
        basis, months = basis_of(
            period,
            cumulative=is_amount,
            name=indicator,
            unit=display_unit,
            value=value,
        )
        cumulative = is_amount
    # quote 必须含正文采用的展示值，否则 validate 会报「数值未出现在原文片段」
    scope_note = "前三季度累计" if is_ytd_ratio and period.endswith("Q3") else "报告期累计"
    quote = (
        f"{source_name} 报告期列{col} 原值={raw} 采用={value}{display_unit}"
        f"（{_SOURCE}，{scope_note}口径）"
    )
    return IndicatorFact(
        subject=subject,
        period=period,
        name=indicator,
        value=value,
        unit=display_unit,
        column="结构化财务摘要",
        scope="consolidated",
        cumulative=cumulative,
        basis=basis,
        coverage_months=months,
        evidence=Evidence(
            file_id=_synthetic_id(code, period, indicator),
            filename=_SOURCE,
            page=None,
            page_end=None,
            section="财务摘要",
            quote=quote,
        ),
        reported_yoy="",
    )


def _rd_rate_fact(
    subject: str, code: str, period: str, col: str
) -> IndicatorFact | None:
    em = market_code(code)
    if not em:
        return None
    df = _load_profit(em)
    if "REPORT_DATE" not in df.columns:
        return None
    # col 20250930 → 2025-09-30
    stamp = f"{col[0:4]}-{col[4:6]}-{col[6:8]}"
    dated = df.copy()
    dated["_d"] = dated["REPORT_DATE"].astype(str)
    rows = dated[dated["_d"].str.startswith(stamp)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    revenue = row.get("TOTAL_OPERATE_INCOME")
    research = row.get("RESEARCH_EXPENSE")
    try:
        rev = float(revenue)
        rd = float(research)
    except (TypeError, ValueError):
        return None
    if rev <= 0 or rd != rd:  # NaN check
        return None
    rate = rd / rev * 100.0
    value = f"{rate:.2f}"
    # 利润表单期行是报告期末累计（年初至报告期末），费用率同属累计口径。
    basis, months = CUMULATIVE, cumulative_months(period)
    scope_note = "前三季度累计" if period.endswith("Q3") else "报告期累计"
    quote = (
        f"RESEARCH_EXPENSE={research} / TOTAL_OPERATE_INCOME={revenue} "
        f"报告期{stamp} 采用研发费用率={value}%（{_SOURCE_RD}，{scope_note}口径）"
    )
    return IndicatorFact(
        subject=subject,
        period=period,
        name=_RND_RATIO_INDICATOR,
        value=value,
        unit="%",
        column="利润表推算研发费用率",
        scope="consolidated",
        cumulative=True,
        basis=basis,
        coverage_months=months,
        evidence=Evidence(
            file_id=_synthetic_id(code, period, _RND_RATIO_INDICATOR),
            filename=_SOURCE_RD,
            page=None,
            page_end=None,
            section="利润表",
            quote=quote,
        ),
    )


def _format_value(
    raw: object, *, unit: str, amount: bool
) -> tuple[str, str] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "false", "--", "-"}:
        return None
    try:
        number = float(text.replace(",", "").replace("%", ""))
    except ValueError:
        return None
    if number != number:  # NaN
        return None
    if unit == "%":
        return f"{number:.2f}".rstrip("0").rstrip("."), "%"
    if amount:
        # 摘要里是元；整数去小数，保留千分位
        if abs(number - round(number)) < 1e-6:
            return f"{int(round(number)):,}", "元"
        return f"{number:,.2f}", "元"
    return f"{number:.4g}", unit


def _synthetic_id(code: str, period: str, indicator: str) -> str:
    digest = hashlib.sha256(f"akshare:{code}:{period}:{indicator}".encode()).hexdigest()
    # 形如 UUID 便于溯源表展示；前 8 位作正文锚点
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
