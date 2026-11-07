"""按公司类型路由研报指标集。

铁律：先分类，再取模板；模板外的指标不抽、也不写「没抽到」。
银行看息差／资产质量；Fabless 看毛利率／研发／存货；设备看合同负债／存货／在手订单；
Foundry 看稼动率／CAPEX——一套制造业模板通吃，必然出跨行业污染与假缺失。

粗粒度 ``industry_of`` 仍只服务 skill 路由（bank vs generic）；
细粒度 ``company_type_of`` 决定抽哪些指标、排除哪些污染项。
指标名一律用报表原文术语，才能和原文对得上。
"""

from __future__ import annotations

from dataclasses import dataclass

# —— skill 路由用的粗行业（与 SKILL.md industries / indicator_set 对齐）——
BANK = "bank"
GENERIC = "generic"
SEMICONDUCTOR = "semiconductor"

# —— 指标路由用的细类型 ——
FABLESS = "fabless"
FOUNDRY = "foundry"
EQUIPMENT = "equipment"
PACKAGING = "packaging"
MANUFACTURING = "manufacturing"
SOFTWARE = "software"
REAL_ESTATE = "real_estate"
CYCLICAL = "cyclical"

# 半导体赛道名 → 细类型（与 semiconductor_industry_report 的 companies 赛道一致）
_SEGMENT_TO_TYPE = {
    "设计": FABLESS,
    "制造": FOUNDRY,
    "设备": EQUIPMENT,
    "封测": PACKAGING,
}

# 银行专属：任何非银行模板都必须 exclude，防止观察句／缺口污染。
_BANK_ONLY = (
    "净息差",
    "拨备前利润",
    "PPOP",
    "不良贷款率",
    "不良贷款拨备覆盖率",
    "资本充足率",
    "核心一级资本充足率",
    "成本收入比",
    "利息净收入",
    "非利息收入",
    "非利息收入占比",
    "发放贷款和垫款总额",
    "吸收存款总额",
)

# Fabless 无自有产线，不看稼动率；银行不看毛利率／存货。
_BANK_EXCLUDE = ("毛利率", "存货周转率", "研发费用率", "产能利用率", "合同负债")
_FABLESS_EXCLUDE = (*_BANK_ONLY, "产能利用率")
_FOUNDRY_EXCLUDE = (*_BANK_ONLY, "大客户集中度")
# 设备类订单先行：不看稼动率；存货周转／资产负债率让位于合同负债／存货／在手订单。
_EQUIPMENT_EXCLUDE = (*_BANK_ONLY, "产能利用率", "存货周转率", "资产负债率")
_PACKAGING_EXCLUDE = (*_BANK_ONLY, "大客户集中度")
_GENERIC_EXCLUDE = _BANK_ONLY


@dataclass(frozen=True)
class IndicatorProfile:
    """一类公司该看／不该看的核心指标（5–8 个，不是三张表全抄）。"""

    indicators: tuple[str, ...]
    exclude: tuple[str, ...] = ()


# 核心映射：industry_type → indicators / exclude
_PROFILES: dict[str, IndicatorProfile] = {
    BANK: IndicatorProfile(
        indicators=(
            "营业收入",
            "净利润",
            "利润总额",
            "归属于母公司所有者的净利润",
            "利息净收入",
            "非利息收入",
            "非利息收入占比",
            "手续费及佣金净收入",
            "业务及管理费",
            "成本收入比",
            "资产减值损失",
            "净息差",
            "平均总资产回报率",
            "不良贷款率",
            "不良贷款拨备覆盖率",
            "加权平均净资产收益率",
            "发放贷款和垫款总额",
            "吸收存款总额",
            "资产总计",
            "核心一级资本充足率",
            "资本充足率",
        ),
        exclude=_BANK_EXCLUDE,
    ),
    FABLESS: IndicatorProfile(
        indicators=(
            "营业收入",
            "扣非净利润",
            "毛利率",
            "研发费用率",
            "存货周转率",
            "大客户集中度",
            "经营活动产生的现金流量净额",
        ),
        exclude=_FABLESS_EXCLUDE,
    ),
    FOUNDRY: IndicatorProfile(
        indicators=(
            "营业收入",
            "扣非净利润",
            "毛利率",
            "产能利用率",
            "折旧",
            "资本开支",
            "存货",
            "经营活动产生的现金流量净额",
        ),
        exclude=_FOUNDRY_EXCLUDE,
    ),
    EQUIPMENT: IndicatorProfile(
        # 订单先行：合同负债／存货／在手订单优先于研发费用率进组内横比前六列。
        indicators=(
            "营业收入",
            "扣非净利润",
            "毛利率",
            "合同负债",
            "存货",
            "在手订单",
            "研发费用率",
            "经营活动产生的现金流量净额",
        ),
        exclude=_EQUIPMENT_EXCLUDE,
    ),
    PACKAGING: IndicatorProfile(
        indicators=(
            "营业收入",
            "扣非净利润",
            "毛利率",
            "产能利用率",
            "资本开支",
            "经营活动产生的现金流量净额",
        ),
        exclude=_PACKAGING_EXCLUDE,
    ),
    MANUFACTURING: IndicatorProfile(
        indicators=(
            "营业收入",
            "扣非净利润",
            "毛利率",
            "存货周转率",
            "经营活动产生的现金流量净额",
            "产能利用率",
            "资产负债率",
        ),
        exclude=_GENERIC_EXCLUDE,
    ),
    SOFTWARE: IndicatorProfile(
        indicators=(
            "营业收入",
            "扣非净利润",
            "毛利率",
            "研发费用率",
            "经营活动产生的现金流量净额",
            "合同负债",
        ),
        exclude=(*_BANK_ONLY, "存货周转率", "产能利用率", "资本开支"),
    ),
    REAL_ESTATE: IndicatorProfile(
        indicators=(
            "营业收入",
            "扣非净利润",
            "毛利率",
            "合同负债",
            "经营活动产生的现金流量净额",
            "资产负债率",
        ),
        exclude=(*_BANK_ONLY, "研发费用率", "产能利用率"),
    ),
    CYCLICAL: IndicatorProfile(
        indicators=(
            "营业收入",
            "扣非净利润",
            "毛利率",
            "产能利用率",
            "资本开支",
            "存货",
            "经营活动产生的现金流量净额",
        ),
        exclude=(*_BANK_ONLY, "研发费用率", "大客户集中度"),
    ),
    # 行业横比共用集：各赛道都能对上的少数指标；组内细项由 segment 模板补。
    SEMICONDUCTOR: IndicatorProfile(
        indicators=(
            "营业收入",
            "扣非净利润",
            "毛利率",
            "研发费用率",
            "经营活动产生的现金流量净额",
        ),
        exclude=_BANK_ONLY,
    ),
    GENERIC: IndicatorProfile(
        indicators=(
            "营业收入",
            "扣非净利润",
            "毛利率",
            "存货周转率",
            "经营活动产生的现金流量净额",
            "资产负债率",
        ),
        exclude=_GENERIC_EXCLUDE,
    ),
}

# 需求文本里可能出现的简称 → 报表正式指标名。长词放前面避免被短词吃掉。
_ALIASES: tuple[tuple[str, str], ...] = (
    ("加权平均净资产收益率", "加权平均净资产收益率"),
    ("核心一级资本充足率", "核心一级资本充足率"),
    ("不良贷款拨备覆盖率", "不良贷款拨备覆盖率"),
    ("拨备覆盖率", "不良贷款拨备覆盖率"),
    ("不良贷款率", "不良贷款率"),
    ("不良率", "不良贷款率"),
    ("利息净收入", "利息净收入"),
    ("净利息收入", "利息净收入"),
    ("非利息收入占比", "非利息收入占比"),
    ("非利息收入", "非利息收入"),
    ("非息收入", "非利息收入"),
    ("非息占比", "非利息收入占比"),
    ("手续费及佣金净收入", "手续费及佣金净收入"),
    ("业务及管理费", "业务及管理费"),
    ("成本收入比", "成本收入比"),
    ("资产减值损失", "资产减值损失"),
    ("发放贷款和垫款总额", "发放贷款和垫款总额"),
    ("吸收存款总额", "吸收存款总额"),
    ("归属于母公司所有者的净利润", "归属于母公司所有者的净利润"),
    ("经营活动产生的现金流量净额", "经营活动产生的现金流量净额"),
    ("经营现金流", "经营活动产生的现金流量净额"),
    ("存货周转率", "存货周转率"),
    ("存货周转", "存货周转率"),
    ("大客户集中度", "大客户集中度"),
    ("客户集中度", "大客户集中度"),
    ("产能利用率", "产能利用率"),
    ("稼动率", "产能利用率"),
    ("合同负债", "合同负债"),
    ("在手订单", "在手订单"),
    ("新签订单", "在手订单"),
    ("存货", "存货"),
    ("资产负债率", "资产负债率"),
    ("资本开支", "资本开支"),
    ("CAPEX", "资本开支"),
    ("利润总额", "利润总额"),
    ("净息差", "净息差"),
    ("资本充足率", "资本充足率"),
    ("净资产收益率", "加权平均净资产收益率"),
    ("ROE", "加权平均净资产收益率"),
    ("ROA", "平均总资产回报率"),
    ("NIM", "净息差"),
    ("PPOP", "拨备前利润"),
)

_NAME_HINTS: tuple[tuple[str, str], ...] = (
    ("银行", BANK),
    ("置业", REAL_ESTATE),
    ("地产", REAL_ESTATE),
    ("房地产", REAL_ESTATE),
    ("软件", SOFTWARE),
    ("互联", SOFTWARE),
    ("云", SOFTWARE),
    ("钢铁", CYCLICAL),
    ("化工", CYCLICAL),
    ("煤炭", CYCLICAL),
    ("有色", CYCLICAL),
)


def industry_of(subjects: list[str]) -> str:
    """粗行业：主体名含「银行」→ bank；否则 generic（供 skill 路由）。"""
    if any("银行" in (name or "") for name in subjects):
        return BANK
    return GENERIC


def company_type_of(subjects: list[str]) -> str:
    """细类型：决定指标模板。多主体时取第一个可判定的类型；判不出 → generic。"""
    if not subjects:
        return GENERIC
    types = [classify_subject(name) for name in subjects]
    known = [item for item in types if item != GENERIC]
    if not known:
        return GENERIC
    # 多主体且类型不一致时，保守退回 generic，避免把银行指标套到制造业。
    if len(set(known)) > 1:
        return GENERIC
    return known[0]


def classify_subject(subject: str) -> str:
    """单家公司分类：半导体花名册 → 名称关键字 → generic。不调用模型。"""
    text = (subject or "").strip()
    if not text:
        return GENERIC
    if "银行" in text:
        return BANK
    roster = _semiconductor_roster()
    for key, company_type in roster.items():
        if key == text or key in text or text in key:
            return company_type
    for token, company_type in _NAME_HINTS:
        if token in text:
            return company_type
    return GENERIC


def type_for_segment(segment: str) -> str:
    """赛道名 → 细类型；未知赛道回退 semiconductor 共用集。"""
    return _SEGMENT_TO_TYPE.get((segment or "").strip(), SEMICONDUCTOR)


def default_indicators(industry_or_type: str) -> list[str]:
    """indicator_set / 公司类型 → 默认指标列表。"""
    key = _normalize_key(industry_or_type)
    profile = _PROFILES.get(key) or _PROFILES[GENERIC]
    return list(profile.indicators)


def excluded_indicators(industry_or_type: str) -> frozenset[str]:
    """该类公司不该出现的指标（抽数／缺口／观察句一律跳过）。"""
    key = _normalize_key(industry_or_type)
    profile = _PROFILES.get(key) or _PROFILES[GENERIC]
    return frozenset(profile.exclude)


def is_excluded(industry_or_type: str, indicator: str) -> bool:
    name = (indicator or "").strip()
    if not name:
        return False
    banned = excluded_indicators(industry_or_type)
    if name in banned:
        return True
    # 简称／别名也要拦住（PPOP、拨备覆盖率…）
    canonical = normalize_indicator(name)
    return canonical in banned and canonical != name


def profile_of(industry_or_type: str) -> IndicatorProfile:
    key = _normalize_key(industry_or_type)
    return _PROFILES.get(key) or _PROFILES[GENERIC]


def normalize_indicator(text: str) -> str:
    """把需求里的简称归一到报表正式指标名；不认识就原样返回。"""
    cleaned = (text or "").strip()
    for alias, canonical in _ALIASES:
        if alias in cleaned:
            return canonical
    return cleaned


def indicator_aliases() -> tuple[tuple[str, str], ...]:
    return _ALIASES


def _normalize_key(raw: str) -> str:
    text = (raw or "").strip().lower()
    aliases = {
        "bank": BANK,
        "银行": BANK,
        "generic": GENERIC,
        "manufacturing": MANUFACTURING,
        "一般制造": MANUFACTURING,
        "制造业": MANUFACTURING,
        "semiconductor": SEMICONDUCTOR,
        "半导体": SEMICONDUCTOR,
        "fabless": FABLESS,
        "设计": FABLESS,
        "foundry": FOUNDRY,
        "制造": FOUNDRY,
        "equipment": EQUIPMENT,
        "设备": EQUIPMENT,
        "packaging": PACKAGING,
        "封测": PACKAGING,
        "software": SOFTWARE,
        "软件": SOFTWARE,
        "real_estate": REAL_ESTATE,
        "地产": REAL_ESTATE,
        "cyclical": CYCLICAL,
        "周期": CYCLICAL,
    }
    return aliases.get(text) or aliases.get(raw.strip()) or (
        raw.strip() if raw.strip() in _PROFILES else GENERIC
    )


def _semiconductor_roster() -> dict[str, str]:
    """从半导体行业 skill 的 companies 花名册推导公司→类型；失败则空。"""
    try:
        from ai_agent.report.skills import get_skill
    except Exception:
        return {}
    skill = get_skill("semiconductor_industry_report")
    if skill is None or not skill.companies:
        return {}
    out: dict[str, str] = {}
    for name, _code, segment in skill.companies:
        company_type = _SEGMENT_TO_TYPE.get((segment or "").strip())
        if company_type:
            out[name] = company_type
    return out
