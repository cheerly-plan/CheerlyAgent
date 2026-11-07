"""报告状态特征：按数据算标签，不按公司名分叉。

稳定标签（industry）仍由 industry.company_type_of 提供；
本模块只算**经营状态**标签，供呈现规则路由。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ai_agent.report.compare import SeriesCompare, parse_number
from ai_agent.report.extract import IndicatorFact
from ai_agent.report.parse import ReportPlan
from ai_agent.report.supplemental import SupplementalResult, signed_delta

# 归母权益增量 / 上年末绝对值 ≥ 此阈值 → 允许写「占权益增量 X%」。
EQUITY_GROWTH_RATIO = Decimal("0.10")

_PROFIT_NAMES = (
    "扣非净利润",
    "归属于母公司所有者的净利润",
    "归属于上市公司股东的扣除非经常性损益的净利润",
    "净利润",
)


@dataclass(frozen=True)
class Features:
    """一家公司、一期报告的状态快照。"""

    industry: str = ""
    profitable: bool = True
    has_single_quarter: bool = False
    equity_growth_large: bool = False
    # 供豁免句引用；空串表示算不出权益增量。
    equity_delta_text: str = ""
    equity_prior: str = ""
    equity_end: str = ""


def compute_features(
    plan: ReportPlan,
    facts: list[IndicatorFact],
    views: SupplementalResult | None = None,
    series: list[SeriesCompare] | None = None,
) -> Features:
    """只读已抽取事实／视图，不调用模型、不按公司名分支。"""
    views = views or SupplementalResult()
    period = plan.period
    current = [fact for fact in facts if fact.period == period]
    industry = getattr(plan, "company_type", "") or ""
    profitable = _profitable(current)
    has_single = bool(views.quarters) or any(
        (fact.column or "").startswith("本报告期") or fact.column == "本报告期"
        for fact in current
    )
    large, delta_text, prior, end = _equity_growth(views)
    return Features(
        industry=industry,
        profitable=profitable,
        has_single_quarter=has_single,
        equity_growth_large=large,
        equity_delta_text=delta_text,
        equity_prior=prior,
        equity_end=end,
    )


def loss_framing(current_value: str, base_value: str) -> str | None:
    """两端有绝对值且至少一端为负时，返回「扩大／收窄／扭亏／转亏」；否则 None。

    不看公司名：纯数字关系。
    """
    cur = parse_number(current_value)
    base = parse_number(base_value)
    if not cur or not base:
        return None
    c, b = cur[0], base[0]
    if c >= 0 and b >= 0:
        return None
    if b < 0 < c:
        return "扭亏为盈"
    if c < 0 < b:
        return "由盈转亏"
    # 两端均为负：比亏损绝对值
    if abs(c) > abs(b):
        return "亏损扩大"
    if abs(c) < abs(b):
        return "亏损收窄"
    return "亏损持平"


def _profitable(current: list[IndicatorFact]) -> bool:
    for name in _PROFIT_NAMES:
        for fact in current:
            if fact.name != name:
                continue
            parsed = parse_number(fact.value)
            if parsed is None:
                continue
            return parsed[0] > 0
    # 没有利润指标时不当成亏损叙事，避免误伤。
    return True


def _equity_growth(
    views: SupplementalResult,
) -> tuple[bool, str, str, str]:
    equity = next(
        (
            item
            for item in views.year_ends
            if "所有者权益" in item.name or "净资产" in item.name
        ),
        None,
    )
    if equity is None or not equity.prior_value:
        return False, "", "", ""
    delta = signed_delta(equity.value, equity.prior_value)
    prior = parse_number(equity.prior_value)
    if delta is None or prior is None or prior[0] == 0:
        return False, "", equity.prior_value, equity.value
    from ai_agent.report.supplemental import format_money_delta

    ratio = abs(delta) / abs(prior[0])
    return (
        ratio >= EQUITY_GROWTH_RATIO,
        format_money_delta(delta),
        equity.prior_value,
        equity.value,
    )
