"""口径（累计 / 单季 / 时点比率）作为数据的一等公民。

以前口径只体现在文案括号里，计算层看不到，于是出现「1-9 月累计减 1-3 月累计」
还报增长率、以及把百分比指标套上「百万元」单位的错误。这里统一定义口径的
判定、覆盖期数和中文标签，extract / compare / compose / pdf 全部共用。
"""

from __future__ import annotations

import re

CUMULATIVE = "cumulative"
SINGLE = "single"
POINT = "point"
# 期末余额（贷款总额、存款总额、资产总计）是时点存量，不是「累计发生额」。
# 标成 9 个月累计会让读者以为它是前三季度加总出来的，属于新的口径错误。
STOCK = "stock"

_PERIOD = re.compile(r"^((?:19|20)\d{2})(Q([1-4])|H1)?$")
_MONTHS = {"Q1": 3, "H1": 6, "Q3": 9, "Q4": 12}
_RATIO_NAME = re.compile(r"(率|比率|息差|倍数|指数)$")
_RATIO_UNIT = re.compile(r"(%|％|百分点)")
# 期末快照类比率：跨期比百分点有意义，但不受累计期数影响
_POINT_IN_TIME = ("不良贷款率", "拨备覆盖率", "资本充足率")
# 用显式名单，不要用「总额$」这类宽泛正则：「利润总额」「负债总额」是期间
# 发生额或科目合计，不是期末时点存量，套上 stock 口径等于造新错。
_STOCK_NAMES = (
    "发放贷款和垫款总额",
    "吸收存款总额",
    "不良贷款总额",
    "资产总计",
    "所有者权益合计",
    "归属于母公司所有者权益合计",
)
# 原文没写年化、我们也不替它推断的比率。同表 ROE 明确标了「(%,年化)」，
# 净息差没有——这种差异必须写进正文，不能拿外部行业均值硬比。
_UNANNOTATED_RATIO = ("净息差",)


def period_token(period: str) -> str:
    match = _PERIOD.fullmatch(period or "")
    return match.group(2) if match else ""


def cumulative_months(period: str) -> int:
    """累计报告覆盖的月份数：Q1=3 / H1=6 / Q3=9 / Q4=12。"""
    return _MONTHS.get(period_token(period), 0)


def is_ratio_indicator(name: str, unit: str, value: str) -> bool:
    """比率类指标：不良率、净息差、ROE、资本充足率等。

    银行报表的「单位：百万元人民币（百分比除外）」不能套到这些行上。
    """
    return bool(
        _RATIO_NAME.search((name or "").strip())
        or _RATIO_UNIT.search((unit or "").strip())
        or (value or "").strip().endswith(("%", "％"))
    )


def basis_of(
    period: str, *, cumulative: bool, name: str, unit: str, value: str
) -> tuple[str, int]:
    """返回 (口径, 覆盖月数)。

    比率要再分两类：不良率/资本充足率是**期末快照**（覆盖 0 个月），
    净息差/ROE/成本收入比是**年初至今期间比率**（跟随累计期数）。
    后者拿 9 个月的 ROE 减 3 个月的 ROE 同样没有意义。
    """
    if is_ratio_indicator(name, unit, value):
        if any(key in (name or "") for key in _POINT_IN_TIME):
            return POINT, 0
        return POINT, cumulative_months(period) or 3
    if (name or "").strip() in _STOCK_NAMES:
        return STOCK, 0
    if cumulative:
        return CUMULATIVE, cumulative_months(period)
    return SINGLE, 3


def ratio_unit(name: str, unit: str, value: str) -> str:
    """比率指标的单位一律收敛成 %，避免正文写出「8.70百万元」。"""
    if is_ratio_indicator(name, unit, value):
        return "%"
    return unit


def label(period: str, basis: str, coverage_months: int = 0) -> str:
    """给正文用的中文口径短语，放在指标名前，不让读者误把累计当单季。"""
    year = period[:4] if len(period) >= 4 and period[:4].isdigit() else period
    if basis == CUMULATIVE:
        return {
            3: f"{year}年一季度累计",
            6: f"{year}年上半年累计",
            9: f"{year}年前三季度累计",
            12: f"{year}年全年累计",
        }.get(coverage_months, f"{year}年累计")
    if basis == SINGLE:
        return f"{period}单季"
    if basis == POINT:
        return f"{period}时点／期间比率"
    if basis == STOCK:
        return f"{period}期末余额"
    return f"{period}（口径未标明）"


def disclosure_note(name: str) -> str:
    """原文口径交代不全时给正文的警示语；没有警示就返回空串。"""
    if (name or "").strip() in _UNANNOTATED_RATIO:
        return "原文未注明是否年化，按披露值引用，不自行折算、不与外部行业均值直接比较"
    return ""
