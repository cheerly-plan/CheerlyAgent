"""呈现规则：按 Features 条件选句，不按公司名。

最小切片：
- 同比：基期为负或本期亏损 → 扩大／收窄／扭亏句式；否则常规「增减／幅度」
- 权益占比：增量够大才写占比；否则豁免并只列科目增减
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ai_agent.report.compare import Change
from ai_agent.report.features import Features, loss_framing


@dataclass(frozen=True)
class Rule:
    """when 为真才 render；返回 None 表示本规则不产出（交给下一条）。"""

    when: Callable[[Features], bool]
    render: Callable[..., str | None]

    def apply(self, features: Features, **ctx) -> str | None:
        if not self.when(features):
            return None
        return self.render(features=features, **ctx)


def render_yoy_rate_bit(current_value: str, change: Change) -> str:
    """结论节用的短同比标签，与正文同一套方向规则。

    有亏损／扭亏框架时返回「亏损扩大 46.66%」；否则「同比 34.31%」。
    无幅度则返回空串。
    """
    if not change.rate:
        return ""
    framing = loss_framing(current_value, change.base_value)
    if framing:
        # 方向词已表达增减，幅度去符号，避免「亏损收窄 -48.95%」打架。
        rate = str(change.rate).lstrip("+-")
        return f"{framing} {rate}"
    return f"同比 {change.rate}"


def render_yoy_sentence(
    *,
    prefix: str,
    name: str,
    period: str,
    current_value: str,
    unit: str,
    change: Change,
    note: str,
    features: Features,
) -> str:
    """同比句：亏损／基期负走方向词；否则走常规增减。"""
    framing = loss_framing(current_value, change.base_value)
    if framing:
        # 至少一端为负：用扩大／收窄／扭亏，避免把亏损变化写成「同比 +x%」增长叙事。
        rate_bit = f"，幅度 {change.rate}" if change.rate else ""
        # change.delta / base_value 已带单位文案时不再叠 unit；仅本期绝对值用 unit 拼一次。
        delta_bit = f"，差额 {change.delta}" if change.delta else ""
        base_shown = change.base_value
        if base_shown and unit and not str(base_shown).endswith(unit):
            base_shown = f"{base_shown}{unit}"
        cur_shown = current_value
        if cur_shown and unit and not str(cur_shown).endswith(unit):
            cur_shown = f"{cur_shown}{unit}"
        note_bit = f"（{note}）" if note else ""
        return (
            f"{prefix}{name}本期为 {cur_shown}，"
            f"较 {change.base_period}（{base_shown}）{framing}"
            f"{rate_bit}{delta_bit}{note_bit}。"
        )
    return _default_yoy(
        prefix=prefix,
        name=name,
        period=period,
        change=change,
        note=note,
        reported_only=change.source == "reported",
    )


def _default_yoy(
    *,
    prefix: str,
    name: str,
    period: str,
    change: Change,
    note: str,
    reported_only: bool,
) -> str:
    if reported_only:
        extra = f"（{note}）" if note else ""
        return f"{prefix}{name} {period} 同比 {change.rate}{extra}。"
    tail = f"，幅度 {change.rate}" if change.rate else ""
    extra = f"（{note}）" if note else ""
    return (
        f"{prefix}{name}较 {change.base_period} 同比增减 {change.delta}{tail}{extra}。"
    )


def is_equity_row(name: str) -> bool:
    """归母／合并权益类科目名判定；compose／features／validate 共用。"""
    text = name or ""
    return "所有者权益" in text or "净资产" in text


def equity_share_applicable(features: Features) -> bool:
    return features.equity_growth_large


def equity_share_exemption_line(features: Features) -> str:
    delta = features.equity_delta_text or "过小"
    return (
        f"归母权益增量 {delta} 相对构成科目变动量级过小（未达可解释占比阈值），"
        "占比不适用；下表仅列科目期末／上年末增减，不写占权益增量比重。"
    )
