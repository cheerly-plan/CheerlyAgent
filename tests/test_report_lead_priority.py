"""「本期结论」出句顺序的端到端护栏。

注册表层已测过 ``lead_names`` / ``lead_order``，但没有用例证明这些顺序
真的落到正文里——P0 竖切期的 fallback 语义 bug 正是「名单没缩、正文悄悄缩」。
这里从 registry 派生期望顺序，不写死指标名单，改配置会同步改变断言。
"""

from __future__ import annotations

from ai_agent.report import registry
from ai_agent.report.compose import (
    _LEAD_BANK,
    _LEAD_MAX,
    _LEAD_NET_PROFIT,
    compose_report,
)
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan

_SUBJECT = "某某银行股份有限公司"


def _fact(name: str, value: str) -> IndicatorFact:
    return IndicatorFact(
        subject=_SUBJECT,
        period="2025Q3",
        name=name,
        value=value,
        unit="元",
        column="本报告期",
        scope="consolidated",
        evidence=Evidence(
            file_id="file-1",
            filename="某某银行2025年第三季度报告.pdf",
            page=12,
            page_end=12,
            section="主要会计数据",
            quote=f"{name} | {value}",
        ),
    )


def _bank_plan() -> ReportPlan:
    return ReportPlan(
        title=f"{_SUBJECT} 2025Q3 财报深度分析",
        kind="company",
        deliverable="markdown",
        skill_id="bank_financial_report",
        subjects=[_SUBJECT],
        period="2025Q3",
        dimensions=["同比"],
        indicators=list(_LEAD_BANK),
        outline=[OutlineItem("本期结论", "")],
        gaps=[],
    )


def _expected_names(present: set[str]) -> list[str]:
    """复刻 _conclusion_sentences 的选取规则，但名单全部来自注册表。"""
    expected: list[str] = []
    used_profit = False
    for name in _LEAD_BANK:
        if len(expected) >= _LEAD_MAX:
            break
        if name not in present:
            continue
        if name in _LEAD_NET_PROFIT and used_profit:
            continue
        if name in _LEAD_NET_PROFIT:
            used_profit = True
        expected.append(name)
    return expected


def test_conclusion_sentences_follow_bank_priority_order():
    names = list(_LEAD_BANK)
    facts = [_fact(name, str(100 + i)) for i, name in enumerate(names)]
    document = compose_report(_bank_plan(), facts)

    section = document.markdown.split("## 本期结论", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.strip()]
    expected = _expected_names(set(names))

    assert _LEAD_BANK == registry.ordering_list("lead_bank")
    assert len(lines) == len(expected) == min(_LEAD_MAX, len(names))
    for line, name in zip(lines, expected):
        assert f"{name}为 " in line, f"结论行 {line!r} 未按优先级输出 {name}"


def test_conclusion_skips_duplicate_profit_names():
    """净利润系只出一次，且优先取优先级里更靠前的那个。"""
    names = ["营业收入", "归属于母公司所有者的净利润", "净利润", "净息差"]
    facts = [_fact(name, "100") for name in names]
    document = compose_report(_bank_plan(), facts)
    section = document.markdown.split("## 本期结论", 1)[1].split("\n## ", 1)[0]

    expected = _expected_names(set(names))
    assert expected == ["营业收入", "归属于母公司所有者的净利润", "净息差"]
    assert "净利润为 " not in section.replace("归属于母公司所有者的净利润为 ", "")
    for name in expected:
        assert f"{name}为 " in section
