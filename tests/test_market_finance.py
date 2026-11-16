"""akshare 结构化补齐：只填缺口，不覆盖已有 PDF 抽取。"""

from __future__ import annotations

import pandas as pd

from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.market_finance import period_column, supplement_facts_from_market
from ai_agent.report.parse import OutlineItem, ReportPlan


def test_period_column_maps_quarter_end():
    assert period_column("2025Q3") == "20250930"
    assert period_column("2024Q1") == "20240331"
    assert period_column("bad") is None


def test_supplement_fills_missing_margin_from_abstract(monkeypatch):
    abstract = pd.DataFrame(
        {
            "指标": ["毛利率", "营业总收入", "扣非净利润"],
            "20250930": [41.406925, 27301379656.81, 5102078327.94],
        }
    )
    profit = pd.DataFrame(
        {
            "REPORT_DATE": ["2025-09-30 00:00:00"],
            "TOTAL_OPERATE_INCOME": [27301379656.81],
            "RESEARCH_EXPENSE": [2_730_137_965.0],
        }
    )
    monkeypatch.setattr(
        "ai_agent.report.market_finance._load_abstract", lambda code: abstract
    )
    monkeypatch.setattr(
        "ai_agent.report.market_finance._load_profit", lambda em: profit
    )

    plan = ReportPlan(
        title="t",
        kind="industry",
        deliverable="pdf",
        skill_id="semiconductor_industry_report",
        industry="半导体",
        subjects=["北方华创"],
        period="2025Q3",
        indicators=["营业收入", "毛利率", "研发费用率"],
        outline=[OutlineItem("组内对比", "")],
    )
    facts, missing = supplement_facts_from_market(plan, [], ["北方华创 没有抽到 2025Q3 毛利率"])
    names = {fact.name: fact for fact in facts}
    assert "毛利率" in names
    assert names["毛利率"].value.startswith("41.4")
    assert names["毛利率"].unit == "%"
    assert names["毛利率"].value in names["毛利率"].evidence.quote
    assert "akshare" in names["毛利率"].evidence.filename
    assert "研发费用率" in names
    assert float(names["研发费用率"].value) == 10.0
    assert names["研发费用率"].value in names["研发费用率"].evidence.quote


def test_supplement_does_not_override_non_prefer_indicator(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.market_finance._load_abstract",
        lambda code: (_ for _ in ()).throw(AssertionError("should not fetch")),
    )
    existing = IndicatorFact(
        subject="北方华创",
        period="2025Q3",
        name="营业收入",
        value="100",
        unit="元",
        column="正文",
        scope="",
        evidence=Evidence("f1", "北方华创季报.pdf", 2, 2, "经营", "营业收入100元"),
    )
    plan = ReportPlan(
        title="t",
        kind="industry",
        deliverable="pdf",
        skill_id="semiconductor_industry_report",
        subjects=["北方华创"],
        period="2025Q3",
        indicators=["营业收入"],
        outline=[],
    )
    facts, _missing = supplement_facts_from_market(plan, [existing], [])
    assert len(facts) == 1
    assert facts[0].value == "100"


def test_supplement_overrides_narrative_margin_with_abstract(monkeypatch):
    abstract = pd.DataFrame(
        {
            "指标": ["毛利率"],
            "20250930": [23.151126],
        }
    )
    monkeypatch.setattr(
        "ai_agent.report.market_finance._load_abstract", lambda code: abstract
    )
    monkeypatch.setattr(
        "ai_agent.report.market_finance._load_profit",
        lambda em: (_ for _ in ()).throw(AssertionError("rd not needed")),
    )
    existing = IndicatorFact(
        subject="中芯国际",
        period="2025Q3",
        name="毛利率",
        value="25.5",
        unit="%",
        column="经营情况简要分析（正文叙述）",
        scope="",
        evidence=Evidence(
            "f1",
            "中芯国际季报.pdf",
            2,
            2,
            "经营",
            "三季度公司毛利率为25.5%",
        ),
    )
    plan = ReportPlan(
        title="t",
        kind="industry",
        deliverable="pdf",
        skill_id="semiconductor_industry_report",
        subjects=["中芯国际"],
        period="2025Q3",
        indicators=["毛利率"],
        outline=[],
    )
    facts, _missing = supplement_facts_from_market(plan, [existing], [])
    assert len(facts) == 1
    assert facts[0].value.startswith("23.1")
    assert "akshare" in facts[0].evidence.filename


def test_supplement_fetches_ratio_base_period(monkeypatch):
    """比率类指标（毛利率）额外抽去年同期基期，让 compare 能算同比。

    复现北方华创问题：正文「毛利率从 41.41% 降至 39.44%」，基期 41.41% 不在
    事实台账，被零幻觉门禁误拦。抽取层应把 2024Q3 的毛利率一并抽进来。
    """
    abstract = pd.DataFrame(
        {
            "指标": ["毛利率", "营业总收入"],
            "20240930": [41.406925, 25000000000.0],
            "20250930": [39.444061, 27301379656.81],
        }
    )
    monkeypatch.setattr(
        "ai_agent.report.market_finance._load_abstract", lambda code: abstract
    )
    monkeypatch.setattr(
        "ai_agent.report.market_finance._load_profit",
        lambda em: (_ for _ in ()).throw(AssertionError("rd not needed")),
    )
    plan = ReportPlan(
        title="t",
        kind="company",
        deliverable="pdf",
        skill_id="",
        subjects=["北方华创"],
        period="2025Q3",
        indicators=["毛利率", "营业收入"],
        outline=[],
    )
    facts, _missing = supplement_facts_from_market(plan, [], [])

    by_name_period = {(f.name, f.period): f for f in facts}
    # 毛利率：本期 + 去年同期基期都在
    assert ("毛利率", "2025Q3") in by_name_period
    assert ("毛利率", "2024Q3") in by_name_period
    assert by_name_period[("毛利率", "2025Q3")].value.startswith("39.4")
    assert by_name_period[("毛利率", "2024Q3")].value.startswith("41.4")
    # 基期 fact 有出处，且 period 正确
    base = by_name_period[("毛利率", "2024Q3")]
    assert "akshare" in base.evidence.filename
    assert base.period == "2024Q3"
    # 金额类指标（营业收入）不额外抽基期
    assert ("营业收入", "2024Q3") not in by_name_period
    assert ("营业收入", "2025Q3") in by_name_period


def test_supplement_ratio_base_period_skips_when_column_absent(monkeypatch):
    """基期列缺失时（无去年同期数据），安全跳过，不报错、不产基期 fact。"""
    abstract = pd.DataFrame(
        {
            "指标": ["毛利率"],
            "20250930": [39.444061],
        }
    )
    monkeypatch.setattr(
        "ai_agent.report.market_finance._load_abstract", lambda code: abstract
    )
    monkeypatch.setattr(
        "ai_agent.report.market_finance._load_profit",
        lambda em: (_ for _ in ()).throw(AssertionError("rd not needed")),
    )
    plan = ReportPlan(
        title="t",
        kind="company",
        deliverable="pdf",
        skill_id="",
        subjects=["北方华创"],
        period="2025Q3",
        indicators=["毛利率"],
        outline=[],
    )
    facts, _missing = supplement_facts_from_market(plan, [], [])
    periods = {f.period for f in facts}
    assert periods == {"2025Q3"}
