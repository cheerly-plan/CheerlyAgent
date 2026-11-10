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
