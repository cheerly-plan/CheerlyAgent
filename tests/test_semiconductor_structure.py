"""半导体经营解读：用已授权比率写梯队与结构关系。"""

from ai_agent.report.analyze import analyze_operations
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan


def _fact(subject: str, name: str, value: str) -> IndicatorFact:
    return IndicatorFact(
        subject=subject,
        period="2025Q3",
        name=name,
        value=value,
        unit="%",
        column="结构化财务摘要",
        scope="consolidated",
        evidence=Evidence(
            file_id=f"{subject}-{name}",
            filename="akshare.stock_financial_abstract",
            page=None,
            page_end=None,
            section="财务摘要",
            quote=f"{name} 采用={value}%",
        ),
    )


def test_semiconductor_structure_writes_ladder_not_restatement():
    plan = ReportPlan(
        title="半导体行业 2025Q3 财报对比分析（赛道分组版）",
        kind="industry",
        deliverable="markdown",
        skill_id="semiconductor_industry_report",
        industry="半导体",
        subjects=["中芯国际", "北方华创", "中微公司", "韦尔股份", "兆易创新", "长电科技", "寒武纪"],
        period="2025Q3",
        indicators=["毛利率", "研发费用率"],
        outline=[OutlineItem("经营解读", "")],
    )
    facts = [
        _fact("寒武纪", "毛利率", "55.29"),
        _fact("北方华创", "毛利率", "41.41"),
        _fact("中微公司", "毛利率", "39.1"),
        _fact("兆易创新", "毛利率", "38.59"),
        _fact("韦尔股份", "毛利率", "30.43"),
        _fact("中芯国际", "毛利率", "25.5"),
        _fact("长电科技", "毛利率", "13.74"),
        _fact("中微公司", "研发费用率", "22.25"),
        _fact("寒武纪", "研发费用率", "18.29"),
        _fact("长电科技", "研发费用率", "5.36"),
    ]
    section = analyze_operations(plan, facts, enabled=True)
    text = "\n".join(section.lines)
    assert "毛利率梯队" in text
    assert "寒武纪 55.29%" in text
    assert "研发费用率梯队" in text
    assert "上游" in text or "设计" in text
    assert "≈" in text and "%" in text
    assert section.derived_facts
    assert any(fact.name.endswith("均值") for fact in section.derived_facts)
    assert "被合并方" not in text
    assert "净利润为 0" not in text
    assert "本节从略" not in text
