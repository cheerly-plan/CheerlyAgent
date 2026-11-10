"""银行行业经营解读：按 structure 写息差／资产质量梯队，不走毛利率叙事。"""

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


def test_bank_industry_structure_uses_nim_not_margin():
    plan = ReportPlan(
        title="银行行业 2025Q3 财报对比分析（赛道分组版）",
        kind="industry",
        deliverable="markdown",
        skill_id="bank_industry_report",
        industry="银行",
        subjects=[
            "工商银行",
            "建设银行",
            "招商银行",
            "兴业银行",
            "宁波银行",
            "杭州银行",
        ],
        period="2025Q3",
        indicators=["净息差", "不良贷款率", "不良贷款拨备覆盖率"],
        outline=[OutlineItem("经营解读", "")],
    )
    facts = [
        _fact("招商银行", "净息差", "1.95"),
        _fact("兴业银行", "净息差", "1.80"),
        _fact("宁波银行", "净息差", "1.75"),
        _fact("工商银行", "净息差", "1.42"),
        _fact("建设银行", "净息差", "1.40"),
        _fact("杭州银行", "净息差", "1.38"),
        _fact("宁波银行", "不良贷款率", "0.76"),
        _fact("杭州银行", "不良贷款率", "0.76"),
        _fact("招商银行", "不良贷款率", "0.95"),
        _fact("工商银行", "不良贷款率", "1.35"),
        _fact("建设银行", "不良贷款率", "1.35"),
        _fact("兴业银行", "不良贷款率", "1.10"),
    ]
    section = analyze_operations(plan, facts, enabled=True)
    text = "\n".join(section.lines)
    assert "净息差梯队" in text
    assert "招商银行 1.95%" in text
    assert "不良贷款率梯队" in text
    assert "毛利率" not in text
    assert "研发费用率" not in text
    assert "本节从略" not in text
    assert section.derived_facts


def test_bank_company_fallback_priority_uses_profile_not_margin():
    """单家银行无原文时，数字并列按 bank.structure／lead，不写毛利率。"""
    from ai_agent.report.analyze import _company_fact_priority, _deterministic_lines

    plan = ReportPlan(
        title="招商银行 2025Q3",
        kind="company",
        deliverable="markdown",
        skill_id="bank_financial_report",
        industry="银行",
        subjects=["招商银行"],
        period="2025Q3",
        indicators=["净息差", "不良贷款率", "营业收入"],
        outline=[OutlineItem("经营解读", "")],
    )
    assert "毛利率" not in _company_fact_priority(plan)
    assert _company_fact_priority(plan)[0] == "净息差"
    facts = [
        _fact("招商银行", "净息差", "1.95"),
        _fact("招商银行", "不良贷款率", "0.95"),
    ]
    body = "\n".join(_deterministic_lines(plan, facts, []))
    assert "净息差" in body
    assert "毛利率" not in body
