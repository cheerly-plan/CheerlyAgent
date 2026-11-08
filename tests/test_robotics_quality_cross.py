"""机器人赛道经营解读：表内利润—现金流—合同负债交叉 + 原文引述。"""

from ai_agent.report.analyze import EvidenceItem, analyze_operations
from ai_agent.report.analyze_structure import quality_cross_lines
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan


def _money(subject: str, name: str, value: str, *, file_id: str = "") -> IndicatorFact:
    return IndicatorFact(
        subject=subject,
        period="2026Q3",
        name=name,
        value=value,
        unit="元",
        column="年初至报告期末",
        scope="consolidated",
        evidence=Evidence(
            file_id=file_id or f"{subject}-{name}",
            filename=f"{subject}.pdf",
            page=1,
            page_end=None,
            section="主要财务数据",
            quote=f"{name} {value}元",
        ),
    )


def _ratio(subject: str, name: str, value: str) -> IndicatorFact:
    return IndicatorFact(
        subject=subject,
        period="2026Q3",
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


def test_quality_cross_prefers_sign_divergence_and_states_relation():
    facts = [
        _money("新松机器人", "扣非净利润", "-361000000"),
        _money("新松机器人", "经营活动产生的现金流量净额", "-644000000"),
        _money("新松机器人", "合同负债", "2292000000"),
        _money("双环传动", "扣非净利润", "845000000"),
        _money("双环传动", "经营活动产生的现金流量净额", "1427000000"),
        _money("双环传动", "合同负债", "120000000"),
        _money("汇川技术", "扣非净利润", "3907000000"),
        _money("汇川技术", "经营活动产生的现金流量净额", "895000000"),
        _money("汇川技术", "合同负债", "1053000000"),
    ]
    lines = quality_cross_lines(facts)
    text = "\n".join(lines)
    assert "利润—现金流—合同负债交叉" in text
    assert "不推断收入确认或回款" in text
    assert "新松机器人" in text
    assert "扣非净利润与经营活动产生的现金流量净额均为负" in text
    assert "合同负债" in text and "时点余额" in text
    assert "双环传动" in text
    assert "经营现金流高于扣非净利润" in text
    assert "汇川技术" in text
    assert "经营现金流低于扣非净利润" in text
    # 双负（新松）优先于同号相对落差（汇川／双环）。
    assert text.index("新松机器人") < text.index("汇川技术")
    assert text.index("新松机器人") < text.index("双环传动")


def test_robotics_analysis_appends_quality_cross_after_ladder():
    plan = ReportPlan(
        title="机器人行业 2026Q3 财报对比分析（赛道分组版）",
        kind="industry",
        deliverable="markdown",
        skill_id="robotics_industry_report",
        industry="机器人",
        subjects=[
            "汇川技术",
            "鸣志电器",
            "埃斯顿",
            "新松机器人",
            "绿的谐波",
            "双环传动",
            "中大力德",
            "拓斯达",
        ],
        period="2026Q3",
        indicators=["毛利率", "研发费用率", "扣非净利润", "经营活动产生的现金流量净额", "合同负债"],
        outline=[OutlineItem("经营解读", "")],
    )
    facts = [
        _ratio("鸣志电器", "毛利率", "38.13"),
        _ratio("汇川技术", "毛利率", "28.24"),
        _ratio("双环传动", "毛利率", "22.10"),
        _ratio("绿的谐波", "毛利率", "35.00"),
        _ratio("埃斯顿", "研发费用率", "12.50"),
        _ratio("新松机器人", "研发费用率", "10.20"),
        _ratio("中大力德", "研发费用率", "8.10"),
        _money("新松机器人", "扣非净利润", "-361000000"),
        _money("新松机器人", "经营活动产生的现金流量净额", "-644000000"),
        _money("新松机器人", "合同负债", "2292000000"),
        _money("双环传动", "扣非净利润", "845000000"),
        _money("双环传动", "经营活动产生的现金流量净额", "1427000000"),
        _money("双环传动", "合同负债", "120000000"),
    ]
    # evidence=[]：跳过 KB，专测质量交叉；无原文时归因从略。
    section = analyze_operations(plan, facts, evidence=[], enabled=True)
    text = "\n".join(section.lines)
    assert "毛利率梯队" in text
    assert "利润—现金流—合同负债交叉" in text
    assert "新松机器人" in text
    assert "均为负" in text
    assert "经营现金流高于扣非净利润" in text
    assert "不引入无出处的订单／景气推断" in text
    assert "不推断收入确认或回款" in text
    assert "原文披露" not in text
    assert any("未召回可引述的经营分析原文" in gap for gap in section.gaps)


def test_robotics_analysis_appends_cited_evidence_quotes():
    """赛道报告放开原文引述：有出处可写，无出处不编因果。"""
    plan = ReportPlan(
        title="机器人行业 2026Q3 财报对比分析（赛道分组版）",
        kind="industry",
        deliverable="markdown",
        skill_id="robotics_industry_report",
        industry="机器人",
        subjects=["汇川技术", "鸣志电器", "埃斯顿", "新松机器人"],
        period="2026Q3",
        indicators=["毛利率", "研发费用率"],
        outline=[OutlineItem("经营解读", "")],
    )
    facts = [
        _ratio("鸣志电器", "毛利率", "38.13"),
        _ratio("汇川技术", "毛利率", "28.24"),
        _ratio("埃斯顿", "研发费用率", "12.50"),
        _ratio("新松机器人", "研发费用率", "10.20"),
    ]
    pack = [
        EvidenceItem(
            evidence_id="e0",
            file_id="a" * 36,
            filename="汇川技术2026Q3.pdf",
            page=8,
            section="经营情况讨论与分析",
            quote=(
                "报告期内，工业自动化下游需求回暖，公司伺服与变频器出货规模提升，"
                "整体经营业绩实现较好增长。"
            ),
        ),
        EvidenceItem(
            evidence_id="e1",
            file_id="b" * 36,
            filename="新松机器人2026Q3.pdf",
            page=6,
            section="经营情况讨论与分析",
            quote=(
                "报告期内，公司持续加大机器人本体与系统集成研发投入，"
                "产能与客户布局持续优化。"
            ),
        ),
        EvidenceItem(
            evidence_id="e2",
            file_id="c" * 36,
            filename="noise.pdf",
            page=2,
            section="合并利润表",
            quote="合并利润表 编制单位 审计类型 净利润为0",
        ),
    ]
    section = analyze_operations(plan, facts, evidence=pack, enabled=True)
    text = "\n".join(section.lines)
    assert "毛利率梯队" in text
    assert text.count("原文披露：") == 2
    assert "伺服与变频器出货" in text
    assert "研发投入" in text
    assert "第8页" in text
    assert "净利润为0" not in text
    assert "合并利润表" not in text
    assert "季报原文引述" in text
    assert "不引入无出处的订单／景气推断" in text
