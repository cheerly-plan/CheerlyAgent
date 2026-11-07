"""研报正文只复用抽取值和计算结果，不出现表格里没有的数字。"""

from ai_agent.report.compose import compose_report
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan


def _fact(period: str, value: str, name: str = "营业收入") -> IndicatorFact:
    return IndicatorFact(
        subject="中国能源建设股份有限公司",
        period=period,
        name=name,
        value=value,
        unit="元",
        column="本报告期",
        scope="consolidated",
        evidence=Evidence(
            file_id="file-1",
            filename="中国能源建设2025年第三季度报告.pdf",
            page=12,
            page_end=12,
            section="主要会计数据",
            quote="营业收入 | " + value,
        ),
    )


def _plan() -> ReportPlan:
    return ReportPlan(
        title="中国能源建设股份有限公司 2025Q3 财报深度分析",
        kind="company",
        deliverable="pdf",
        subjects=["中国能源建设股份有限公司"],
        period="2025Q3",
        dimensions=["同比", "环比"],
        indicators=["营业收入", "毛利率"],
        outline=[
            OutlineItem("本期结论", ""),
            OutlineItem("主要指标与出处", ""),
            OutlineItem("同比与环比", ""),
            OutlineItem("趋势", ""),
            OutlineItem("缺失与口径", ""),
        ],
        gaps=["没有找到 2025Q2 的季报，环比基期无法调取"],
    )


def test_report_uses_extracted_numbers_and_computed_change():
    document = compose_report(
        _plan(),
        [
            _fact("2025Q3", "100,924,954"),
            _fact("2024Q3", "111,000,000"),
        ],
    )
    text = document.markdown
    assert "100,924,954元" in text
    assert "增减 -10,075,046元，幅度 -9.08%" in text
    assert "第12页" in text
    assert "file-1" in text
    assert "没有抽到 2025Q3 毛利率" in text
    assert "999" not in text
    assert "环比增减" not in text
    assert document.body.startswith("<!DOCTYPE html>")
    assert "100,924,954元" in document.html
    assert "<script>" not in document.html


def test_peer_table_needs_two_companies():
    plan = _plan()
    plan.kind = "industry"
    plan.industry = "半导体"
    plan.subjects = ["中国能源建设股份有限公司", "中国黄金集团黄金珠宝股份有限公司"]
    plan.outline = [OutlineItem("公司间对比", "")]
    other = _fact("2025Q3", "50")
    other.subject = "中国黄金集团黄金珠宝股份有限公司"
    other.evidence.file_id = "file-2"
    document = compose_report(plan, [_fact("2025Q3", "100,924,954"), other])
    assert "中国黄金集团黄金珠宝股份有限公司" in document.markdown
    assert "100,924,954元" in document.markdown
    assert "不做公司间对比" not in document.markdown


def test_segment_peer_tables_group_by_skill_segment():
    plan = ReportPlan(
        title="半导体行业 2025Q3 财报对比分析（赛道分组版）",
        kind="industry",
        deliverable="markdown",
        skill_id="semiconductor_industry_report",
        industry="半导体",
        subjects=["中芯国际", "北方华创", "中微公司", "韦尔股份"],
        period="2025Q3",
        indicators=["营业收入"],
        outline=[
            OutlineItem("阅读说明", ""),
            OutlineItem("赛道分组", ""),
            OutlineItem("组内对比", ""),
            OutlineItem("下期预告", ""),
        ],
    )

    def make(subject: str, value: str, unit: str = "元") -> IndicatorFact:
        fact = _fact("2025Q3", value)
        fact.subject = subject
        fact.unit = unit
        fact.evidence.file_id = subject
        return fact

    plan.indicators = ["营业收入", "扣非净利润", "毛利率", "研发费用率"]

    def make_more(subject: str, name: str, value: str, unit: str = "%") -> IndicatorFact:
        fact = _fact("2025Q3", value, name=name)
        fact.subject = subject
        fact.unit = unit
        fact.evidence.file_id = f"{subject}-{name}"
        return fact

    document = compose_report(
        plan,
        [
            make("中芯国际", "49,510,416", "千元"),
            make("北方华创", "27,301,379,656.81"),
            make("中微公司", "8,062,565,115.42"),
            make("韦尔股份", "21,782,627,742.87"),
            make_more("北方华创", "毛利率", "41.41"),
            make_more("中微公司", "毛利率", "39.1"),
            make_more("韦尔股份", "毛利率", "30.43"),
            make_more("中芯国际", "毛利率", "25.5"),
        ],
    )
    text = document.markdown
    assert "三级归因" in text
    assert "### 全行业毛利率梯队" in text
    assert "| 名次 | 公司 | 毛利率 | 出处 |" in text
    assert "### 设备" in text
    # 设备赛道按 equipment 模板路由；组内横比前六列含合同负债／存货／在手订单。
    assert (
        "| 公司 | 营业收入 | 扣非净利润 | 毛利率 | 合同负债 | 存货 | 在手订单 |"
        in text
    )
    assert "北方华创" in text and "中微公司" in text
    # 手工构造的 fact 未经抽取归一，仍可能带千元；阅读说明声明默认元。
    assert "千元" in text
    assert "金额默认统一为元" in text
    assert "WSTS" in text
    assert "深度分析" not in plan.title
    # 不再按指标拆成多张三列表标题
    assert "营业收入（合并）" not in text
