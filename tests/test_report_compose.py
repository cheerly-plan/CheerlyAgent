"""研报正文只复用抽取值和计算结果，不出现表格里没有的数字。"""

from ai_agent.report.compare import NOT_COMPARABLE_NOTE, SeriesCompare
from ai_agent.report.compose import _gaps, _segment_table_indicators, compose_report
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.industry import default_indicators
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
    assert text.count("名单备注：") == 1
    assert "韦尔股份现名豪威集团（603501）" in text
    assert "股票代码暂略" not in text
    assert "### 全行业毛利率梯队" in text
    assert "| 名次 | 公司 | 毛利率 | 出处 |" in text
    assert "### 设备" in text
    # 设备赛道：前六列订单侧 + 补经营现金流（与封测表对齐）。
    assert (
        "| 公司 | 营业收入 | 扣非净利润 | 毛利率 | 合同负债 | 存货 | 在手订单 | "
        "经营活动产生的现金流量净额 |"
        in text
    )
    # 设计赛道同样补现金流列。
    assert (
        "| 公司 | 营业收入 | 扣非净利润 | 毛利率 | 研发费用率 | 存货周转率 | "
        "大客户集中度 | 经营活动产生的现金流量净额 |"
        in text
    )
    assert "北方华创" in text and "中微公司" in text
    # 中芯千元入口折成元，脚注保留原单位。
    assert "49,510,416,000元" in text
    assert "由千元折算为元" in text
    assert "金额默认统一为元" in text
    assert "WSTS" in text
    assert "深度分析" not in plan.title
    # 不再按指标拆成多张三列表标题
    assert "营业收入（合并）" not in text


def test_segment_table_keeps_ocf_beyond_core_budget():
    equipment = _segment_table_indicators(default_indicators("equipment"))
    assert equipment[:6] == [
        "营业收入",
        "扣非净利润",
        "毛利率",
        "合同负债",
        "存货",
        "在手订单",
    ]
    assert equipment[-1] == "经营活动产生的现金流量净额"
    fabless = _segment_table_indicators(default_indicators("fabless"))
    assert "经营活动产生的现金流量净额" in fabless
    assert fabless.index("经营活动产生的现金流量净额") >= 6
    packaging = _segment_table_indicators(default_indicators("packaging"))
    assert packaging[-1] == "经营活动产生的现金流量净额"
    assert len(packaging) == 6


def test_series_gaps_same_reason_summarized_once():
    """同一「不推导单季环比」原因跨公司／指标只概括一条，例外单列。"""
    reason = "2025Q3 的累计报告不齐（2025Q1、2025H1 需同为累计口径），不推导单季环比"
    exception = "2025Q3 推导出的单季值非正，不写环比"
    companies = ["寒武纪", "长电科技", "兆易创新", "北方华创", "中芯国际", "中微公司"]
    indicators = ["营业收入", "扣非净利润"]
    series = [
        SeriesCompare(subject=co, name=name, scope="consolidated", unit="元", gaps=[reason])
        for co in companies
        for name in indicators
    ]
    series.append(
        SeriesCompare(
            subject="中科飞测",
            name="营业收入",
            scope="consolidated",
            unit="元",
            gaps=[exception],
        )
    )
    # 再塞一批「不可比」确认计数合并仍在
    for co in companies[:3]:
        series.append(
            SeriesCompare(
                subject=co,
                name="毛利率",
                scope="consolidated",
                unit="%",
                gaps=[NOT_COMPARABLE_NOTE],
            )
        )
    plan = ReportPlan(
        title="半导体行业 2025Q3",
        kind="industry",
        deliverable="markdown",
        subjects=companies + ["中科飞测"],
        period="2025Q3",
        indicators=["营业收入"],
        outline=[OutlineItem("缺失与口径", "")],
        gaps=[
            "A 股无独立二季报；缺 2025Q1 的累计报告，2025Q3 单季环比无法推导",
            "A 股无独立二季报；缺 2025H1 的累计报告，2025Q3 单季环比无法推导",
        ],
    )
    # 每家给一条营收，避免「没有抽到」干扰断言
    facts = [
        IndicatorFact(
            subject=co,
            period="2025Q3",
            name="营业收入",
            value="1",
            unit="元",
            column="本报告期",
            scope="consolidated",
            evidence=Evidence(
                file_id=co,
                filename=f"{co}.pdf",
                page=1,
                page_end=1,
                section="",
                quote="1",
            ),
        )
        for co in plan.subjects
    ]
    gaps = _gaps(plan, facts, series)
    clustered = [g for g in gaps if "不推导单季环比" in g and "家公司" in g]
    assert len(clustered) == 1
    assert "6 家公司" in clustered[0]
    assert "寒武纪" in clustered[0] and "兆易创新" in clustered[0]
    assert "营业收入" in clustered[0] and "扣非净利润" in clustered[0]
    # 不得再逐条复读（「公司+单指标：原因」）
    per_item = [
        g
        for g in gaps
        if g.endswith(f"：{reason}") and "家公司" not in g and "项指标（" not in g
    ]
    assert per_item == []
    assert any(g.startswith("中科飞测营业收入：") and exception in g for g in gaps)
    assert any("3 项指标：" in g and NOT_COMPARABLE_NOTE in g for g in gaps)
    # Q1／H1 两条累计基期合并为一句
    base_lines = [g for g in gaps if "缺 " in g and "累计报告" in g and "单季环比无法推导" in g]
    assert len(base_lines) == 1
    assert "2025Q1" in base_lines[0] and "2025H1" in base_lines[0]


def test_missing_indicator_gaps_summarized_across_companies():
    plan = ReportPlan(
        title="半导体行业 2025Q3",
        kind="industry",
        deliverable="markdown",
        subjects=["北方华创", "中微公司", "中科飞测"],
        period="2025Q3",
        indicators=["合同负债"],
        outline=[OutlineItem("缺失与口径", "")],
        company_type="equipment",
    )
    gaps = _gaps(plan, [], [])
    assert len([g for g in gaps if "合同负债" in g]) == 1
    assert "3 家公司" in gaps[0]
    assert "北方华创" in gaps[0] and "中科飞测" in gaps[0]
