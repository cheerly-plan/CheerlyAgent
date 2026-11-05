"""对比计算不调用模型：同比、环比、百分点和单位换算都是确定的。"""

from ai_agent.report.compare import Measure, compare_measures
from ai_agent.tools.report_compare import report_compare_tool


def _row(period: str, value: str, unit: str = "元", name: str = "营业收入") -> Measure:
    return Measure(
        subject="中国能源建设股份有限公司",
        name=name,
        period=period,
        value=value,
        unit=unit,
        scope="consolidated",
    )


def test_yoy_and_qoq_are_exact():
    series = compare_measures(
        [
            _row("2024Q3", "111,000,000"),
            _row("2025Q2", "90,000,000"),
            _row("2025Q3", "100,924,954"),
        ]
    )[0]
    yoy = series.yoy["2025Q3"]
    assert yoy.base_period == "2024Q3"
    assert yoy.delta == "-10,075,046元"
    assert yoy.rate == "-9.08%"
    qoq = series.qoq["2025Q3"]
    assert qoq.base_period == "2025Q2"
    assert qoq.delta == "10,924,954元"
    assert qoq.rate == "12.14%"
    # 2024Q3 到 2025Q2 不是相邻季度，趋势里标成跨期
    assert series.trend[0].adjacent is False
    assert series.trend[1].adjacent is True


def test_margin_uses_percentage_points():
    series = compare_measures(
        [
            _row("2024Q3", "11.00%", unit="%", name="毛利率"),
            _row("2025Q3", "12.30%", unit="%", name="毛利率"),
        ]
    )[0]
    yoy = series.yoy["2025Q3"]
    assert yoy.delta == "1.30个百分点"
    assert yoy.rate == ""


def test_unit_scale_converts_before_subtracting():
    series = compare_measures(
        [
            _row("2024", "5,000", unit="万元"),
            _row("2025", "1", unit="亿元"),
        ]
    )[0]
    yoy = series.yoy["2025"]
    assert yoy.delta == "0.50亿元"
    assert yoy.rate == "100%"
    assert series.qoq == {}


def test_zero_base_has_no_rate():
    series = compare_measures(
        [_row("2024Q3", "0"), _row("2025Q3", "10")]
    )[0]
    yoy = series.yoy["2025Q3"]
    assert yoy.delta == "10元"
    assert yoy.rate == ""
    assert yoy.note == "基期为 0，不算增幅"


def test_conflicting_period_is_dropped():
    series = compare_measures(
        [
            _row("2024Q3", "100"),
            _row("2025Q3", "80"),
            _row("2025Q3", "90"),
        ]
    )[0]
    assert "2025Q3" not in series.yoy
    assert series.points == [("2024Q3", "100")]
    assert series.gaps == ["2025Q3 有两个不同数字，这一期不参与计算"]


def test_tool_returns_the_computed_text():
    text = report_compare_tool.run(
        points=[
            {
                "subject": "中国能源建设股份有限公司",
                "name": "营业收入",
                "period": "2024Q3",
                "value": "111,000,000",
                "unit": "元",
                "scope": "consolidated",
            },
            {
                "subject": "中国能源建设股份有限公司",
                "name": "营业收入",
                "period": "2025Q3",
                "value": "100,924,954",
                "unit": "元",
                "scope": "consolidated",
            },
        ]
    )
    assert "增减 -10,075,046元，幅度 -9.08%" in text
    assert "环比" not in text
