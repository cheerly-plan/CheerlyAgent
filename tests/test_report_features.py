"""状态特征 + 呈现规则：按数据分叉，不按公司名。"""

from __future__ import annotations

from ai_agent.report.compare import Change, SeriesCompare
from ai_agent.report.compose import (
    _change_lines,
    _equity_composition_lines,
    compose_report,
)
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.features import (
    EQUITY_GROWTH_RATIO,
    Features,
    compute_features,
    loss_framing,
)
from ai_agent.report.parse import OutlineItem, ReportPlan
from ai_agent.report.rules import (
    equity_share_applicable,
    render_yoy_rate_bit,
    render_yoy_sentence,
)
from ai_agent.report.supplemental import (
    BalanceMoveView,
    SupplementalResult,
    YearEndView,
)


def _plan(**kwargs) -> ReportPlan:
    base = dict(
        title="t",
        kind="company",
        deliverable="markdown",
        subjects=["测试公司"],
        period="2026Q3",
        dimensions=["同比", "环比"],
        indicators=["扣非净利润", "营业收入"],
        outline=[OutlineItem("同比与环比", "")],
        company_type="fabless",
    )
    base.update(kwargs)
    return ReportPlan(**base)


def _fact(name: str, value: str, period: str = "2026Q3") -> IndicatorFact:
    return IndicatorFact(
        subject="测试公司",
        period=period,
        name=name,
        value=value,
        unit="元",
        column="年初至报告期末",
        scope="consolidated",
        evidence=Evidence(
            file_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            filename="t.pdf",
            page=1,
            page_end=1,
            section="主要财务数据",
            quote=f"{name} {value}",
        ),
    )


def test_loss_framing_widen_and_narrow():
    assert loss_framing("-200", "-100") == "亏损扩大"
    assert loss_framing("-50", "-100") == "亏损收窄"
    assert loss_framing("100", "-50") == "扭亏为盈"
    assert loss_framing("-50", "100") == "由盈转亏"
    assert loss_framing("200", "100") is None


def test_profitable_feature_from_koufei():
    plan = _plan()
    features = compute_features(plan, [_fact("扣非净利润", "-1,000")])
    assert features.profitable is False
    features2 = compute_features(plan, [_fact("扣非净利润", "1,000")])
    assert features2.profitable is True


def test_equity_growth_large_threshold():
    # 上年末 100，期末 120 → +20% ≥ 10%
    views = SupplementalResult(
        year_ends=[
            YearEndView(
                name="归属于上市公司股东的所有者权益",
                value="120",
                prior_value="100",
                change="+20%",
                unit="元",
            )
        ]
    )
    features = compute_features(_plan(), [], views)
    assert features.equity_growth_large is True
    # +5% < 10%
    views2 = SupplementalResult(
        year_ends=[
            YearEndView(
                name="归属于上市公司股东的所有者权益",
                value="105",
                prior_value="100",
                change="+5%",
                unit="元",
            )
        ]
    )
    features2 = compute_features(_plan(), [], views2)
    assert features2.equity_growth_large is False
    assert EQUITY_GROWTH_RATIO > 0


def test_yoy_rule_uses_loss_framing_not_growth_narrative():
    change = Change(
        base_period="2025Q3",
        base_value="-100",
        delta="-100",
        rate="100%",
        source="",
    )
    text = render_yoy_sentence(
        prefix="",
        name="扣非净利润",
        period="2026Q3",
        current_value="-200",
        unit="元",
        change=change,
        note="累计口径同比",
        features=Features(profitable=False),
    )
    assert "亏损扩大" in text
    assert "本期为 -200元" in text
    assert "较 2025Q3（-100元）" in text
    assert "差额 -100" in text
    assert "元元" not in text


def test_yoy_rule_normal_when_both_positive():
    change = Change(
        base_period="2025Q3",
        base_value="100",
        delta="100",
        rate="100%",
        source="",
    )
    text = render_yoy_sentence(
        prefix="",
        name="营业收入",
        period="2026Q3",
        current_value="200",
        unit="元",
        change=change,
        note="累计口径同比",
        features=Features(profitable=True),
    )
    assert "同比增减" in text
    assert "亏损" not in text


def test_equity_composition_skips_share_when_growth_small():
    views = SupplementalResult(
        year_ends=[
            YearEndView(
                name="归属于上市公司股东的所有者权益",
                value="105",
                prior_value="100",
                change="+5%",
                unit="元",
            )
        ],
        balance_moves=[
            BalanceMoveView(
                name="其他综合收益",
                end_value="10",
                prior_value="8",
                unit="元",
                evidence=Evidence(
                    file_id="a",
                    filename="t.pdf",
                    page=1,
                    page_end=1,
                    section="资产负债表",
                    quote="其他综合收益 10 8",
                ),
            ),
            BalanceMoveView(
                name="未分配利润",
                end_value="50",
                prior_value="48",
                unit="元",
                evidence=Evidence(
                    file_id="a",
                    filename="t.pdf",
                    page=1,
                    page_end=1,
                    section="资产负债表",
                    quote="未分配利润 50 48",
                ),
            ),
            BalanceMoveView(
                name="资本公积",
                end_value="40",
                prior_value="39",
                unit="元",
                evidence=Evidence(
                    file_id="a",
                    filename="t.pdf",
                    page=1,
                    page_end=1,
                    section="资产负债表",
                    quote="资本公积 40 39",
                ),
            ),
        ],
    )
    features = compute_features(_plan(), [], views)
    assert equity_share_applicable(features) is False
    lines = _equity_composition_lines(views, features)
    text = "\n".join(lines)
    assert "不适用" in text
    assert "占比不适用" in text
    assert "约 74%" not in text


def test_change_lines_wires_loss_framing():
    series = [
        SeriesCompare(
            subject="测试公司",
            name="扣非净利润",
            scope="consolidated",
            unit="元",
            points=[("2025Q3", "-100"), ("2026Q3", "-200")],
            yoy={
                "2026Q3": Change(
                    base_period="2025Q3",
                    base_value="-100",
                    delta="-100",
                    rate="100%",
                )
            },
        )
    ]
    lines = _change_lines(
        _plan(), series, Features(profitable=False, industry="fabless")
    )
    assert any("亏损扩大" in line for line in lines)


def test_lead_reuses_yoy_direction_bit():
    change = Change(
        base_period="2025Q3",
        base_value="-100",
        delta="-100",
        rate="46.66%",
        source="",
    )
    assert render_yoy_rate_bit("-200", change) == "亏损扩大 46.66%"
    narrow = Change(
        base_period="2025Q3",
        base_value="-200",
        delta="100",
        rate="-48.95%",
        source="",
    )
    assert render_yoy_rate_bit("-100", narrow) == "亏损收窄 48.95%"

    facts = [
        _fact("营业收入", "200"),
        _fact("扣非净利润", "-200"),
        _fact("营业收入", "100", period="2025Q3"),
        _fact("扣非净利润", "-100", period="2025Q3"),
    ]
    plan = _plan(
        outline=[
            OutlineItem("本期结论", ""),
            OutlineItem("同比与环比", ""),
            OutlineItem("主要指标与出处", ""),
        ],
        indicators=["营业收入", "扣非净利润"],
    )
    document = compose_report(plan, facts)
    lead = document.markdown.split("## 本期结论", 1)[1].split("\n## ", 1)[0]
    assert "亏损扩大" in lead
    assert "扣非净利润为 -200元（同比" not in lead.replace(" ", "")
