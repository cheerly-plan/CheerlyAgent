"""累计口径回归：Q1/H1 基期、空格千分位、累计转单季、推导值门禁。

A 股没有独立二季报，Q3 的真单季环比只能用 Q1 / H1 / Q3 三份累计报告
相减还原。这组用例把「什么条件下允许推导、什么条件下必须老实不写」钉住。
"""

from __future__ import annotations

from ai_agent.report.compare import (
    Change,
    Measure,
    SeriesCompare,
    compare_measures,
    cumulative_base_periods,
    format_compare,
)
from ai_agent.report.compose import compose_report
from ai_agent.report.extract import Evidence, IndicatorFact, _NUMBER, read_indicator
from ai_agent.report.parse import OutlineItem, ReportPlan, parse_request
from ai_agent.report.pdf import charts_from_series
from ai_agent.report.validate import validate_report

_SUBJECT = "中国银行股份有限公司"


def _measure(
    period: str,
    value: str,
    *,
    cumulative: bool = True,
    unit: str = "百万元",
    source_key: str = "f-1",
) -> Measure:
    return Measure(
        subject=_SUBJECT,
        name="营业收入",
        period=period,
        value=value,
        unit=unit,
        scope="consolidated",
        cumulative=cumulative,
        source_key=source_key,
    )


def _chain() -> list[Measure]:
    """真实语料：Q1 1-3月累计、H1 1-6月累计、Q3 1-9月累计。"""
    return [
        _measure("2026Q1", "178,846", source_key="f-q1"),
        _measure("2026H1", "356 903", source_key="f-h1"),
        _measure("2026Q3", "526,365", source_key="f-q3"),
    ]


def _fact(period: str, value: str, file_id: str) -> IndicatorFact:
    return IndicatorFact(
        subject=_SUBJECT,
        period=period,
        name="营业收入",
        value=value,
        unit="百万元",
        column="年初至报告期末",
        scope="consolidated",
        cumulative=True,
        evidence=Evidence(
            file_id=file_id,
            filename=f"{_SUBJECT}{period}报告.pdf",
            page=2,
            page_end=2,
            section="主要财务数据",
            quote=f"单位：百万元\n项目 | 年初至报告期末\n营业收入 | {value}",
        ),
    )


def _plan() -> ReportPlan:
    return ReportPlan(
        title="中国银行股份有限公司 2026Q3 财报深度分析",
        kind="company",
        deliverable="pdf",
        subjects=[_SUBJECT],
        period="2026Q3",
        dimensions=["同比", "环比"],
        indicators=["营业收入"],
        outline=[
            OutlineItem("本期结论", "结论"),
            OutlineItem("主要指标与出处", "指标"),
            OutlineItem("同比与环比", "增减"),
            OutlineItem("趋势", "趋势"),
            OutlineItem("缺失与口径", "缺口"),
        ],
        gaps=[],
    )


# ——— 期间链 ———


def test_cumulative_base_periods_cover_quarter_three_and_four():
    assert cumulative_base_periods("2026Q3") == ["2026Q1", "2026H1"]
    assert cumulative_base_periods("2026Q4") == ["2026Q1", "2026H1", "2026Q3"]
    assert cumulative_base_periods("2026H1") == ["2026Q1"]
    assert cumulative_base_periods("2026Q1") == []
    assert cumulative_base_periods("2026") == []
    assert cumulative_base_periods("not-a-period") == []


def test_parse_plans_cumulative_bases_instead_of_nonexistent_q2(monkeypatch):
    def row(period: str) -> dict:
        return {
            "file_id": f"f-{period}",
            "filename": f"{_SUBJECT}{period}报告",
            "subject": _SUBJECT,
            "subject_aliases": [_SUBJECT, "中国银行"],
            "period": period,
            "status": "ready",
            "tags": [],
            "description": "",
        }

    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [row(p) for p in ("2026Q3", "2025Q3", "2026Q1", "2026H1")],
    )
    plan = parse_request("生成中国银行2026Q3财报深度分析PDF研报")
    roles = {(item.period, item.role) for item in plan.files}
    assert ("2026Q1", "累计基期") in roles
    assert ("2026H1", "累计基期") in roles
    assert ("2026Q3", "本期") in roles
    assert ("2025Q3", "同比基期") in roles
    assert not any(period == "2026Q2" for period, _role in roles)
    assert not any("2026Q2" in gap for gap in plan.gaps)


def test_parse_reports_missing_cumulative_bases_honestly(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [
            {
                "file_id": "f-q3",
                "filename": f"{_SUBJECT}2026Q3报告",
                "subject": _SUBJECT,
                "subject_aliases": [_SUBJECT, "中国银行"],
                "period": "2026Q3",
                "status": "ready",
                "tags": [],
                "description": "",
            }
        ],
    )
    plan = parse_request("生成中国银行2026Q3财报深度分析PDF研报")
    assert any("A 股无独立二季报" in gap and "2026Q1" in gap for gap in plan.gaps)
    assert any("A 股无独立二季报" in gap and "2026H1" in gap for gap in plan.gaps)
    assert not any("2026Q2" in gap for gap in plan.gaps)


# ——— 空格千分位 ———


def test_space_grouped_number_is_one_value():
    assert _NUMBER.match("356 903")
    assert _NUMBER.match("1 234 567")
    assert _NUMBER.match("526,365")
    assert _NUMBER.match("8.48%")
    assert not _NUMBER.match("1 2 3")
    assert not _NUMBER.match("1 23")


def test_read_indicator_accepts_space_grouped_cumulative_column():
    table = """单位：百万元人民币（百分比除外）
| 项目 | 2026年1-6月 | 2025年1-6月 | 本报告期比上年同期增减(%) |
| --- | --- | --- | --- |
| 营业收入 | 356 903 | 329 003 | 8.48% |
"""
    got = read_indicator(table, ["营业收入"], period="2026H1")
    assert got is not None
    assert got["value"] == "356 903"
    assert got["column"] == "2026年1-6月"
    assert got["cumulative"] is True
    assert "329 003" not in got["value"]


# ——— 累计 → 单季 ———


def test_derived_qoq_matches_the_official_single_quarter():
    series = compare_measures(_chain())[0]
    qoq = series.qoq["2026Q3"]
    assert qoq.source == "derived"
    assert qoq.base_period == "2026Q2"
    assert qoq.base_value == "178,057百万元"
    assert qoq.delta == "-8,595百万元"
    assert qoq.rate == "-4.83%"
    assert qoq.evidence == ("f-q1", "f-h1", "f-q3")
    assert "累计值相减" in qoq.note
    # 526,365 − 356,903 = 169,462，与三季报原文「本报告期」单季一致
    assert 526365 - 356903 == 169462
    assert "累计值推导的基期" in format_compare([series])


def test_derived_qoq_requires_the_full_chain():
    rows = [item for item in _chain() if item.period != "2026H1"]
    series = compare_measures(rows)[0]
    assert not any(change.source == "derived" for change in series.qoq.values())
    assert "2026Q3" not in series.qoq
    # 纯计算层不重复报「缺累计报告」；缺口由 parse 按实际库藏给出。
    # 但期间不可比的定性说明允许出现（它不是重复缺口）。
    assert not any("累计报告" in gap for gap in series.gaps)


def test_derived_qoq_refuses_non_cumulative_values():
    rows = _chain()
    rows[1] = _measure("2026H1", "356,903", cumulative=False, source_key="f-h1")
    series = compare_measures(rows)[0]
    assert not any(change.source == "derived" for change in series.qoq.values())


def test_derived_qoq_refuses_incompatible_units():
    rows = _chain()
    rows[1] = _measure("2026H1", "356,903", unit="亿元", source_key="f-h1")
    series = compare_measures(rows)[0]
    assert not any(change.source == "derived" for change in series.qoq.values())


def test_derived_qoq_refuses_missing_source_keys():
    rows = _chain()
    rows[1] = _measure("2026H1", "356,903", source_key="")
    series = compare_measures(rows)[0]
    assert not any(change.source == "derived" for change in series.qoq.values())


def test_single_quarter_q2_does_not_override_correct_derivation():
    """本期是累计值时，Q2 单季不能直接当环比基期（累计减单季无意义）。

    旧版会拿 526,365 − 178,057 当「真实环比」并跳过推导；现在仍走累计链。
    """
    rows = [*_chain(), _measure("2026Q2", "178,057", cumulative=False, source_key="f-q2")]
    series = compare_measures(rows)[0]
    assert series.qoq["2026Q3"].source == "derived"
    assert series.qoq["2026Q3"].rate == "-4.83%"


def test_unit_noise_does_not_break_cross_period_yoy():
    series = compare_measures(
        [
            _measure("2025Q3", "491,204", unit="百万元人民币（另有说明者除外）", source_key="a"),
            _measure("2026Q3", "526,365", unit="百万元人民币（百分比除外）", source_key="b"),
        ]
    )[0]
    assert series.yoy["2026Q3"].rate == "7.16%"


def test_cumulative_trend_refuses_incomparable_periods():
    """Q1/H1/Q3 是 3/6/9 个月累计：相减只是期数差，不产出任何增减或幅度。"""
    series = compare_measures(_chain())[0]
    assert series.trend == []
    assert any("覆盖期数不同" in gap for gap in series.gaps)
    text = format_compare([series])
    assert "跨期）增减" not in text  # 旧版会把 H1−Q1 写成趋势增减
    assert "99.56%" not in text
    assert "47.48%" not in text
    # 同口径同比仍然合法
    assert series.yoy == {} or all(change.rate for change in series.yoy.values())


def test_chart_skips_incomparable_periods_marks_comparable_ones():
    """1-3月/1-6月/1-9月累计不能连成一条增长曲线；同口径的两年 Q3 可以。"""
    series = compare_measures(_chain())[0]
    assert charts_from_series([series]) == []

    rows = [*_chain(), _measure("2025Q3", "491,204", source_key="f-yoy")]
    comparable = compare_measures(rows)[0]
    charts = charts_from_series([comparable])
    assert len(charts) == 1
    assert "累计口径，非单季" in charts[0].title
    assert charts[0].labels == ["2025Q3", "2026Q3"]


# ——— 推导值门禁 ———


def test_validate_allows_derived_value_with_full_evidence_chain():
    facts = [
        _fact("2026Q1", "178,846", "f-q1"),
        _fact("2026H1", "356 903", "f-h1"),
        _fact("2026Q3", "526,365", "f-q3"),
    ]
    plan = _plan()
    document = compose_report(plan, facts)
    series = compare_measures(
        [
            _measure("2026Q1", "178,846", source_key="f-q1"),
            _measure("2026H1", "356 903", source_key="f-h1"),
            _measure("2026Q3", "526,365", source_key="f-q3"),
        ]
    )
    result = validate_report(document, facts, plan=plan, series=series)
    assert "-4.83%" in document.markdown
    assert result.passed, [item.message for item in result.findings]


def test_validate_blocks_derived_value_without_evidence_chain():
    """手工塞一个无出处的推导 Change：正文数字必须被拦下，不能当成已核对。"""
    facts = [
        _fact("2026Q1", "178,846", "f-q1"),
        _fact("2026H1", "356 903", "f-h1"),
        _fact("2026Q3", "526,365", "f-q3"),
    ]
    plan = _plan()
    document = compose_report(plan, facts)
    tampered = SeriesCompare(
        subject=_SUBJECT,
        name="营业收入",
        scope="consolidated",
        unit="百万元",
        qoq={
            "2026Q3": Change(
                base_period="2026Q2",
                base_value="178,057百万元",
                delta="-8,595百万元",
                rate="-4.83%",
                source="derived",
                evidence=(),  # 出处链被抹掉
            )
        },
    )
    result = validate_report(document, facts, plan=plan, series=[tampered])
    assert not result.passed
    assert any("推导环比缺完整出处链" in item.message for item in result.findings)
    assert any("未核对数字" in item.message for item in result.findings)
