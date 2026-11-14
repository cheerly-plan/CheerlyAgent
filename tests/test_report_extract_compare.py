"""report_extract_indicators 的 compare_to：一次抽两期并给出增减。"""

from unittest.mock import patch

from ai_agent.report.extract import Evidence, ExtractResult, IndicatorFact
from ai_agent.tools.report.report_extract import _report_extract


def _fact(period: str, value: str) -> IndicatorFact:
    return IndicatorFact(
        subject="绿的谐波",
        period=period,
        name="营业收入",
        value=value,
        unit="元",
        column="本期",
        scope="consolidated",
        evidence=Evidence(
            file_id="f1",
            filename="绿的谐波2026Q3.pdf",
            page=12,
            page_end=None,
            section="",
            quote="",
        ),
    )


def _extract(period: str, value: str):
    return ExtractResult(facts=[_fact(period, value)])


def test_compare_to_extracts_two_periods_and_computes_change():
    calls: list[str] = []

    def fake(subjects, period, indicators):
        calls.append(period)
        return _extract(period, "420000000" if period == "2026Q3" else "380000000")

    with patch(
        "ai_agent.tools.report.report_extract.extract_indicators", side_effect=fake
    ):
        out = _report_extract(["绿的谐波"], "2026Q3", ["营业收入"], compare_to="2026Q2")

    assert calls == ["2026Q3", "2026Q2"]
    assert "2026Q2" in out and "2026Q3" in out
    assert "环比" in out
    assert "10.53%" in out


def test_without_compare_to_extracts_once():
    calls: list[str] = []

    def fake(subjects, period, indicators):
        calls.append(period)
        return _extract(period, "420000000")

    with patch(
        "ai_agent.tools.report.report_extract.extract_indicators", side_effect=fake
    ):
        out = _report_extract(["绿的谐波"], "2026Q3", ["营业收入"])
    assert calls == ["2026Q3"]
    # 抬头固定提示里本来就有「环比」二字，这里检查的是没算增减
    assert "对比期" not in out
    assert "增减" not in out.replace("要算同比/环比请把原值传给", "")


def test_compare_to_same_period_is_ignored():
    calls: list[str] = []

    def fake(subjects, period, indicators):
        calls.append(period)
        return _extract(period, "420000000")

    with patch(
        "ai_agent.tools.report.report_extract.extract_indicators", side_effect=fake
    ):
        _report_extract(["绿的谐波"], "2026Q3", ["营业收入"], compare_to="2026Q3")
    assert calls == ["2026Q3"]


def test_compare_to_with_no_previous_data_still_answers():
    def fake(subjects, period, indicators):
        if period == "2026Q2":
            return ExtractResult(missing=["没有找到 2026Q2 的对应季报"])
        return _extract(period, "420000000")

    with patch(
        "ai_agent.tools.report.report_extract.extract_indicators", side_effect=fake
    ):
        out = _report_extract(["绿的谐波"], "2026Q3", ["营业收入"], compare_to="2026Q2")
    assert "2026Q3" in out
    assert "没有找到 2026Q2" in out
