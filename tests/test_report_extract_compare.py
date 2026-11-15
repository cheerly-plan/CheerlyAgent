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

    def fake(subjects, period, indicators, *, basis="cumulative"):
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

    def fake(subjects, period, indicators, *, basis="cumulative"):
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

    def fake(subjects, period, indicators, *, basis="cumulative"):
        calls.append(period)
        return _extract(period, "420000000")

    with patch(
        "ai_agent.tools.report.report_extract.extract_indicators", side_effect=fake
    ):
        _report_extract(["绿的谐波"], "2026Q3", ["营业收入"], compare_to="2026Q3")
    assert calls == ["2026Q3"]


def test_compare_to_with_no_previous_data_still_answers():
    def fake(subjects, period, indicators, *, basis="cumulative"):
        if period == "2026Q2":
            return ExtractResult(missing=["没有找到 2026Q2 的对应季报"])
        return _extract(period, "420000000")

    with patch(
        "ai_agent.tools.report.report_extract.extract_indicators", side_effect=fake
    ):
        out = _report_extract(["绿的谐波"], "2026Q3", ["营业收入"], compare_to="2026Q2")
    assert "2026Q3" in out
    assert "没有找到 2026Q2" in out


def _stub_two_columns(basis_seen: list[str]):
    """模拟季报双列：单季 / 累计。返回按 basis 选到正确列的桩。"""
    from ai_agent.report.extract import Evidence, ExtractResult, IndicatorFact

    def fake(subjects, period, indicators, *, basis="cumulative"):
        basis_seen.append(basis)
        value, column, cum = (
            ("673,041,663.13", "本报告期", False)
            if basis == "single"
            else ("1,614,059,366.89", "年初至报告期末", True)
        )
        return ExtractResult(
            facts=[
                IndicatorFact(
                    subject="中科飞测",
                    period=period,
                    name="营业收入",
                    value=value,
                    unit="元",
                    column=column,
                    scope="consolidated",
                    cumulative=cum,
                    basis=basis,
                    evidence=Evidence(
                        file_id="f1",
                        filename="中科飞测2026Q3.pdf",
                        page=1,
                        page_end=None,
                        section="",
                        quote="",
                    ),
                )
            ]
        )

    return fake


def test_basis_single_extracts_single_quarter():
    seen: list[str] = []
    with patch(
        "ai_agent.tools.report.report_extract.extract_indicators",
        side_effect=_stub_two_columns(seen),
    ):
        out = _report_extract(["中科飞测"], "2026Q3", ["营业收入"], basis="single")
    assert seen == ["single"]
    assert "673,041,663.13" in out


def test_basis_both_extracts_both_columns():
    """both 时单季与累计各抽一遍，两条都要出现。

    季报两列覆盖的科目并不相同：营收两列都有，总资产这类只有期末列。
    只给一种口径，正文里另一种数字就全成了「未核对数字」。
    """
    seen: list[str] = []
    with patch(
        "ai_agent.tools.report.report_extract.extract_indicators",
        side_effect=_stub_two_columns(seen),
    ):
        out = _report_extract(["中科飞测"], "2026Q3", ["营业收入"], basis="both")
    assert set(seen) == {"single", "cumulative"}
    assert "673,041,663.13" in out
    assert "1,614,059,366.89" in out
