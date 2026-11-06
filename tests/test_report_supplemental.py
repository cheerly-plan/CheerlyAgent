"""单季视图 / 较上年末 / 数据观察的回归评测集。

钉住这一轮的几个具体错误：单季值必须来自官方「本报告期」列而不是累计倒推；
一句里两个指标的「比上年末」变动不能串到后一个；期末时点余额不做跨年同比；
净息差不能替原文断言年化；观察节不许出现因果归因。
"""

from __future__ import annotations

import uuid

import pytest

from ai_agent.report.basis import CUMULATIVE, STOCK, basis_of, disclosure_note
from ai_agent.report.compare import STOCK_NOTE, Measure, compare_measures
from ai_agent.report.compose import compose_report
from ai_agent.report.extract import Evidence
from ai_agent.report.parse import OutlineItem, ReportPlan
from ai_agent.report.supplemental import (
    QuarterView,
    SupplementalResult,
    YearEndView,
    _change_after,
    extract_supplemental,
)
from ai_agent.report.validate import validate_report

_MAIN_TABLE = """单位：百万元
<table>
<tr><td>项目</td><td>本报告期</td><td>本报告期比上年同期增减(%)</td><td>年初至报告期末</td><td>年初至报告期末比上年同期增减(%)</td></tr>
<tr><td>营业收入</td><td>169,462</td><td>4.48</td><td>526,365</td><td>7.16</td></tr>
<tr><td>净利润</td><td>67,201</td><td>5.91</td><td>199,232</td><td>5.09</td></tr>
<tr><td>基本和稀释每股收益(人民币元)</td><td>0.19</td><td>3.98</td><td>0.55</td><td>0.27</td></tr>
</table>
"""
_NARRATIVE = """2026年前三季度，集团实现净利润1,992.32亿元。净息差1.27%。
集团不良贷款总额3,041.10亿元，不良贷款率1.22%，比上年末下降0.01个百分点，
不良贷款拨备覆盖率198.30%，比上年末下降2.07个百分点。
发放贷款和垫款总额  $ ^{3} $ 250,202.22亿元，比上年末增加15,667.30亿元，增长6.68%。
吸收存款总额  $ ^{3} $ 270,638.63亿元，比上年末增加8,814.32亿元，增长3.37%。
9月末，集团资产总计411,084.49亿元，比上年末增加27,503.73亿元，增长7.17%。
"""


@pytest.fixture
def env(monkeypatch):
    file_id = uuid.uuid4()
    filename = "中国银行股份有限公司2026 年第三季度报告"
    files = [
        {
            "file_id": file_id,
            "filename": filename,
            "status": "ready",
            "subject": "中国银行股份有限公司",
            "subject_aliases": ["中行"],
            "period": "2026Q3",
        }
    ]
    monkeypatch.setattr(
        "ai_agent.report.supplemental.store.list_kb_files", lambda: files
    )

    def search(keywords, **kwargs):
        kind = (kwargs.get("kinds") or ["text"])[0]
        text = _MAIN_TABLE if kind == "table" else _NARRATIVE
        return [
            {
                "file_id": file_id,
                "filename": filename,
                "page": 2 if kind == "table" else 7,
                "page_end": None,
                "section": "一、主要财务数据" if kind == "table" else "三、其他重要信息",
                "text": text,
                "kind": kind,
            }
        ]

    monkeypatch.setattr(
        "ai_agent.report.supplemental.store.search_kb_chunks_keyword", search
    )
    return file_id, filename


def _plan() -> ReportPlan:
    return ReportPlan(
        title="中国银行股份有限公司 2026Q3 财报深度分析",
        kind="company",
        deliverable="markdown",
        subjects=["中国银行股份有限公司"],
        period="2026Q3",
        dimensions=["同比", "环比"],
        indicators=["营业收入", "净利润", "净息差"],
        outline=[
            OutlineItem("单季节奏", ""),
            OutlineItem("较上年末变动", ""),
            OutlineItem("数据观察", ""),
        ],
    )


def _evidence(file_id, filename) -> Evidence:
    return Evidence(
        file_id=str(file_id),
        filename=filename,
        page=2,
        page_end=None,
        section="一、主要财务数据",
        quote="单位：百万元 项目 | 本报告期 | 本报告期比上年同期增减(%) | "
        "年初至报告期末 | 年初至报告期末比上年同期增减(%)\n"
        "营业收入 | 169,462 | 4.48 | 526,365 | 7.16",
    )


def _views(file_id, filename) -> SupplementalResult:
    return SupplementalResult(
        quarters=[
            QuarterView(
                name="营业收入",
                unit="百万元",
                cumulative_value="526,365",
                cumulative_yoy="7.16%",
                quarter_value="169,462",
                quarter_yoy="4.48%",
                evidence=_evidence(file_id, filename),
            )
        ],
        year_ends=[
            YearEndView(
                name="不良贷款率",
                value="1.22",
                unit="%",
                change="-0.01个百分点",
                evidence=Evidence(
                    file_id=str(file_id),
                    filename=filename,
                    page=7,
                    page_end=None,
                    section="三、其他重要信息",
                    quote="不良贷款率1.22%，比上年末下降0.01个百分点",
                ),
            )
        ],
    )


def test_official_quarter_column_is_used_instead_of_cumulative_derivation(env):
    file_id, _ = env
    views = extract_supplemental(
        ["中国银行股份有限公司"], "2026Q3", ["营业收入", "净利润"]
    )
    revenue = next(item for item in views.quarters if item.name == "营业收入")
    assert revenue.quarter_value == "169,462"
    assert revenue.quarter_yoy == "4.48%"
    assert revenue.cumulative_value == "526,365"
    assert revenue.cumulative_yoy == "7.16%"
    assert "169,462" in revenue.evidence.quote
    assert "4.48" in revenue.evidence.quote
    assert views.gaps == []


def test_quarter_view_only_keeps_requested_indicators(env):
    views = extract_supplemental(["中国银行股份有限公司"], "2026Q3", ["营业收入"])
    assert [item.name for item in views.quarters] == ["营业收入"]


def test_missing_both_columns_is_reported_as_gap(env, monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.supplemental.store.search_kb_chunks_keyword",
        lambda *a, **k: [],
    )
    views = extract_supplemental(["中国银行股份有限公司"], "2026Q3", ["营业收入"])
    assert views.quarters == []
    assert any("没有同时披露单季列与累计列" in item for item in views.gaps)


def test_year_end_changes_stay_with_their_own_indicator(env):
    views = extract_supplemental(["中国银行股份有限公司"], "2026Q3", [])
    changes = {item.name: item.change for item in views.year_ends}
    assert changes["不良贷款率"] == "-0.01个百分点"
    assert changes["不良贷款拨备覆盖率"] == "-2.07个百分点"


def test_year_end_prefers_growth_rate_over_amount_increment(env):
    views = extract_supplemental(["中国银行股份有限公司"], "2026Q3", [])
    changes = {item.name: item.change for item in views.year_ends}
    assert changes["发放贷款和垫款总额"] == "+6.68%"
    assert changes["吸收存款总额"] == "+3.37%"
    assert changes["资产总计"] == "+7.17%"
    assert "15,667.30" not in " ".join(changes.values())


def test_change_after_needs_an_original_direction_word():
    assert _change_after("比上年末下降0.01个百分点。", 0).startswith("-")
    assert _change_after("比上年末增长6.68%。", 0).startswith("+")
    assert _change_after("本期没有比较。", 0) == ""


def test_stock_series_does_not_produce_cross_period_yoy():
    rows = [
        Measure(
            subject="中行",
            name="发放贷款和垫款总额",
            period=period,
            value=value,
            unit="亿元",
            basis=STOCK,
            coverage_months=0,
        )
        for period, value in (("2025Q3", "233,530.68"), ("2026Q3", "250,202.22"))
    ]
    series = compare_measures(rows)[0]
    assert series.yoy == {}
    assert series.qoq == {}
    assert STOCK_NOTE in series.gaps
    assert not series.trend


def test_profit_total_is_not_a_stock_balance():
    assert basis_of("2026Q3", cumulative=True, name="利润总额", unit="百万元", value="1")[0] == CUMULATIVE
    assert basis_of("2026Q3", cumulative=True, name="发放贷款和垫款总额", unit="亿元", value="1")[0] == STOCK


def test_unannualized_ratio_is_flagged_without_inventing_a_definition():
    assert "未注明是否年化" in disclosure_note("净息差")
    assert "不自行折算" in disclosure_note("净息差")
    assert disclosure_note("营业收入") == ""


def test_new_sections_render_and_stay_free_of_causal_claims(env):
    file_id, filename = env
    views = _views(file_id, filename)
    document = compose_report(_plan(), [], views=views)
    assert "## 单季节奏" in document.markdown
    assert "## 较上年末变动" in document.markdown
    assert "## 数据观察" in document.markdown
    assert "169,462百万元" in document.markdown
    assert "4.48%" in document.markdown
    assert "-0.01个百分点" in document.markdown
    observation = document.markdown.split("## 数据观察")[1]
    for forbidden in ("因为", "由于", "导致", "所以", "主要来自"):
        assert forbidden not in observation
    assert "不构成因果归因" in observation
    assert validate_report(document, [], plan=_plan(), series=[], views=views).passed


def test_validate_rejects_view_number_missing_from_quote(env):
    file_id, filename = env
    views = _views(file_id, filename)
    views.quarters[0].quarter_value = "999,999"
    document = compose_report(_plan(), [], views=views)
    result = validate_report(document, [], plan=_plan(), series=[], views=views)
    assert not result.passed
    assert any("视图数值" in item.message for item in result.findings)


def test_validate_rejects_direction_not_supported_by_quote(env):
    file_id, filename = env
    views = _views(file_id, filename)
    views.year_ends[0].evidence.quote = "不良贷款率1.22%，比上年末增长0.01个百分点"
    document = compose_report(_plan(), [], views=views)
    result = validate_report(document, [], plan=_plan(), series=[], views=views)
    assert not result.passed
    assert any("变动方向" in item.message for item in result.findings)
