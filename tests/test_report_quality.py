"""研报质量回归：把评审里指出的三类错误钉死，防止以后改回去。

1. 累计 1-9 月与累计 1-3 月不能相减报增长率（口径混乱 + 跨期相减）
2. 比率指标不能被套上金额单位（「8.70百万元」）
3. 银行模板必须问净息差 / 不良率 / 资本充足率，不能问毛利率
另外覆盖：期间比率跟随累计期数、正文抽取必须带原文片段与页码。
"""

from __future__ import annotations

from ai_agent.report.basis import POINT, basis_of, ratio_unit
from ai_agent.report.compare import Measure, compare_measures, format_compare
from ai_agent.report.compose import _trend_lines
from ai_agent.report.extract import _narrative_facts
from ai_agent.report.parse import OutlineItem, ReportPlan, parse_request

_SUBJECT = "中国银行股份有限公司"


def _measure(period: str, value: str, months: int) -> Measure:
    return Measure(
        subject=_SUBJECT,
        name="营业收入",
        period=period,
        value=value,
        unit="百万元",
        scope="consolidated",
        cumulative=True,
        basis="cumulative",
        coverage_months=months,
        source_key=f"f-{period}",
    )


def _plan() -> ReportPlan:
    return ReportPlan(
        title=f"{_SUBJECT} 2026Q3 财报深度分析",
        kind="company",
        deliverable="markdown",
        subjects=[_SUBJECT],
        period="2026Q3",
        dimensions=["同比", "环比"],
        indicators=["营业收入"],
        outline=[
            OutlineItem("本期结论", ""),
            OutlineItem("主要指标与出处", ""),
            OutlineItem("同比与环比", ""),
            OutlineItem("趋势", ""),
            OutlineItem("缺失与口径", ""),
        ],
        files=[],
        gaps=[],
    )


def test_incomparable_cumulative_periods_never_produce_fake_growth():
    """评审点名的 -312,358 / -63.59% 必须彻底消失，同比仍要算对。"""
    series = compare_measures(
        [
            _measure("2025Q3", "491,204", 9),
            _measure("2026Q1", "178,846", 3),
            _measure("2026H1", "356,903", 6),
            _measure("2026Q3", "526,365", 9),
        ]
    )[0]
    assert series.trend == []
    assert series.yoy["2026Q3"].rate == "7.16%"  # 同为 9 个月累计，合法同比
    text = format_compare([series])
    for fake in ("-312,358", "-63.59%", "99.56%", "47.48%"):
        assert fake not in text
    # 178,057 是 Q2 单季推导的合法基期值，只该出现在环比里，不该出现在趋势里
    assert "跨期）增减" not in text


def test_composed_report_states_incomparability_instead_of_numbers():
    series = compare_measures(
        [
            _measure("2025Q3", "491,204", 9),
            _measure("2026Q1", "178,846", 3),
            _measure("2026Q3", "526,365", 9),
        ]
    )
    lines = _trend_lines(series)
    text = "\n".join(lines)
    assert "不做跨期增减计算" in text
    assert "-63.59%" not in text
    assert "增减" not in text.replace("增减计算", "")


def test_ratio_unit_never_becomes_money():
    assert ratio_unit("加权平均净资产收益率", "百万元", "8.70") == "%"
    assert ratio_unit("净息差", "百万元人民币（百分比除外）", "1.27") == "%"
    assert ratio_unit("营业收入", "百万元", "526,365") == "百万元"


def test_period_ratio_binds_to_coverage_months():
    """ROE / 净息差是年初至今期间比率，9 个月和 3 个月同样不可比。"""
    assert basis_of("2026Q3", cumulative=True, name="净息差", unit="%", value="1.27") == (POINT, 9)
    assert basis_of("2026Q1", cumulative=True, name="净息差", unit="%", value="1.30") == (POINT, 3)
    # 不良率、资本充足率是期末快照，不跟累计期数走
    assert basis_of("2026Q3", cumulative=True, name="不良贷款率", unit="%", value="1.22") == (POINT, 0)
    assert basis_of("2026Q3", cumulative=True, name="资本充足率", unit="%", value="18.45") == (POINT, 0)

    rows = [
        Measure(
            subject=_SUBJECT, name="净息差", period=period, value=value,
            unit="%", basis=POINT, coverage_months=months,
        )
        for period, value, months in (("2026Q1", "1.30", 3), ("2026Q3", "1.27", 9))
    ]
    assert compare_measures(rows)[0].trend == []


def test_bank_template_drops_gross_margin_and_asks_the_right_questions(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [
            {
                "file_id": "f-q3",
                "filename": f"{_SUBJECT}2026年第三季度报告",
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
    assert "毛利率" not in plan.indicators
    for name in ("净息差", "不良贷款率", "不良贷款拨备覆盖率", "资本充足率", "利息净收入"):
        assert name in plan.indicators


def test_generic_company_keeps_the_old_manufacturing_template(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [
            {
                "file_id": "f-1",
                "filename": "中国能源建设2025年第三季度报告",
                "subject": "中国能源建设股份有限公司",
                "subject_aliases": ["中国能源建设"],
                "period": "2025Q3",
                "status": "ready",
                "tags": [],
                "description": "",
            }
        ],
    )
    plan = parse_request("生成中国能源建设2025Q3财报深度分析PDF研报")
    assert plan.indicators == ["营业收入", "扣非净利润", "毛利率"]


def test_narrative_extraction_needs_exact_phrase_and_keeps_evidence(monkeypatch):
    """正文比率必须精确到短语，且带回原文片段与页码，防止凭空造数。"""
    text = (
        "2026年前三季度，净息差1.27%。集团不良贷款率1.22%。"
        "核心一级资本充足率为12.04%，一级资本充足率为13.86%，资本充足率为18.45%。"
    )
    rows = [
        {
            "text": text,
            "file_id": "f-1",
            "filename": "q3.txt",
            "page": 7,
            "page_end": 8,
            "section": "三、其他重要信息",
            "subject": _SUBJECT,
            "kind": "text",
        }
    ]
    monkeypatch.setattr(
        "ai_agent.report.extract.store.search_kb_chunks_keyword", lambda *a, **k: rows
    )
    by_id = {"f-1": {"subject": _SUBJECT, "filename": "q3.txt"}}

    nim = _narrative_facts("净息差", "2026Q3", [], by_id)
    assert len(nim) == 1 and nim[0].value == "1.27" and nim[0].unit == "%"
    assert "1.27" in nim[0].evidence.quote
    assert nim[0].evidence.page == 7
    assert nim[0].basis == POINT

    capital = _narrative_facts("资本充足率", "2026Q3", [], by_id)
    assert len(capital) == 1
    assert capital[0].value == "18.45"  # 不是 12.04 / 13.86
    assert "18.45" in capital[0].evidence.quote

    # 原文里没有拨备覆盖率，就该抽不到，不能拿别的百分比凑数
    assert _narrative_facts("不良贷款拨备覆盖率", "2026Q3", [], by_id) == []


def test_narrative_strips_glued_footnote_digits(monkeypatch):
    """PDF 把上标脚注粘进数字时要剥掉：2958.40→958.40，3250,202.22→250,202.22。"""
    text = (
        "成本收入比25.72%。集团资产减值损失2958.40亿元，同比增加153.22亿元，增长19.03%。"
        "发放贷款和垫款总额3250,202.22亿元，比上年末增加15,667.30亿元。"
        "2 资产减值损失包括信用减值损失和其他资产减值损失。"
        "3 发放贷款和垫款总额和吸收存款总额均包括应计利息。"
    )
    rows = [
        {
            "text": text,
            "file_id": "f-1",
            "filename": "q3.txt",
            "page": 7,
            "page_end": 7,
            "section": "三、其他重要信息",
            "subject": _SUBJECT,
            "kind": "text",
        }
    ]
    monkeypatch.setattr(
        "ai_agent.report.extract.store.search_kb_chunks_keyword", lambda *a, **k: rows
    )
    by_id = {"f-1": {"subject": _SUBJECT, "filename": "q3.txt"}}

    impairment = _narrative_facts("资产减值损失", "2026Q3", [], by_id)
    assert len(impairment) == 1
    assert impairment[0].value == "958.40"
    assert impairment[0].unit == "亿元"

    loans = _narrative_facts("发放贷款和垫款总额", "2026Q3", [], by_id)
    assert len(loans) == 1
    assert loans[0].value == "250,202.22"

    # 没有粘连脚注时不能误剥：958.40 应保持原样
    clean_rows = [
        {
            **rows[0],
            "text": "集团资产减值损失958.40亿元。发放贷款和垫款总额250,202.22亿元。",
        }
    ]
    monkeypatch.setattr(
        "ai_agent.report.extract.store.search_kb_chunks_keyword",
        lambda *a, **k: clean_rows,
    )
    assert _narrative_facts("资产减值损失", "2026Q3", [], by_id)[0].value == "958.40"
    assert _narrative_facts("发放贷款和垫款总额", "2026Q3", [], by_id)[0].value == "250,202.22"
