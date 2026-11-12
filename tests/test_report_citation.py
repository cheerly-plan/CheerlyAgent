"""正文出处用 8 位短锚点，完整 file_id 只留在溯源附件和 JSON。

背景：每条结论后面跟一串 36 字符 UUID 会把数字淹没。短锚点既是全 ID
的前缀（可回查溯源表），又带方括号便于校验器整体剔除，避免锚点里的
数字（如 75379）被当成未核对金额。
"""

import re

from ai_agent.report.compose import compose_report
from ai_agent.report.extract import Evidence, IndicatorFact, citation_anchor
from ai_agent.report.parse import OutlineItem, ReportPlan
from ai_agent.report.supplemental import QuarterView, SupplementalResult, YearEndView
from ai_agent.report.validate import validate_report

FILE_ID = "7b75379b-69a7-41ad-a670-053372460d3f"
FILENAME = "中国银行股份有限公司2026 年第三季度报告"
ANCHOR = "[7b75379b]"
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _evidence(quote: str) -> Evidence:
    return Evidence(
        file_id=FILE_ID,
        filename=FILENAME,
        page=2,
        page_end=2,
        section="一、主要财务数据",
        quote=quote,
    )


def _fact(value: str = "526,365,000,000", period: str = "2026Q3") -> IndicatorFact:
    return IndicatorFact(
        subject="中国银行股份有限公司",
        period=period,
        name="营业收入",
        value=value,
        unit="元",
        column="年初至报告期末",
        scope="consolidated",
        evidence=_evidence(f"营业收入 | {value}"),
    )


def _plan() -> ReportPlan:
    return ReportPlan(
        title="中国银行股份有限公司 2026Q3 财报深度分析",
        kind="company",
        deliverable="pdf",
        subjects=["中国银行股份有限公司"],
        period="2026Q3",
        dimensions=["同比", "环比"],
        indicators=["营业收入"],
        outline=[
            OutlineItem("本期结论", ""),
            OutlineItem("主要指标与出处", ""),
            OutlineItem("同比与环比", ""),
            OutlineItem("缺失与口径", ""),
        ],
        gaps=[],
    )


def test_anchor_is_a_prefix_of_full_id():
    assert citation_anchor(FILE_ID) == ANCHOR
    assert FILE_ID.startswith(ANCHOR.strip("[]"))
    assert citation_anchor("") == ""
    # 非 UUID 的测试夹具也能用，别假设一定是 36 字符
    assert citation_anchor("file-1") == "[file-1]"


def test_body_has_short_anchor_and_no_full_uuid():
    document = compose_report(_plan(), [_fact(), _fact("491,204", "2025Q3")])
    for text in (document.markdown, document.html):
        assert _UUID.search(text) is None
        assert ANCHOR in text
        assert FILENAME in text
        assert "第2页" in text
        assert "一、主要财务数据" in text


def test_provenance_keeps_full_file_id():
    document = compose_report(_plan(), [_fact()])
    result = validate_report(document, [_fact()], plan=_plan())
    assert FILE_ID in result.provenance.markdown
    assert result.provenance.items[0]["file_id"] == FILE_ID
    assert result.provenance.items[0]["filename"] == FILENAME
    # 溯源附件不受展示层改动影响
    assert f"{FILENAME}（{FILE_ID}）" in result.provenance.markdown


def test_anchor_digits_are_not_reported_as_unchecked_numbers():
    """短锚点里的 7 / 75379 不是金额，校验器扫数字前必须整体剔除。"""
    facts = [_fact(), _fact("491,204,000,000", "2025Q3")]
    document = compose_report(_plan(), facts)
    result = validate_report(document, facts, plan=_plan())
    assert result.passed, [(f.kind, f.message) for f in result.findings]
    assert result.findings == []
    assert result.review_queue == []


def test_citation_gate_still_requires_a_marker():
    facts = [_fact(), _fact("491,204,000,000", "2025Q3")]
    document = compose_report(_plan(), facts)
    document.markdown = document.markdown.replace(ANCHOR, "").replace(FILENAME, "某份文件")
    result = validate_report(document, facts, plan=_plan())
    assert not result.passed
    assert any("看不到出处标记" in item.message for item in result.findings)


def test_supplemental_citations_use_short_anchor_too():
    quote = (
        "单位：百万元 项目 | 本报告期 | 年初至报告期末 "
        "营业收入 | 169,462 | 4.48 | 526,365 | 7.16；不良贷款率1.22%，比上年末下降0.01个百分点"
    )
    views = SupplementalResult(
        quarters=[
            QuarterView(
                name="营业收入",
                unit="百万元",
                cumulative_value="526,365",
                cumulative_yoy="7.16%",
                quarter_value="169,462",
                quarter_yoy="4.48%",
                evidence=_evidence(quote),
            )
        ],
        year_ends=[
            YearEndView(
                name="不良贷款率",
                value="1.22",
                unit="%",
                change="-0.01个百分点",
                evidence=_evidence(quote),
            )
        ],
    )
    plan = _plan()
    plan.outline.extend([OutlineItem("单季节奏", ""), OutlineItem("较上年末变动", "")])
    facts = [_fact(), _fact("491,204", "2025Q3")]
    document = compose_report(plan, facts, views)
    assert _UUID.search(document.markdown) is None
    assert document.markdown.count(ANCHOR) >= 3
    assert _UUID.search(document.html) is None
