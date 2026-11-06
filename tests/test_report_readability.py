"""正文可读性：口径统一声明一次，溯源集中到表格，数字原值原单位不动。

用户评审：逐条机械标注让「范围未标明」出现几十次、结论句被出处塞满。
本组用例钉住重排后的契约，尤其是「不能为了好读编造统一口径」这条底线。
"""

import re

from ai_agent.report.basis import CUMULATIVE, POINT, STOCK
from ai_agent.report.compose import compose_report
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan
from ai_agent.report.validate import validate_report

FILE_ID = "7b75379b-69a7-41ad-a670-053372460d3f"
FILENAME = "中国银行股份有限公司2026 年第三季度报告"
SUBJECT = "中国银行股份有限公司"
ANCHOR = "[7b75379b]"


def _fact(
    name: str = "营业收入",
    value: str = "526,365",
    unit: str = "百万元",
    *,
    basis: str = CUMULATIVE,
    months: int = 9,
    scope: str = "",
) -> IndicatorFact:
    return IndicatorFact(
        subject=SUBJECT,
        period="2026Q3",
        name=name,
        value=value,
        unit=unit,
        column="年初至报告期末",
        scope=scope,
        basis=basis,
        coverage_months=months,
        cumulative=basis == CUMULATIVE,
        evidence=Evidence(
            file_id=FILE_ID,
            filename=FILENAME,
            page=2,
            page_end=2,
            section="一、主要财务数据",
            quote=f"{name} | {value}",
        ),
    )


def _plan(*titles: str) -> ReportPlan:
    return ReportPlan(
        title=f"{SUBJECT} 2026Q3 财报深度分析",
        kind="company",
        deliverable="pdf",
        subjects=[SUBJECT],
        period="2026Q3",
        dimensions=["同比", "环比"],
        indicators=["营业收入", "净息差", "资产总计"],
        outline=[OutlineItem(title, "") for title in titles]
        or [OutlineItem("本期结论", ""), OutlineItem("主要指标与出处", "")],
        gaps=[],
    )


def _section(markdown: str, heading: str) -> str:
    rest = markdown.split(f"## {heading}", 1)[1]
    return rest.split("\n## ", 1)[0]


def test_reading_note_is_stated_once_without_inventing_scope():
    document = compose_report(_plan("本期结论", "主要指标与出处"), [_fact()])
    for text in (document.markdown, document.html):
        assert text.count("阅读说明") == 1
        assert "不代为推断" in text
    # 这两句是事实错误：scope 多为空，且期末余额/时点比率不是累计值
    assert "均为合并报表口径" not in document.markdown
    assert "均为年初至报告期末累计值" not in document.markdown
    # 声明里不能出现阿拉伯数字，否则会被校验器当成未核对数值
    note = "\n".join(
        line for line in document.markdown.splitlines() if line.startswith("> ")
    )
    assert note
    assert re.search(r"\d", note) is None


def test_lead_sentences_carry_only_numbers_and_change():
    facts = [
        _fact(),
        _fact("净息差", "1.27", "%", basis=POINT),
        _fact("资产总计", "41,108,449", "百万元", basis=STOCK),
    ]
    document = compose_report(_plan("本期结论", "主要指标与出处"), facts)
    lead = _section(document.markdown, "本期结论")
    assert "526,365百万元" in lead
    assert "1.27%" in lead
    assert "41,108,449百万元" in lead
    # 出处、范围、取数列、页码全部不进结论句
    for noise in ("范围未标明", "取数列", FILENAME, ANCHOR, "第2页", "（合并", "时点／期间比率"):
        assert noise not in lead
    # 累计金额仍保留口径前缀，防止被读成单季
    assert "2026年前三季度累计营业收入" in lead


def test_no_repeated_unknown_scope_anywhere_in_document():
    document = compose_report(_plan("本期结论", "主要指标与出处"), [_fact()])
    assert document.markdown.count("范围未标明") == 0
    assert document.html.count("范围未标明") == 0
    assert "／—／" in document.markdown  # 表格里用破折号，含义在声明里


def test_traceability_lives_in_the_indicator_table():
    document = compose_report(_plan("本期结论", "主要指标与出处"), [_fact()])
    table = _section(document.markdown, "主要指标与出处")
    assert "| 期间口径／范围／取数列 | 出处 |" in table
    for marker in (FILENAME, ANCHOR, "第2页", "一、主要财务数据", "年初至报告期末"):
        assert marker in table


def test_special_disclosure_moves_to_table_footnote():
    document = compose_report(
        _plan("本期结论", "主要指标与出处"), [_fact("净息差", "1.27", "%", basis=POINT)]
    )
    lead = _section(document.markdown, "本期结论")
    table = _section(document.markdown, "主要指标与出处")
    assert "原文未注明是否年化" not in lead
    assert table.count("注：原文未注明是否年化") == 1


def test_numbers_are_not_converted():
    """用户要求数字不动：不做百万元→亿元之类的单位换算。"""
    document = compose_report(_plan("本期结论", "主要指标与出处"), [_fact()])
    lead = _section(document.markdown, "本期结论")
    assert "526,365百万元" in lead
    assert "5,263.65" not in document.markdown
    assert "亿元" not in lead


def test_validation_still_passes_with_citations_only_in_table():
    facts = [_fact()]
    document = compose_report(_plan("本期结论", "主要指标与出处"), facts)
    result = validate_report(document, facts, plan=_plan("本期结论", "主要指标与出处"))
    assert result.passed, [(f.kind, f.message) for f in result.findings]
    assert result.findings == []
    assert FILE_ID in result.provenance.markdown  # 溯源附件不受影响


def test_plan_without_table_section_gets_one_automatically():
    """结论句不再自带出处，大纲缺表格节时必须补，否则全文无出处。"""
    plan = _plan("本期结论")
    facts = [_fact()]
    document = compose_report(plan, facts)
    assert "## 主要指标与出处" in document.markdown
    assert ANCHOR in _section(document.markdown, "主要指标与出处")
    assert validate_report(document, facts, plan=plan).passed


def test_invalid_request_has_no_reading_note():
    plan = ReportPlan(title="", kind="unknown", deliverable="markdown")
    document = compose_report(plan, [])
    assert "阅读说明" not in document.markdown
    assert "没有大纲" in document.markdown


# ——— 单公司报告省略主语 ———


def test_single_company_body_omits_repeated_full_name():
    """单公司报告正文不逐句复读全称：主语由标题承担，表格没有「公司」列。

    出处列里的文件名（含公司全称）是溯源信息，必须保留，不算主语复读。
    """
    facts = [_fact(), _fact("净息差", "1.27", "%", basis=POINT)]
    plan = _plan("本期结论", "主要指标与出处", "同比与环比")
    document = compose_report(plan, facts)
    # 标题保留全称
    assert SUBJECT in document.markdown.splitlines()[0]
    # 叙述句不出现全称；表格行不以「| 全称」开头（公司列已去掉）
    for line in document.markdown.splitlines()[1:]:
        if line.startswith("|"):
            assert not line.startswith(f"| {SUBJECT}")
        elif line.startswith("#"):
            continue
        else:
            assert SUBJECT not in line, line
    assert "| 公司 |" not in document.markdown  # 表格首列已去掉
    # 句子仍然完整可读
    assert "2026年前三季度累计营业收入为 526,365百万元" in document.markdown
    assert validate_report(document, facts, plan=plan).passed


def test_multi_subject_report_keeps_full_names():
    """行业/多主体报告必须保留全称：省略主语就分不清说的是哪家。"""
    peer = _fact()
    peer.subject = "中国工商银行股份有限公司"
    plan = ReportPlan(
        title="银行业 2026Q3 横向对比",
        kind="industry",
        deliverable="markdown",
        subjects=["中国银行股份有限公司", "中国工商银行股份有限公司"],
        industry="bank",
        period="2026Q3",
        dimensions=["同比"],
        indicators=["营业收入"],
        outline=[OutlineItem("覆盖范围", ""), OutlineItem("主要指标与出处", "")],
        gaps=[],
    )
    document = compose_report(plan, [_fact(), peer])
    assert "中国银行股份有限公司" in document.markdown
    assert "中国工商银行股份有限公司" in document.markdown
    assert "| 公司 |" in document.markdown  # 多主体时表格保留公司列
