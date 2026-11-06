"""研报校验只对照抽取值和计算结果，不调用模型。"""

from ai_agent.report.compose import compose_report
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan
from ai_agent.report.validate import format_validation, validate_report
from ai_agent.tools.registry import TOOLS
from ai_agent.tools.report.report_validate import report_validate_tool


def _fact(
    period: str,
    value: str,
    *,
    name: str = "营业收入",
    quote: str | None = None,
    file_id: str = "file-1",
    filename: str = "中国能源建设2025年第三季度报告.pdf",
    page: int | None = 12,
) -> IndicatorFact:
    return IndicatorFact(
        subject="中国能源建设股份有限公司",
        period=period,
        name=name,
        value=value,
        unit="元",
        column="本报告期",
        scope="consolidated",
        evidence=Evidence(
            file_id=file_id,
            filename=filename,
            page=page,
            page_end=page,
            section="主要会计数据",
            quote=quote if quote is not None else f"营业收入 | {value}",
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
        indicators=["营业收入"],
        outline=[
            OutlineItem("本期结论", ""),
            OutlineItem("主要指标与出处", ""),
            OutlineItem("同比与环比", ""),
            OutlineItem("缺失与口径", ""),
        ],
        gaps=["没有找到 2025Q2 的季报，环比基期无法调取"],
    )


def test_clean_report_passes_with_provenance():
    facts = [_fact("2025Q3", "100,924,954"), _fact("2024Q3", "111,000,000")]
    document = compose_report(_plan(), facts)
    result = validate_report(document, facts, plan=_plan())
    assert result.passed
    assert result.review_queue == []
    assert "100,924,954" in result.provenance.markdown
    assert "营业收入 | 100,924,954" in result.provenance.markdown
    assert "file-1" in result.provenance.markdown
    assert "校验通过" in format_validation(result)


def test_unknown_number_goes_to_review_queue():
    facts = [_fact("2025Q3", "100,924,954"), _fact("2024Q3", "111,000,000")]
    document = compose_report(_plan(), facts)
    document.markdown += "\n\n额外写了 999,888,777元。\n"
    result = validate_report(document, facts, plan=_plan())
    assert not result.passed
    assert any(item.value == "999,888,777" for item in result.review_queue)
    assert any("未核对数字" in item.message for item in result.findings)


def test_quote_must_contain_value():
    bad = _fact("2025Q3", "100,924,954", quote="营业收入 | 别的数")
    document = compose_report(_plan(), [bad, _fact("2024Q3", "111,000,000")])
    result = validate_report(document, [bad, _fact("2024Q3", "111,000,000")], plan=_plan())
    assert not result.passed
    assert any("未出现在原文片段" in item.message for item in result.findings)
    assert any("未出现在原文片段" in item.reason for item in result.review_queue)


def test_missing_citation_marker_fails():
    facts = [_fact("2025Q3", "100,924,954"), _fact("2024Q3", "111,000,000")]
    document = compose_report(_plan(), facts)
    document.markdown = document.markdown.replace("file-1", "").replace(
        "中国能源建设2025年第三季度报告.pdf", "某份文件"
    )
    result = validate_report(document, facts, plan=_plan())
    assert not result.passed
    assert any("看不到出处标记" in item.message for item in result.findings)


def test_tool_returns_summary_and_stays_off_chat_loop():
    facts = [_fact("2025Q3", "100,924,954"), _fact("2024Q3", "111,000,000")]
    document = compose_report(_plan(), facts)
    text = report_validate_tool.run(
        markdown=document.markdown,
        title=document.title,
        period="2025Q3",
        facts=[
            {
                "subject": fact.subject,
                "name": fact.name,
                "period": fact.period,
                "value": fact.value,
                "unit": fact.unit,
                "scope": fact.scope,
                "file_id": fact.evidence.file_id,
                "filename": fact.evidence.filename,
                "page": fact.evidence.page,
                "page_end": fact.evidence.page_end,
                "section": fact.evidence.section,
                "quote": fact.evidence.quote,
            }
            for fact in facts
        ],
    )
    assert "校验通过" in text
    assert "溯源附件" in text
    # 分层契约：校验积木（无路径参数、不写盘）已注册进聊天循环；
    # 渲染工具（含落盘与 M2 审核门禁）仍只能由服务端流水线触发。
    assert "report_validate" in [tool.name for tool in TOOLS]
    assert "report_pdf" not in [tool.name for tool in TOOLS]
