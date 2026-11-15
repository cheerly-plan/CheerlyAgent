"""研报校验只对照抽取值和计算结果，不调用模型。"""

import re

from ai_agent.report.compare import Change, SeriesCompare
from ai_agent.report.compose import ReportDocument, compose_report
from ai_agent.report.extract import Evidence, IndicatorFact, sanitize_evidence_quote
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


def test_unconverted_thousand_yuan_is_unit_error():
    """同表对比前金额须折成元；仍挂千元直接 error（中芯千元当元类硬伤）。"""
    fact = _fact(
        "2025Q3",
        "49,510,416",
        quote="单位：千元 | 营业收入 | 49,510,416",
    )
    fact.unit = "千元"
    document = ReportDocument(
        title="t",
        markdown="## 组内对比\n中芯国际营业收入 49,510,416千元\n",
        html="",
        deliverable="markdown",
    )
    result = validate_report(document, [fact], plan=_plan())
    assert not result.passed
    assert any("须折算为元" in item.message for item in result.findings)


def test_quote_says_thousand_yuan_but_unit_yuan_without_conversion():
    fact = _fact(
        "2025Q3",
        "49,510,416",
        quote="单位：千元 | 营业收入 | 49,510,416",
    )
    fact.unit = "元"
    fact.unit_converted = False
    document = ReportDocument(
        title="t",
        markdown="## 组内对比\n中芯国际营业收入 49,510,416元\n",
        html="",
        deliverable="markdown",
    )
    result = validate_report(document, [fact], plan=_plan())
    assert not result.passed
    assert any("原文为千元" in item.message and "未折算" in item.message for item in result.findings)


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
    """正文既没文件名/file_id，也没有能覆盖事实页码的「第N页」时才算缺出处。

    只去掉文件名、保留页码是**够**的：表格形式的分折会把出处统一写在表下
    （「〔以上均出自该季报第1–2页〕」），单元格里不再逐条挂文件名。
    """
    facts = [_fact("2025Q3", "100,924,954"), _fact("2024Q3", "111,000,000")]
    document = compose_report(_plan(), facts)

    # 只有文件名被替换、页码还在 → 仍算有出处
    keep_page = document.markdown.replace("file-1", "").replace(
        "中国能源建设2025年第三季度报告.pdf", "某份文件"
    )
    assert validate_report(
        ReportDocument(
            title=document.title,
            markdown=keep_page,
            html="",
            deliverable="markdown",
        ),
        facts,
        plan=_plan(),
    ).passed

    # 文件名和页码一起去掉 → 判缺出处
    no_marker = re.sub(r"第\s*\d+(?:\s*[–-]\s*\d+)?\s*页", "", keep_page)
    document.markdown = no_marker
    result = validate_report(document, facts, plan=_plan())
    assert not result.passed
    assert any("看不到出处标记" in item.message for item in result.findings)


def test_akshare_missing_page_skips_review_queue():
    fact = IndicatorFact(
        subject="中国能源建设股份有限公司",
        period="2025Q3",
        name="毛利率",
        value="48.35",
        unit="%",
        column="年初至报告期末",
        scope="consolidated",
        evidence=Evidence(
            file_id="ak-1",
            filename="akshare.stock_financial_abstract",
            page=None,
            page_end=None,
            section="财务摘要",
            quote="毛利率 采用=48.35%（akshare.stock_financial_abstract）",
        ),
    )
    plan = ReportPlan(
        title="t",
        kind="company",
        deliverable="markdown",
        subjects=["中国能源建设股份有限公司"],
        period="2025Q3",
        dimensions=[],
        indicators=["毛利率"],
        outline=[OutlineItem("本期结论", ""), OutlineItem("主要指标与出处", "")],
        gaps=[],
    )
    document = compose_report(plan, [fact])
    result = validate_report(document, [fact], plan=plan)
    assert not any("没有页码" in item.reason for item in result.review_queue)


def test_lead_yoy_inconsistency_is_error():
    facts = [
        _fact("2025Q3", "-100", name="扣非净利润", quote="扣非净利润 | -100"),
        _fact("2024Q3", "-50", name="扣非净利润", quote="扣非净利润 | -50"),
    ]
    # 手工造「结论裸同比、正文有方向词」的不一致正文。
    markdown = (
        "# t\n\n"
        "## 本期结论\n\n"
        "扣非净利润为 -100元（同比 100%）。\n\n"
        "## 同比与环比\n\n"
        "扣非净利润本期为 -100元，较 2024Q3（-50元）亏损扩大，幅度 100%。\n"
    )
    document = ReportDocument(title="t", markdown=markdown, html="", deliverable="markdown")
    series = [
        SeriesCompare(
            subject="中国能源建设股份有限公司",
            name="扣非净利润",
            scope="consolidated",
            unit="元",
            points=[("2024Q3", "-50"), ("2025Q3", "-100")],
            yoy={
                "2025Q3": Change(
                    base_period="2024Q3",
                    base_value="-50",
                    delta="-50",
                    rate="100%",
                )
            },
        )
    ]
    plan = ReportPlan(
        title="t",
        kind="company",
        deliverable="markdown",
        subjects=["中国能源建设股份有限公司"],
        period="2025Q3",
        dimensions=["同比"],
        indicators=["扣非净利润"],
        outline=[OutlineItem("本期结论", ""), OutlineItem("同比与环比", "")],
        gaps=[],
    )
    result = validate_report(document, facts, plan=plan, series=series)
    assert not result.passed
    assert any("表述不一致" in item.message for item in result.findings)


def test_sanitize_evidence_quote_fixes_breaks_and_noise():
    raw = (
        "净利润3, 70万元，同比增长783. %。"
        "□适用 √不适用 ## （三）主要会计数据"
    )
    cleaned = sanitize_evidence_quote(raw)
    assert "3, 70" not in cleaned
    assert "370" in cleaned
    assert "783%" in cleaned
    assert "□" not in cleaned
    assert "##" not in cleaned


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
    assert "validate_figures" in [tool.name for tool in TOOLS]
    assert "report_pdf" not in [tool.name for tool in TOOLS]


def test_security_code_is_not_an_unchecked_number():
    """证券代码是标识符，不是财务数字。

    实测一条季报分析里，正文的「中科飞测（688361）」被判成
    「未核对数字 688361」，凭空多出一条 error。
    """
    from ai_agent.report.validate import _body_security_codes

    assert _body_security_codes("中科飞测（688361）2026Q3") == {"688361"}
    assert _body_security_codes("证券代码：688361") == {"688361"}
    assert _body_security_codes("没有代码的句子") == set()


def test_security_code_does_not_require_plan():
    """validate_figures 由模型直接调用时没有 plan，白名单不能因此失效。"""
    facts = [_fact("2025Q3", "100,924,954")]
    document = compose_report(_plan(), facts)
    document.markdown = (
        document.markdown.replace("file-1", "")
        .replace("中国能源建设2025年第三季度报告.pdf", "某份文件")
        .replace("第12页", "〔以上均出自该季报第12页〕")
        + "\n\n中国能源建设（601868）2025Q3 营收 100,924,954 元。\n"
    )
    result = validate_report(document, facts, plan=None)
    numbers = [f.message for f in result.findings if "未核对数字 601868" in f.message]
    assert numbers == []


def test_page_level_citation_is_enough():
    """表格形式的分折把出处统一写在表下，单元格里不再逐个挂文件名。"""
    from ai_agent.report.validate import _page_cited_in_body

    fact = _fact("2025Q3", "100,924,954")
    assert fact.evidence.page == 12
    assert _page_cited_in_body(fact, "| 营业收入 | 100,924,954 |\n\n〔以上均出自该季报第1–12页〕")
    assert _page_cited_in_body(fact, "见第12页")
    # 页码覆盖不到该事实时不认
    assert not _page_cited_in_body(fact, "见第3页")
    assert not _page_cited_in_body(fact, "没有任何页码")


def test_tool_name_does_not_say_report():
    """工具名不带 report_ 前缀：它在对话里答业绩分析时也要用，不是研报专用。

    带前缀 + 「校验研报」的进度文案，会让只开知识检索的场景也显示研报字样。
    """
    names = [tool.name for tool in TOOLS]
    assert "validate_figures" in names
    assert "report_validate" not in names


def test_progress_hint_has_no_report_wording():
    from ai_agent.agents._loop.tools_exec import _tool_progress_hint

    hint = _tool_progress_hint("validate_figures", "", 1)
    assert "研报" not in hint
    assert hint == "正在校验数据与出处…"


def test_description_covers_plain_chat_analysis():
    """描述要说清「对话里直接答一段分析」也要先校验，否则模型只在研报时用。"""
    from ai_agent.prompts.tools import REPORT_VALIDATE_DESC

    assert "研报" in REPORT_VALIDATE_DESC  # 仍是合法的触发场景之一
    assert "对话" in REPORT_VALIDATE_DESC  # 但不再是唯一场景
    assert "第N页" in REPORT_VALIDATE_DESC  # 页码级出处也算
