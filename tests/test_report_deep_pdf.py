"""深度分析附录合并进现有 PDF：标注、门禁、默认不变、失败回退。

PDF 渲染一律打桩（不依赖 WeasyPrint / 中文字体），零真实模型调用。
沙箱里 pytest tmp_path 可能因 basetemp 已存在报 EEXIST，沿用临时目录写法。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import ai_agent.api.routers.report as report_router
import ai_agent.report.pipeline as pipeline
from ai_agent.api.schemas import ReportRequest
from ai_agent.report import review as review_store
from ai_agent.report.deep import DeepOptions
from ai_agent.report.deep.pdf_appendix import (
    APPENDIX_REVIEW_REASON,
    render_deep_appendix_html,
)
from ai_agent.report.deep.schema import (
    ChapterResult,
    ChecklistItem,
    Claim,
    Convergence,
    DataGap,
    DataSourceRow,
    DeepAnalysisResult,
    ExecutionReport,
    Falsifier,
    Mainline,
    Scenario,
    TrackingItem,
)
from ai_agent.report.extract import Evidence, ExtractResult, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan, SourceFile
from ai_agent.report.pdf import ChartSpec, PdfFile, apply_print_template
from ai_agent.report.validate import ProvenanceAttachment, ValidationResult

_SUBJECT = "中国能源建设股份有限公司"
_FILENAME = "中国能源建设2026年第三季度报告.pdf"
_REQUEST = "生成中国能源建设2026Q3财报深度分析研报"


@pytest.fixture
def out_dir():
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def _deep_result() -> DeepAnalysisResult:
    """手工构造一份带各种结论类型的结构化结果，便于断言标注与转义。"""
    return DeepAnalysisResult(
        target=_SUBJECT,
        period="2026Q3",
        mode="company",
        company_type="周期成长",
        mainlines=[Mainline("工程建设", "订单驱动", "毛利←项目结构")],
        chapters=[
            ChapterResult(
                number=2,
                title="财报可信度",
                question="报表数字能否追溯",
                method="Tier 分级 + 现金利润匹配",
                verdict="评级：中（台账内事实可支撑）",
                claims=[
                    Claim("营业收入为 100,924,954 元", "known", "Tier 1", ["e0"], 0.9),
                    Claim(
                        "护城河来自成本优势 <script>alert(1)</script>",
                        "inferred",
                        "Tier 2",
                        [],
                        0.4,
                    ),
                    Claim("客户集中度待核实", "uncertain", "Tier 3", [], 0.2),
                ],
                sources=[
                    DataSourceRow("营业收入", _FILENAME, "第 12 页", "Tier 1", "原文抽取")
                ],
                gaps=[DataGap("近5年审计报告", [2], "可信度评级不完整")],
                status="reviewed",
            )
        ],
        convergence=Convergence(
            final_chain="毛利率←产品结构（待实证）",
            moat_rating="中等",
            moat_sources={"成本优势": "未找到实证"},
        ),
        checklist=[ChecklistItem("是否有证伪信号", "⚠️", "已列示")],
        tracking=[TrackingItem("成本优势", "单位成本", "高于同行即证伪")],
        falsifiers=[Falsifier("成本优势", "单位成本连续两季高于同行", "高")],
        scenarios=[Scenario("基准", 0.6, "持平", "毛利率稳定", "维持")],
        cross_checks=["第 2 章与第 9 章评级一致"],
        execution=ExecutionReport(
            enabled_chapters=[2], llm_calls=3, notes=["stub 执行说明"]
        ),
    )


def test_appendix_has_warning_labels_and_escapes_model_text():
    html_text = render_deep_appendix_html(_deep_result())
    assert "不属于已核验财报数据" in html_text
    assert "不构成估值或买卖建议" in html_text
    assert "推断层 · 未经事实层校验" in html_text
    assert "【已知】" in html_text and "【推断】" in html_text and "【不确定】" in html_text
    assert "deep-claim-inferred" in html_text
    assert "deep-claim-uncertain" in html_text
    assert "Tier 1" in html_text and "e0" in html_text
    assert "概率（模型主观赋值）" in html_text
    assert "数据缺口总表" in html_text
    # LLM 文本必须被转义，不能把脚本注入进 PDF 的 HTML
    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text
    assert APPENDIX_REVIEW_REASON.startswith("深度分析附录已合并进 PDF")


def test_print_template_only_changes_when_appendix_is_present():
    source = "<html><head></head><body><p>正文</p></body></html>"
    font = Path("/tmp/report-cjk.ttf")
    charts = [ChartSpec("趋势", ["A", "B"], ["1", "2"])]

    plain = apply_print_template(source, title="研报", font=font, charts=charts)
    assert ".deep-appendix" not in plain
    assert "element(deep-flag)" not in plain

    appendix = '<section class="deep-appendix">附录</section>'
    merged = apply_print_template(
        source, title="研报", font=font, charts=charts, appendix_html=appendix
    )
    assert "element(deep-flag)" in merged
    assert ".deep-appendix" in merged
    assert "deep-claim-inferred" in merged
    # 附录排在图表之后、</body> 之前
    assert merged.index("<figure") < merged.index('<section class="deep-appendix">')
    assert merged.index('<section class="deep-appendix">') < merged.index("</body>")


def _fact(period: str, value: str) -> IndicatorFact:
    return IndicatorFact(
        subject=_SUBJECT,
        period=period,
        name="营业收入",
        value=value,
        unit="元",
        column="本报告期",
        scope="consolidated",
        reported_yoy="",
        evidence=Evidence(
            file_id="file-1",
            filename=_FILENAME,
            page=12,
            page_end=12,
            section="主要会计数据",
            quote=f"营业收入 | {value}",
        ),
    )


def _plan() -> ReportPlan:
    return ReportPlan(
        title=f"{_SUBJECT} 2026Q3 财报深度分析",
        kind="company",
        deliverable="markdown",
        subjects=[_SUBJECT],
        period="2026Q3",
        dimensions=["同比"],
        indicators=["营业收入", "毛利率"],
        outline=[OutlineItem("本期结论", "")],
        files=[
            SourceFile("file-1", _FILENAME, _SUBJECT, "2026Q3", "本期"),
            SourceFile("file-0", "2025Q3.pdf", _SUBJECT, "2025Q3", "同比基期"),
        ],
    )


def _fake_extract(_subjects, period, _indicators):
    values = {"2026Q3": "100,924,954", "2025Q3": "111,000,000"}
    if period in values:
        return ExtractResult(facts=[_fact(period, values[period])], missing=[])
    return ExtractResult(missing=[f"没有找到 {period} 的对应季报"])


def _mock_pipeline(monkeypatch, calls, *, fail_on_appendix: bool = False):
    """把确定性流水线收敛成可控桩：正文/校验/PDF 都不依赖外部环境。"""
    monkeypatch.setattr(pipeline, "parse_request", lambda _text: _plan())
    monkeypatch.setattr(pipeline, "extract_indicators", _fake_extract)
    monkeypatch.setattr(
        pipeline,
        "validate_report",
        lambda *_a, **_k: ValidationResult(
            passed=True, provenance=ProvenanceAttachment(markdown="# 溯源\n")
        ),
    )

    def fake_render(_document, *, charts=None, appendix_html=None, **_kwargs):
        calls.append(appendix_html)
        if fail_on_appendix and appendix_html is not None:
            raise RuntimeError("stub render boom")
        marker = b"APPENDIX" if appendix_html else b"FACT"
        return PdfFile(
            filename="x.pdf", content=b"%PDF-1.4\n" + marker, pages=2, gaps=[]
        )

    monkeypatch.setattr(pipeline, "render_report_pdf", fake_render)


def _run(out_dir, deep):
    return pipeline.run_report(
        _REQUEST, deliverable="pdf", write_pdf=True, output_dir=out_dir, deep=deep
    )


def test_deep_off_leaves_pdf_rendering_untouched(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls)
    result = _run(out_dir, None)
    assert len(calls) == 1 and calls[0] is None
    assert result.deep_analysis is None
    assert result.deep_pdf_merged is False
    assert (out_dir / result.artifacts.pdf).read_bytes() == b"%PDF-1.4\nFACT"
    assert "deep" not in report_router._payload(result)


def test_merge_default_appends_appendix_and_requires_manual_review(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls)
    result = _run(out_dir, DeepOptions(enabled=False, collect_external=False))

    assert result.deep_error == ""
    assert result.deep_pdf_merged is True
    assert len(calls) == 2
    assert calls[0] is None
    assert '<section class="deep-appendix"' in calls[1]
    assert "不属于已核验财报数据" in calls[1]
    # 同一份 PDF、同一个文件名，正文 Markdown/溯源一字不改
    assert result.artifacts.pdf.endswith(".pdf")
    assert (out_dir / result.artifacts.pdf).read_bytes() == b"%PDF-1.4\nAPPENDIX"
    assert "本层是分析推断层" not in result.document.markdown
    assert "深度分析附录" not in result.document.markdown
    assert set(result.artifacts.__dict__) == {"markdown", "provenance", "review", "pdf"}

    # 门禁重定义：合并档不能自动定稿，PDF 下载必须等人工批准
    assert any("深度分析附录" in item.reason for item in result.validation.review_queue)
    record = review_store.get_record(result.report_id, out_dir)
    assert record is not None
    assert record.status != review_store.FINALIZED
    assert any("深度分析附录" in item.reason for item in record.items)
    allowed, reason = review_store.gate(record, out_dir)
    assert not allowed and "待人工审核" in reason
    assert review_store.pdf_download_gate(result.artifacts.pdf, out_dir)[0] is False

    payload = report_router._payload(result)
    assert payload["deep"]["pdf_merged"] is True
    assert payload["deep"]["pdf_merge_error"] == ""
    assert payload["pdf"]["filename"] == result.artifacts.pdf


def test_separate_mode_keeps_fact_only_pdf_and_auto_finalizes(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls)
    result = _run(
        out_dir,
        DeepOptions(enabled=False, collect_external=False, pdf_mode="separate"),
    )
    assert len(calls) == 1 and calls[0] is None
    assert result.deep_pdf_merged is False
    assert result.validation.review_queue == []
    assert (out_dir / result.artifacts.pdf).read_bytes() == b"%PDF-1.4\nFACT"
    record = review_store.get_record(result.report_id, out_dir)
    assert record.status == review_store.FINALIZED
    assert review_store.pdf_download_gate(result.artifacts.pdf, out_dir)[0] is True
    assert report_router._payload(result)["deep"]["pdf_merged"] is False


def test_merge_failure_falls_back_to_fact_pdf_without_false_gate(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls, fail_on_appendix=True)
    result = _run(out_dir, DeepOptions(enabled=False, collect_external=False))
    assert len(calls) == 2
    assert result.deep_pdf_merged is False
    assert "stub render boom" in result.deep_pdf_error
    # 重渲染失败不是深度分析失败，独立产物照常交付
    assert result.deep_error == ""
    assert result.deep_analysis is not None
    assert (out_dir / result.deep_artifacts["markdown"]).is_file()
    assert (out_dir / result.artifacts.pdf).read_bytes() == b"%PDF-1.4\nFACT"
    # 没有误追加审核项：不该因为合并失败把纯事实层 PDF 卡在人工队列里
    assert result.validation.review_queue == []
    record = review_store.get_record(result.report_id, out_dir)
    assert record.status == review_store.FINALIZED
    assert report_router._payload(result)["deep"]["pdf_merged"] is False
    assert "stub render boom" in report_router._payload(result)["deep"]["pdf_merge_error"]


def test_markdown_delivery_never_merges(out_dir, monkeypatch):
    """只要没有 PDF（markdown 交付），merge 自动退化，不额外渲染。"""
    calls: list = []
    _mock_pipeline(monkeypatch, calls)
    result = pipeline.run_report(
        _REQUEST,
        deliverable="markdown",
        write_pdf=False,
        output_dir=out_dir,
        deep=DeepOptions(enabled=False, collect_external=False),
    )
    assert calls == []
    assert result.pdf is None
    assert result.deep_pdf_merged is False
    assert result.deep_analysis is not None


def test_api_request_defaults_to_merge():
    request = ReportRequest(request=_REQUEST)
    assert request.deep == "off"
    assert request.deep_pdf == "merge"
    assert ReportRequest(request=_REQUEST, deep_pdf="separate").deep_pdf == "separate"
