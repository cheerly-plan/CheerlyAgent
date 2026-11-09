"""双 PDF 档：底稿与推断层分别成文件、分别审核、分别定稿。

PDF 渲染一律打桩（不依赖 WeasyPrint / 中文字体），零真实模型调用。
沙箱里 pytest tmp_path 可能因 basetemp 已存在报 EEXIST，沿用临时目录写法。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import ai_agent.api.routers.report as report_router
import ai_agent.report.pipeline as pipeline
from ai_agent.api.schemas import ReportRequest
from ai_agent.core.config import settings
from ai_agent.report import review as review_store
from ai_agent.report.cleanup import _ARTIFACT
from ai_agent.report.deep import DeepOptions
from ai_agent.report.deep.pdf_appendix import (
    deep_standalone_document,
    render_deep_appendix_html,
)
from ai_agent.report.deep.schema import (
    ChapterResult,
    Claim,
    DeepAnalysisResult,
    ExecutionReport,
)
from ai_agent.report.extract import Evidence, ExtractResult, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan, SourceFile
from ai_agent.report.pdf import PdfFile, apply_print_template
from ai_agent.report.validate import ProvenanceAttachment, ValidationResult

_SUBJECT = "中国能源建设股份有限公司"
_FILENAME = "中国能源建设2026年第三季度报告.pdf"
_REQUEST = "生成中国能源建设2026Q3财报深度分析研报"


@pytest.fixture
def out_dir():
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def _deep_result() -> DeepAnalysisResult:
    return DeepAnalysisResult(
        target=_SUBJECT,
        period="2026Q3",
        mode="company",
        chapters=[
            ChapterResult(
                number=2,
                title="财报可信度",
                question="报表数字能否追溯",
                method="Tier 分级 + 现金利润匹配",
                verdict="评级：中（台账内事实可支撑）",
                claims=[
                    Claim("护城河来自成本优势 <script>alert(1)</script>",
                          "inferred", "Tier 2", [], 0.4),
                ],
                status="reviewed",
            )
        ],
        execution=ExecutionReport(enabled_chapters=[2], llm_calls=0, notes=["stub"]),
    )


def test_standalone_document_is_complete_html_with_warning():
    html_text = deep_standalone_document(_deep_result())
    assert html_text.startswith("<!DOCTYPE html><html")
    assert "<head>" in html_text and "</body></html>" in html_text
    assert "deep-appendix deep-standalone" in html_text
    assert "独立深度投资分析报告（推断层）" in html_text
    assert "配套生成的" in html_text and "事实层底稿 PDF" in html_text
    assert "独立的人工审核记录" in html_text
    assert "不属于已核验财报数据" in html_text
    assert "不构成估值或买卖建议" in html_text
    assert "【推断】" in html_text
    # 独立文件不是任何报告的附录：去掉「附录」字样，也不再自称「本附录」
    assert "本附录" not in html_text
    assert "<h2>附录 " not in html_text
    assert "<h2>A · 执行报告</h2>" in html_text
    # 模型文本仍必须转义
    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text


def test_merge_appendix_wording_is_unchanged():
    """兼容档的附录文案保持原样，别被 standalone 改造带跑。"""
    html_text = render_deep_appendix_html(_deep_result(), standalone=False)
    assert 'class="deep-appendix"' in html_text
    assert "deep-standalone" not in html_text
    assert "本附录" in html_text
    assert "<h2>附录 A · 执行报告</h2>" in html_text


def test_deep_css_flag_injects_styles_without_appending_html():
    source = "<html><head></head><body><p>推断层正文</p></body></html>"
    font = Path("/tmp/report-cjk.ttf")
    plain = apply_print_template(source, title="研报", font=font)
    assert ".deep-appendix" not in plain

    styled = apply_print_template(source, title="研报", font=font, deep_css=True)
    assert ".deep-appendix.deep-standalone" in styled
    assert "element(deep-flag)" in styled
    assert "deep-claim-inferred" in styled
    # 只注入 CSS，不追加任何附录 HTML
    assert "<section" not in styled
    assert styled.count("<p>推断层正文</p>") == 1


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


def _mock_pipeline(monkeypatch, calls, *, fail_deep_pdf: bool = False):
    """把确定性流水线收敛成可控桩：两份 PDF 都不依赖真实渲染环境。"""
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
        calls.append(("fact", appendix_html))
        return PdfFile(filename="x.pdf", content=b"%PDF-1.4\nFACT", pages=2, gaps=[])

    def fake_deep_render(source, *, title, deep_css=False, **_kwargs):
        calls.append(("deep", source, title, deep_css))
        if fail_deep_pdf:
            raise RuntimeError("stub deep render boom")
        return PdfFile(filename="deep.pdf", content=b"%PDF-1.4\nDEEP", pages=3, gaps=[])

    monkeypatch.setattr(pipeline, "render_report_pdf", fake_render)
    monkeypatch.setattr(pipeline, "render_html_pdf", fake_deep_render)


def _run(out_dir, deep):
    return pipeline.run_report(
        _REQUEST, deliverable="pdf", write_pdf=True, output_dir=out_dir, deep=deep
    )


def test_separate_default_produces_two_pdfs_and_two_review_records(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls)
    result = _run(out_dir, DeepOptions(enabled=False, collect_external=False))

    assert result.deep_error == ""
    assert result.deep_pdf_merged is False
    assert result.deep_pdf is not None
    assert result.deep_pdf_review_id == f"{result.report_id}{review_store.DEEP_SUFFIX}"
    # 两次渲染：一次事实层（不附加录），一次独立推断层（deep_css=True）
    assert [c[0] for c in calls] == ["fact", "deep"]
    assert calls[0][1] is None
    assert calls[1][3] is True
    assert "独立深度投资分析报告" in calls[1][1]
    assert "财报深度分析报告" in calls[1][2]
    assert "推断层" in calls[1][2]

    fact_name = result.artifacts.pdf
    deep_name = result.deep_artifacts["pdf"]
    assert fact_name.endswith(".pdf") and not fact_name.endswith(".deep.pdf")
    assert deep_name.endswith(".deep.pdf")
    assert "底稿" not in deep_name
    assert "财报深度分析报告" in deep_name
    assert result.report_id in deep_name
    assert (out_dir / fact_name).read_bytes() == b"%PDF-1.4\nFACT"
    assert (out_dir / deep_name).read_bytes() == b"%PDF-1.4\nDEEP"
    assert set(result.artifacts.__dict__) == {"markdown", "provenance", "review", "pdf"}

    base = review_store.get_record(result.report_id, out_dir)
    deep_record = review_store.get_record(result.deep_pdf_review_id, out_dir)
    assert base is not None and deep_record is not None
    assert base.status == review_store.FINALIZED
    assert base.auto_finalized is True
    assert deep_record.status == review_store.PENDING
    assert deep_record.auto_finalized is False
    assert len(deep_record.items) == 1
    assert "独立深度分析 PDF" in deep_record.items[0].reason
    # 深度记录指纹覆盖推断层三份产物
    assert deep_record.artifacts["pdf"] == deep_name
    assert deep_record.artifacts["markdown"].endswith(".deep.md")
    assert deep_record.artifacts["provenance"].endswith(".deep.json")


def test_base_pdf_auto_finalizes_but_deep_pdf_is_blocked(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls)
    result = _run(out_dir, DeepOptions(enabled=False, collect_external=False))

    assert result.validation.review_queue == []
    assert review_store.pdf_download_gate(result.artifacts.pdf, out_dir)[0] is True
    allowed, reason = review_store.pdf_download_gate(result.deep_artifacts["pdf"], out_dir)
    assert allowed is False
    assert "待人工审核" in reason
    # 底稿的审核记录里没有推断层审核项
    base = review_store.get_record(result.report_id, out_dir)
    assert all("独立深度" not in item.reason for item in base.items)


def test_deep_pdf_download_passes_after_approve_and_finalize(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls)
    result = _run(out_dir, DeepOptions(enabled=False, collect_external=False))
    deep_id = result.deep_pdf_review_id

    review_store.decide(deep_id, "approve", "admin", output_dir=out_dir)
    # 批准了但未定稿，仍不放行
    assert review_store.pdf_download_gate(result.deep_artifacts["pdf"], out_dir)[0] is False
    review_store.finalize(deep_id, "admin", output_dir=out_dir)

    allowed, reason = review_store.pdf_download_gate(result.deep_artifacts["pdf"], out_dir)
    assert allowed is True and reason == "已定稿"
    # 两条记录命运独立：底稿早已定稿，推断层刚定稿
    assert review_store.get_record(result.report_id, out_dir).status == review_store.FINALIZED
    assert review_store.get_record(deep_id, out_dir).status == review_store.FINALIZED


def test_deep_pdf_bytes_are_bound_by_content_hash(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls)
    result = _run(out_dir, DeepOptions(enabled=False, collect_external=False))
    deep_id = result.deep_pdf_review_id
    review_store.decide(deep_id, "approve", "admin", output_dir=out_dir)
    review_store.finalize(deep_id, "admin", output_dir=out_dir)

    (out_dir / result.deep_artifacts["pdf"]).write_bytes(b"%PDF-1.4\nTAMPERED")
    record = review_store.get_record(deep_id, out_dir)
    assert record.status == review_store.PENDING  # _verify 自动作废定稿
    allowed, reason = review_store.gate(record, out_dir)
    assert allowed is False
    assert "状态为 pending" in reason or "改动" in reason


def test_merge_mode_has_no_standalone_deep_pdf_or_second_record(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls)
    result = _run(
        out_dir,
        DeepOptions(enabled=False, collect_external=False, pdf_mode="merge"),
    )
    assert [c[0] for c in calls] == ["fact", "fact"]
    assert result.deep_pdf is None
    assert "pdf" not in result.deep_artifacts
    assert result.deep_pdf_review_id == ""
    assert review_store.get_record(f"{result.report_id}-deep", out_dir) is None


def test_standalone_render_failure_is_isolated_from_fact_layer(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls, fail_deep_pdf=True)
    result = _run(out_dir, DeepOptions(enabled=False, collect_external=False))

    assert "stub deep render boom" in result.deep_pdf_error
    assert result.deep_error == ""
    assert result.deep_analysis is not None
    assert result.deep_pdf is None
    assert "pdf" not in result.deep_artifacts
    assert (out_dir / result.deep_artifacts["markdown"]).is_file()
    assert (out_dir / result.artifacts.pdf).read_bytes() == b"%PDF-1.4\nFACT"
    # 渲染失败不能凭空造一条没有文件的审核记录，也不能拖住底稿定稿
    assert result.deep_pdf_review_id == ""
    assert review_store.get_record(result.report_id, out_dir).status == review_store.FINALIZED


def test_cleanup_recognizes_deep_pdf_and_removes_second_record(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls)
    result = _run(out_dir, DeepOptions(enabled=False, collect_external=False))
    deep_name = result.deep_artifacts["pdf"]

    parsed = _ARTIFACT.match(deep_name)
    assert parsed is not None
    assert parsed.group("id") == result.report_id
    assert review_store.get_record(result.deep_pdf_review_id, out_dir) is not None

    removed = review_store.remove_records([result.report_id], out_dir)
    assert removed == 1
    assert review_store.get_record(result.report_id, out_dir) is None
    assert review_store.get_record(result.deep_pdf_review_id, out_dir) is None


def test_api_payload_exposes_deep_pdf_and_separate_review(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls)
    result = _run(out_dir, DeepOptions(enabled=False, collect_external=False))
    payload = report_router._payload(result)

    assert payload["deep"]["pdf"]["filename"] == result.deep_artifacts["pdf"]
    assert payload["deep"]["pdf"]["pages"] == 3
    assert payload["deep"]["downloads"]["pdf"].endswith(".deep.pdf")
    assert payload["deep"]["pdf_merged"] is False
    assert payload["deep"]["pdf_mode"] == "separate"
    assert payload["deep"]["pdf_error"] == ""
    assert payload["deep"]["pdf_merge_error"] == ""
    assert payload["deep"]["review"]["report_id"] == result.deep_pdf_review_id
    assert payload["deep"]["review"]["finalized"] is False
    assert "待人工审核" in payload["deep"]["review"]["gate_reason"]
    assert payload["review"]["finalized"] is True
    # 文件名能过下载端点的路径收敛 + 后缀白名单
    monkeypatch.setattr(settings, "report_output_dir", out_dir)
    assert report_router.artifact_path(result.deep_artifacts["pdf"]) is not None


def test_deep_off_never_renders_second_pdf(out_dir, monkeypatch):
    calls: list = []
    _mock_pipeline(monkeypatch, calls)
    result = _run(out_dir, None)
    assert [c[0] for c in calls] == ["fact"]
    assert result.deep_pdf is None
    assert "deep" not in report_router._payload(result)


def test_api_request_default_is_full_separate():
    request = ReportRequest(request=_REQUEST)
    assert request.deep == "full"
    assert request.deep_pdf == "separate"


def test_deep_report_title_from_industry_draft():
    result = SimpleNamespace(
        report_id="981c41ca5a",
        document=SimpleNamespace(
            title="机器人行业 2026Q3 财报对比底稿（赛道分组版）",
        ),
        deep_analysis=DeepAnalysisResult(
            target="机器人（样本：鸣志电器）",
            period="2026Q3",
            mode="industry",
        ),
    )
    assert pipeline.deep_report_title(result) == "机器人行业 2026Q3 财报深度分析报告"
    assert "样本" not in pipeline._deep_stem(result)
    assert pipeline._deep_stem(result).startswith(
        "机器人行业 2026Q3 财报深度分析报告_981c41ca5a"
    )


def test_deep_report_title_appends_report_for_company():
    result = SimpleNamespace(
        report_id="aaaaaaaaaa",
        document=SimpleNamespace(title=f"{_SUBJECT} 2026Q3 财报深度分析"),
        deep_analysis=None,
    )
    assert pipeline.deep_report_title(result) == f"{_SUBJECT} 2026Q3 财报深度分析报告"


# ---------------------------------------------------------------------------
# 在线预览通道 / 下载门禁 / 推断结论红框（2026-11-10）
# 零 LLM、零 WeasyPrint：只打桩文件定位与审核门禁，不跑流水线。
# ---------------------------------------------------------------------------

_PREVIEW_USER = SimpleNamespace(user_id="u-1", username="tester")


@pytest.fixture
def preview_env(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        names = ("研报_981c41ca5a.deep.pdf", "研报_981c41ca5a.pdf", "legacy_abcdef1234.pdf")
        for name in names:
            (out / name).write_bytes(b"%PDF-1.4 stub")

        def artifact(name):
            candidate = out / name
            return candidate if candidate.is_file() else None

        def gate(name, output_dir=None):
            if name.endswith(".deep.pdf"):
                return False, "独立深度分析 PDF 需人工批准并定稿后才可下载"
            return True, "审核通过"

        def record(name, output_dir=None):
            if name.startswith("legacy_"):
                return None
            return SimpleNamespace(status="pending")

        monkeypatch.setattr(report_router, "artifact_path", artifact)
        monkeypatch.setattr(report_router.review_store, "pdf_download_gate", gate)
        monkeypatch.setattr(report_router.review_store, "record_for_artifact", record)
        yield out


def test_preview_allows_unfinalized_deep_pdf_but_download_still_blocked(preview_env):
    response = report_router.download_report("研报_981c41ca5a.deep.pdf", _PREVIEW_USER, 1)
    assert response.status_code == 200
    assert response.headers["X-Report-Download-Allowed"] == "0"
    assert response.headers["Content-Disposition"] == "inline"
    # 默认下载分支逐字不变：同一份文件仍然 409
    with pytest.raises(HTTPException) as exc:
        report_router.download_report("研报_981c41ca5a.deep.pdf", _PREVIEW_USER)
    assert exc.value.status_code == 409


def test_preview_marks_draft_pdf_as_downloadable(preview_env):
    response = report_router.download_report("研报_981c41ca5a.pdf", _PREVIEW_USER, 1)
    assert response.headers["X-Report-Download-Allowed"] == "1"
    assert report_router.download_report("研报_981c41ca5a.pdf", _PREVIEW_USER).status_code == 200


def test_preview_requires_review_record_and_valid_flag(preview_env):
    with pytest.raises(HTTPException) as exc:
        report_router.download_report("legacy_abcdef1234.pdf", _PREVIEW_USER, 1)
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException) as exc:
        report_router.download_report("研报_981c41ca5a.pdf", _PREVIEW_USER, 2)
    assert exc.value.status_code == 422


def test_review_red_boxes_live_only_in_deep_pdf():
    source = "<html><head></head><body><p>事实层</p></body></html>"
    font = Path("/tmp/fake-report-font.ttf")
    plain = apply_print_template(source, title="研报", font=font)
    deep = apply_print_template(source, title="研报", font=font, deep_css=True)
    assert "#d92d20" not in plain
    assert "#d92d20" in deep
    assert ".deep-review-box" in deep
    assert ".deep-claim-inferred td" in deep


def test_appendix_circles_inferred_claims_and_chapter_verdict():
    text = render_deep_appendix_html(_deep_result(), standalone=True)
    assert "deep-review-legend" in text
    assert "需人工审核确认后方可定稿外发" in text
    assert 'class="deep-verdict deep-review-box"' in text
    assert 'class="deep-claim-inferred"' in text
    assert '<table class="deep-claims">' in text


def test_cors_exposes_preview_gate_headers():
    """跨域下自定义响应头默认对 JS 不可见，前端会误判成可下载。"""
    from ai_agent.api.app import create_app

    app = create_app()
    cors = [m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"]
    assert cors
    exposed = cors[0].kwargs.get("expose_headers", [])
    assert "X-Report-Download-Allowed" in exposed
    assert "X-Report-Review-Status" in exposed
