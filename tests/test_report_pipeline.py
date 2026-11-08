"""研报编排层与端点：确定性串六步、降级、不编数、工具不进聊天循环。

不用 pytest 的 tmp_path（沙箱里 basetemp 目录可能已存在而报 EEXIST），
改用 tempfile.TemporaryDirectory 自带随机名。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import ai_agent.api.routers.report as report_router
import ai_agent.report.pipeline as pipeline
from ai_agent.api.schemas import ReportRequest
from ai_agent.core.config import settings
from ai_agent.storage.db import UserRecord
from ai_agent.report.compose import compose_report
from ai_agent.report.extract import Evidence, ExtractResult, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan, SourceFile
from ai_agent.tools.registry import TOOLS, TOOLS_MAP
from ai_agent.tools.report.report_generate import generate_report_tool

_USER = UserRecord(user_id="u-1", username="tester", display_name="tester")
_SUBJECT = "中国能源建设股份有限公司"
_FILENAME = "中国能源建设2026年第三季度报告.pdf"


@pytest.fixture
def out_dir():
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def _fact(
    period: str,
    value: str,
    *,
    column: str = "本报告期",
    reported_yoy: str = "",
) -> IndicatorFact:
    return IndicatorFact(
        subject=_SUBJECT,
        period=period,
        name="营业收入",
        value=value,
        unit="元",
        column=column,
        scope="consolidated",
        reported_yoy=reported_yoy,
        evidence=Evidence(
            file_id="file-1",
            filename=_FILENAME,
            page=12,
            page_end=12,
            section="主要会计数据",
            quote=f"营业收入 | {value}",
        ),
    )


def _plan(*, deliverable: str = "pdf") -> ReportPlan:
    return ReportPlan(
        title=f"{_SUBJECT} 2026Q3 财报深度分析",
        kind="company",
        deliverable=deliverable,
        subjects=[_SUBJECT],
        period="2026Q3",
        dimensions=["同比"],
        indicators=["营业收入", "毛利率"],
        outline=[
            OutlineItem("本期结论", ""),
            OutlineItem("主要指标与出处", ""),
            OutlineItem("同比与环比", ""),
            OutlineItem("趋势", ""),
            OutlineItem("缺失与口径", ""),
        ],
        files=[
            SourceFile("file-1", _FILENAME, _SUBJECT, "2026Q3", "本期"),
            SourceFile("file-0", "2025Q3.pdf", _SUBJECT, "2025Q3", "同比基期"),
        ],
    )


def _fake_extract(subjects, period, indicators):
    values = {"2026Q3": "100,924,954", "2025Q3": "111,000,000"}
    if period in values:
        return ExtractResult(facts=[_fact(period, values[period])], missing=["毛利率：表格里没有对上的一行"])
    return ExtractResult(missing=[f"没有找到 {period} 的对应季报"])


def _mock_sources(monkeypatch, plan: ReportPlan | None = None):
    monkeypatch.setattr(pipeline, "parse_request", lambda _text: plan or _plan())
    monkeypatch.setattr(pipeline, "extract_indicators", _fake_extract)


def test_pipeline_writes_pdf_and_all_artifacts(out_dir, monkeypatch):
    _mock_sources(monkeypatch)
    result = pipeline.run_report("生成中国能源建设2026Q3财报深度分析PDF研报", output_dir=out_dir)

    assert result.status == "ok"
    assert result.pdf_status == "generated"
    assert result.deliverable == "pdf"
    assert result.validation.passed is True
    assert len(result.facts) == 2
    assert "100,924,954" in result.document.markdown
    assert "增减 -10,075,046" in result.document.markdown

    assert (out_dir / result.artifacts.markdown).is_file()
    assert (out_dir / result.artifacts.provenance).is_file()
    assert (out_dir / result.artifacts.review).is_file()
    pdf_path = out_dir / result.artifacts.pdf
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert result.pdf is not None and result.pdf.pages >= 1
    assert "溯源附件" in (out_dir / result.artifacts.provenance).read_text(encoding="utf-8")
    json.loads((out_dir / result.artifacts.review).read_text(encoding="utf-8"))


def test_pipeline_degrades_to_markdown_when_render_fails(out_dir, monkeypatch):
    _mock_sources(monkeypatch)

    def boom(*_args, **_kwargs):
        raise RuntimeError("没有找到可嵌入的中文字体")

    monkeypatch.setattr(pipeline, "render_report_pdf", boom)
    result = pipeline.run_report("生成中国能源建设2026Q3财报深度分析PDF研报", output_dir=out_dir)

    assert result.pdf is None
    assert result.pdf_status == "failed"
    assert "中文字体" in result.pdf_error
    assert result.deliverable == "markdown"
    assert result.status == "ok"
    assert result.validation.passed is True
    assert result.artifacts.pdf == ""
    assert "100,924,954" in (out_dir / result.artifacts.markdown).read_text(encoding="utf-8")


def test_pipeline_no_data_never_invents_numbers(out_dir, monkeypatch):
    plan = _plan()
    plan.files = []
    plan.gaps = ["没有找到 2025Q3 的季报，同比基期无法调取"]
    _mock_sources(monkeypatch, plan)
    monkeypatch.setattr(
        pipeline,
        "extract_indicators",
        lambda *_a, **_k: ExtractResult(missing=["营业收入：表格里没有对上的一行"]),
    )
    result = pipeline.run_report("生成中国能源建设2026Q3财报深度分析研报", output_dir=out_dir, write_pdf=False)

    assert result.status == "no_data"
    assert result.pdf_status == "not_requested"
    assert "100,924,954" not in result.document.markdown
    assert "没有抽到指标" in result.document.markdown
    assert any("没有对上的一行" in gap for gap in result.gaps)
    assert (out_dir / result.artifacts.markdown).is_file()


def test_pipeline_uses_reported_yoy_and_still_passes_validation(out_dir, monkeypatch):
    """缺上一年季报时，正文写出季报原文披露的同比，校验仍须放行。"""
    plan = _plan()
    plan.files = [SourceFile("file-1", _FILENAME, _SUBJECT, "2026Q3", "本期")]
    plan.gaps = ["没有找到 2025Q3 的季报，同比基期的绝对值无法调取，无法跨期重算"]
    _mock_sources(monkeypatch, plan)
    fact = _fact("2026Q3", "526,365", column="年初至报告期末", reported_yoy="7.16%")
    monkeypatch.setattr(
        pipeline, "extract_indicators", lambda *_a, **_k: ExtractResult(facts=[fact])
    )
    result = pipeline.run_report(
        "生成中国能源建设2026Q3财报深度分析研报", output_dir=out_dir, write_pdf=False
    )

    assert result.validation.passed is True
    assert "同比 7.16%" in result.document.markdown
    assert "没有可写入的同比或环比" not in result.document.markdown
    assert "取数列：年初至报告期末" in result.document.markdown
    assert "年初至报告期末" in result.document.html


def test_pipeline_rejects_non_report_request(out_dir, monkeypatch):
    plan = ReportPlan(title="", kind="unknown", deliverable="markdown")
    plan.gaps.append("没有看到研报或财报分析需求")
    _mock_sources(monkeypatch, plan)
    result = pipeline.run_report("今天天气怎么样", output_dir=out_dir, write_pdf=False)

    assert result.status == "no_request"
    assert result.facts == []
    assert "没有大纲" in result.document.markdown


def test_only_the_safe_facade_is_exposed_to_the_chat_loop():
    """渲染工具带 output_path，绝不注册；积木工具无路径参数可注册。"""
    names = {tool.name for tool in TOOLS}
    assert "report_pdf" not in names
    # 抽数/算增减/校验三块积木不写盘、无路径参数，允许模型直接调用；
    # 它们的参数里同样不允许出现 output_path。
    for name in ("report_extract_indicators", "report_compare", "report_validate"):
        assert name in names
        props = TOOLS_MAP[name].schema()["function"]["parameters"]["properties"]
        assert "output_path" not in props
    assert "generate_report" in names
    tool = TOOLS_MAP["generate_report"]
    props = tool.schema()["function"]["parameters"]["properties"]
    assert set(props) == {"request", "report_type", "write_pdf"}
    assert "output_path" not in props


def test_generate_report_tool_returns_links_not_body_numbers(monkeypatch):
    """工具只回下载链接与摘要，不把正文数字塞回模型上下文诱导复述。"""
    fake = SimpleNamespace(
        document=SimpleNamespace(title="中国银行股份有限公司 2026Q3 财报深度分析"),
        status="ok",
        deliverable="pdf",
        pdf_status="generated",
        validation=SimpleNamespace(
            summary="校验通过，但仍有 1 条需人工审核。", findings=[], review_queue=[]
        ),
        artifacts=SimpleNamespace(
            pdf="bank.pdf",
            markdown="bank.md",
            provenance="bank.provenance.md",
            review="bank.review.json",
        ),
        deep_artifacts={
            "pdf": "bank.deep.pdf",
            "markdown": "bank.deep.md",
            "json": "bank.deep.json",
        },
        deep_error="",
        deep_pdf_error="",
        deep_pdf_review_id="abc1234567-deep",
        pdf_error="",
        gaps=["没有找到 2025Q3 的季报，同比基期的绝对值无法调取，无法跨期重算"],
    )
    captured: dict = {}

    def _fake_run(*a, **k):
        captured.update(k)
        return fake

    monkeypatch.setattr("ai_agent.tools.report.report_generate.run_report", _fake_run)
    text = generate_report_tool.run(request="生成中国银行2026Q3财报深度分析PDF研报")

    assert captured.get("deep") is not None
    assert captured["deep"].enabled is True
    assert captured["deep"].pdf_mode == "separate"
    assert "/v1/report/files/bank.pdf" in text
    assert "/v1/report/files/bank.deep.pdf" in text
    assert "底稿 PDF" in text
    assert "深度解析 PDF" in text
    assert "/v1/report/files/bank.provenance.md" in text
    assert "校验通过" in text
    assert "缺口（如实转述，不要补数）" in text
    assert "不要复述" in text
    # M2：fake 没有 review_status，按「未定稿」如实提示，不谎称 PDF 可直接下载
    assert "尚未定稿" in text
    assert "/v1/report/reviews/" in text
    assert "526,365" not in text  # 正文数字不进工具返回
    assert len(text) <= 3000  # execute_tool 的截断上限


def test_markdown_keeps_table_rows_contiguous():
    """表格行之间不能插空行，否则 Markdown 表格散架（PDF/HTML 路径不受影响）。"""
    document = compose_report(_plan(), [_fact("2026Q3", "100,924,954")])
    lines = document.markdown.splitlines()
    # 单公司报告没有「公司」列，主体由标题承担
    header = lines.index("| 指标 | 数值 | 期间口径／范围／取数列 | 出处 |")
    assert lines[header + 1].startswith("| ---")
    assert lines[header + 2].startswith("| 营业收入")


def test_artifact_path_blocks_traversal_and_unknown_suffix(out_dir, monkeypatch):
    monkeypatch.setattr(settings, "report_output_dir", out_dir)
    good = out_dir / "report.pdf"
    good.write_bytes(b"%PDF-1.4")

    # macOS 上 /var 是 /private/var 的符号链接，比较 resolve 后的路径
    assert report_router.artifact_path("report.pdf") == good.resolve()
    assert report_router.artifact_path("../config.py") is None
    assert report_router.artifact_path("sub/report.pdf") is None
    assert report_router.artifact_path("report.exe") is None
    assert report_router.artifact_path("missing.pdf") is None
    assert report_router.artifact_path("") is None


def test_create_report_returns_full_payload(out_dir, monkeypatch):
    _mock_sources(monkeypatch)
    real_result = pipeline.run_report("生成中国能源建设2026Q3财报深度分析研报", output_dir=out_dir, write_pdf=False)
    monkeypatch.setattr(report_router, "run_report", lambda *a, **k: real_result)

    payload = report_router.create_report(ReportRequest(request="x", write_pdf=False), _USER)

    assert payload["status"] == "ok"
    assert payload["deliverable"] == "markdown"
    assert payload["fact_count"] == 2
    assert payload["validation"]["passed"] is True
    assert "100,924,954" in payload["markdown"]
    assert "溯源附件" in payload["provenance"]
    assert payload["plan"]["period"] == "2026Q3"
    assert payload["downloads"]["markdown"].startswith("/v1/report/files/")
    assert payload["pdf"] is None
    assert payload["gaps"]


def test_download_rejects_missing_file(monkeypatch, out_dir):
    from fastapi import HTTPException

    monkeypatch.setattr(settings, "report_output_dir", out_dir)
    with pytest.raises(HTTPException) as exc:
        report_router.download_report("nope.pdf", _USER)
    assert exc.value.status_code == 404
