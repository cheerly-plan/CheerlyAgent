"""主流水线挂接：默认零变化、产物物理隔离、外部数据源注入、失败隔离。

沿用 test_report_pipeline.py 的打桩方式，不用 pytest tmp_path（沙箱里
basetemp 可能已存在而报 EEXIST）。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import ai_agent.api.routers.report as report_router
import ai_agent.report.deep.bridge as bridge
import ai_agent.report.pipeline as pipeline
from ai_agent.api.schemas import ReportRequest
from ai_agent.report.cleanup import _ARTIFACT
from ai_agent.report.deep import DeepOptions
from ai_agent.report.deep.bridge import attach_deep, resolve_target_mode
from ai_agent.report.extract import Evidence, ExtractResult, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan, SourceFile

_SUBJECT = "中国能源建设股份有限公司"
_FILENAME = "中国能源建设2026年第三季度报告.pdf"
_REQUEST = "生成中国能源建设2026Q3财报深度分析研报"


@pytest.fixture
def out_dir():
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


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


def _fake_extract(subjects, period, indicators):
    values = {"2026Q3": "100,924,954", "2025Q3": "111,000,000"}
    if period in values:
        return ExtractResult(facts=[_fact(period, values[period])], missing=[])
    return ExtractResult(missing=[f"没有找到 {period} 的对应季报"])


def _mock_sources(monkeypatch):
    monkeypatch.setattr(pipeline, "parse_request", lambda _text: _plan())
    monkeypatch.setattr(pipeline, "extract_indicators", _fake_extract)


def test_run_report_without_deep_is_untouched(out_dir, monkeypatch):
    _mock_sources(monkeypatch)
    result = pipeline.run_report(_REQUEST, output_dir=out_dir, write_pdf=False)
    assert result.status == "ok"
    assert result.deep_analysis is None
    assert result.deep_error == ""
    assert result.deep_artifacts == {}
    assert list(out_dir.glob("*.deep.*")) == []
    assert set(result.artifacts.__dict__) == {"markdown", "provenance", "review", "pdf"}
    assert "100,924,954" in result.document.markdown


def test_skeleton_deep_writes_separate_artifacts(out_dir, monkeypatch):
    _mock_sources(monkeypatch)
    result = pipeline.run_report(
        _REQUEST, output_dir=out_dir, write_pdf=False,
        deep=DeepOptions(enabled=False, collect_external=False),
    )
    deep = result.deep_analysis
    assert deep is not None
    assert deep.execution.llm_calls == 0  # 零 token 骨架
    assert deep.target == _SUBJECT
    assert deep.mode == "company"
    assert set(result.deep_artifacts) == {"json", "markdown"}
    payload = json.loads((out_dir / result.deep_artifacts["json"]).read_text(encoding="utf-8"))
    assert payload["target"] == _SUBJECT
    assert "execution" in payload and "checklist" in payload
    markdown = (out_dir / result.deep_artifacts["markdown"]).read_text(encoding="utf-8")
    assert "本层是分析推断层" in markdown
    # 物理隔离：正文里既没有深度层声明，也没有多出任何产物键
    assert "本层是分析推断层" not in result.document.markdown
    assert set(result.artifacts.__dict__) == {"markdown", "provenance", "review", "pdf"}
    from ai_agent.report import review as review_store

    record = review_store.get_record(result.report_id, out_dir)
    assert "deep" not in record.artifacts  # M2 审核内容指纹不受影响


def test_external_sources_reach_deep_result(out_dir, monkeypatch):
    _mock_sources(monkeypatch)
    options = DeepOptions(
        enabled=False,
        list_files=lambda: [{
            "file_id": "f-audit",
            "filename": "中国能源建设2025年审计报告.pdf",
            "subject": _SUBJECT,
            "subject_aliases": [],
            "period": "2025Y",
            "status": "ready",
        }],
        stat_fetcher=lambda _t, _p, _y: [{
            "item": "行业总产能", "value": "120万片",
            "source_file": "协会2025产能统计.pdf", "period": "2025Y", "level": "L3",
        }],
    )
    result = pipeline.run_report(
        _REQUEST, output_dir=out_dir, write_pdf=False, deep=options
    )
    deep = result.deep_analysis
    assert any("inquiry" in gap.item for gap in deep.all_gaps)  # 缺问询函，具体到类型
    assert any("annual" in gap.item for gap in deep.all_gaps)  # 缺近 5 年年报
    assert any("外部数据源：1 条结构化事实候选" in note for note in deep.execution.notes)
    assert any("章节源文件覆盖" in note for note in deep.execution.notes)
    assert any("第6章" in note for note in deep.execution.notes)


def _stub_complete(system: str, user: str) -> str:
    """零真实 token 的补全桩：按 system 前缀返回符合 JSON 契约的响应。"""
    if system.startswith("你是章节评审专家组"):
        return json.dumps({"opinions": [{
            "expert": "财务专家", "issue": "现金流与利润匹配未核对",
            "disposition": "adopted", "reason": "", "revision": "补核对现金流量表",
        }]}, ensure_ascii=False)
    if system.startswith("你是核心链条收敛"):
        return json.dumps({
            "final_chain": "毛利率←产品结构←工艺壁垒（待实证）",
            "decisions": [{"mainline": "甲公司", "decision": "修正", "reason": "与第4章一致"}],
            "delta": "候选链条补上了对照组证据",
            "moat_rating": "中等", "moat_sources": {"成本优势": "未找到实证"},
            "falsifiers": [{"judgment": "成本优势", "signal": "单位成本连续两季高于同行",
                            "severity": "高"}],
            "tracking": [{"judgment": "成本优势", "indicator": "单位成本",
                          "direction": "高于同行即证伪", "source": "季报", "frequency": "季度"}],
            "scenarios": [{"name": "基准", "probability": 0.6, "assumption": "持平",
                           "trigger": "毛利率稳定", "impact": "维持"}],
        }, ensure_ascii=False)
    return json.dumps({
        "verdict": "评级：中（台账内事实可支撑）",
        "claims": [{"text": "营业收入为 100,924,954 元", "claim_type": "known",
                    "tier": "Tier 1", "evidence_ids": ["e0"], "confidence": 0.9}],
        "gaps": [{"item": "近5年审计报告", "impact": "可信度评级不完整"}],
        "tracking": [], "falsifiers": [],
    }, ensure_ascii=False)


def test_full_deep_with_stub_complete_runs_end_to_end(out_dir, monkeypatch):
    _mock_sources(monkeypatch)
    result = pipeline.run_report(
        _REQUEST, output_dir=out_dir, write_pdf=False,
        deep=DeepOptions(enabled=True, collect_external=False, complete=_stub_complete),
    )
    deep = result.deep_analysis
    assert result.status == "ok" and result.deep_error == ""
    assert deep.execution.llm_calls > 0  # 确实跑了章节 + 评审 + 收敛
    assert deep.convergence is not None
    assert any(chapter.status == "reviewed" for chapter in deep.chapters)
    assert (out_dir / result.deep_artifacts["json"]).is_file()
    # 主报告正文依旧是纯事实层，没有被深度层结论污染
    assert "毛利率←产品结构" not in result.document.markdown


def test_deep_failure_is_isolated_from_fact_layer(out_dir, monkeypatch):
    _mock_sources(monkeypatch)

    def boom(_result, _options):
        raise RuntimeError("stub boom")

    monkeypatch.setattr(bridge, "attach_deep", boom)
    result = pipeline.run_report(
        _REQUEST, output_dir=out_dir, write_pdf=False,
        deep=DeepOptions(enabled=False, collect_external=False),
    )
    assert result.status == "ok"
    assert "stub boom" in result.deep_error
    assert result.deep_analysis is None
    assert list(out_dir.glob("*.deep.*")) == []
    assert (out_dir / result.artifacts.markdown).is_file()


def test_api_payload_only_adds_deep_key_when_enabled(out_dir, monkeypatch):
    _mock_sources(monkeypatch)
    result = pipeline.run_report(_REQUEST, output_dir=out_dir, write_pdf=False)
    before = report_router._payload(result)
    assert "deep" not in before

    result.deep_analysis = attach_deep(
        result, DeepOptions(enabled=False, collect_external=False)
    )
    result.deep_artifacts = {"json": "x.deep.json", "markdown": "x.deep.md"}
    after = report_router._payload(result)
    assert after["deep"]["enabled"] is True
    assert after["deep"]["llm_calls"] == 0
    assert after["deep"]["downloads"]["json"] == "/v1/report/files/x.deep.json"
    assert isinstance(after["deep"]["gaps"], list)
    # 其余键一个不多、一个不少
    assert set(after) - set(before) == {"deep"}


def test_api_payload_reports_deep_error(out_dir, monkeypatch):
    _mock_sources(monkeypatch)
    result = pipeline.run_report(_REQUEST, output_dir=out_dir, write_pdf=False)
    result.deep_error = "ValueError: bad"
    payload = report_router._payload(result)
    assert payload["deep"]["enabled"] is False
    assert payload["deep"]["error"] == "ValueError: bad"


def test_report_request_deep_defaults_to_full():
    # 7caa047 起默认双 PDF + full 档；旧断言 "off" 与实现及双 PDF 测试冲突。
    assert ReportRequest(request=_REQUEST).deep == "full"
    assert ReportRequest(request=_REQUEST, deep="skeleton").deep == "skeleton"


def test_resolve_target_mode():
    plan = _plan()
    assert resolve_target_mode(plan, DeepOptions()) == (_SUBJECT, "company")
    plan.kind = "industry"
    plan.industry = "半导体"
    assert resolve_target_mode(plan, DeepOptions()) == ("半导体", "industry")
    assert resolve_target_mode(plan, DeepOptions(target="自定义", mode="company")) == (
        "自定义", "company",
    )


def test_cleanup_recognizes_deep_artifacts():
    for suffix in (".md", ".pdf", ".provenance.md", ".review.json", ".deep.md", ".deep.json"):
        assert _ARTIFACT.match(f"标题_abcdef1234{suffix}") is not None
