"""近 3 年年报自动补齐 + 抽数进 deep_facts：窗口、去重拉取、清单写入、失败兜底。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from ai_agent.report.deep.sources import LocalFactsProvider
from ai_agent.report.ensure_deep_filings import (
    annual_periods,
    ensure_deep_annuals,
    indicator_fact_to_deep_row,
)
from ai_agent.report.extract import Evidence, ExtractResult, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan


_SUBJECT = "甲公司"
_CODE = "688012"


def _plan(**kwargs) -> ReportPlan:
    base = dict(
        title=f"{_SUBJECT} 2026Q3 财报深度分析",
        kind="company",
        deliverable="markdown",
        subjects=[_SUBJECT],
        period="2026Q3",
        dimensions=["同比"],
        indicators=["营业收入", "毛利率"],
        outline=[OutlineItem("本期结论", "")],
        files=[],
    )
    base.update(kwargs)
    return ReportPlan(**base)


def _fact(period: str, value: str = "100") -> IndicatorFact:
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
            file_id="11111111-1111-1111-1111-111111111111",
            filename=f"{_SUBJECT}{period}年年度报告.pdf",
            page=12,
            page_end=12,
            section="主要会计数据",
            quote=f"营业收入 | {value}",
        ),
    )


def test_annual_periods_newest_first():
    assert annual_periods("2026Q3", 3) == ["2026", "2025", "2024"]
    assert annual_periods("2025", 2) == ["2025", "2024"]
    assert annual_periods("", 3) == []


def test_indicator_fact_to_deep_row_requires_page_and_quote():
    ok = indicator_fact_to_deep_row(_fact("2025"))
    assert ok is not None
    assert ok["doc_type"] == "annual"
    assert ok["value"] == "100元"
    assert "11111111-1111-1111-1111-111111111111" in ok["source_file"]

    bad = _fact("2025")
    bad.evidence.page = None
    assert indicator_fact_to_deep_row(bad) is None


def test_ensure_deep_annuals_writes_facts_and_skips_existing(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_fetch(code, period, **kwargs):
        calls.append((code, period))
        assert kwargs.get("max_full_bytes", 0) > 1_000_000  # 深度路径抬高全文阈值
        return SimpleNamespace(
            files=[
                SimpleNamespace(
                    file_id=f"id-{period}",
                    filename=f"{_SUBJECT}{period}年年度报告.pdf",
                    title=f"{_SUBJECT}{period}年年度报告",
                    status="ready",
                )
            ],
            errors=[],
            skipped=[],
        )

    def fake_files_for(subjects, period):
        # 2025 已入库，不应再拉
        if period == "2025":
            return [{
                "file_id": "id-2025",
                "filename": f"{_SUBJECT}2025年年度报告.pdf",
                "subject": _SUBJECT,
                "period": "2025",
                "status": "ready",
            }]
        return []

    def fake_extract(subjects, period, indicators):
        assert "营业收入" in indicators
        return ExtractResult(facts=[_fact(period, value=period[-2:])])

    monkeypatch.setattr(
        "ai_agent.report.ensure_deep_filings.cninfo.resolve_code_by_name",
        lambda name: SimpleNamespace(code=_CODE, name=_SUBJECT) if name == _SUBJECT else None,
    )
    monkeypatch.setattr(
        "ai_agent.report.ensure_deep_filings.cninfo.lookup_stock",
        lambda code: SimpleNamespace(code=code, name=_SUBJECT),
    )

    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        result = ensure_deep_annuals(
            _plan(),
            years=3,
            facts_dir=out,
            fetch_fn=fake_fetch,
            extract_fn=fake_extract,
            files_for_fn=fake_files_for,
            wait_fn=lambda ids: [],
            enable_fetch=True,
        )
        assert result.periods == ["2026", "2025", "2024"]
        assert {period for _, period in calls} == {"2026", "2024"}  # 2025 不重复拉
        assert result.fact_count >= 3
        assert result.facts_path is not None and result.facts_path.exists()
        payload = json.loads(result.facts_path.read_text(encoding="utf-8"))
        assert payload["meta"]["schema"] == "deep-facts/v1"
        assert payload["meta"]["fact_count"] == result.fact_count

        bundle = LocalFactsProvider(out).collect(_SUBJECT, "2026Q3", years=3)
        assert len(bundle.facts) == result.fact_count
        assert any(fact.item == "营业收入" for fact in bundle.facts)


def test_ensure_deep_annuals_fetch_failure_still_writes_empty_file(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.ensure_deep_filings.cninfo.resolve_code_by_name",
        lambda name: SimpleNamespace(code=_CODE, name=_SUBJECT),
    )
    monkeypatch.setattr(
        "ai_agent.report.ensure_deep_filings.cninfo.lookup_stock",
        lambda code: SimpleNamespace(code=code, name=_SUBJECT),
    )

    def boom(*_a, **_k):
        raise RuntimeError("network down")

    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        result = ensure_deep_annuals(
            _plan(),
            years=3,
            facts_dir=out,
            fetch_fn=boom,
            extract_fn=lambda *_a: ExtractResult(facts=[]),
            files_for_fn=lambda *_a: [],
            wait_fn=lambda ids: [],
            enable_fetch=True,
        )
        assert result.fact_count == 0
        assert result.facts_path is not None
        assert any("失败" in note for note in result.notes)
        assert result.facts_path.exists()


def test_attach_deep_runs_ensure_annuals_before_collect(monkeypatch, tmp_path):
    from ai_agent.report.deep.bridge import DeepOptions, attach_deep
    from ai_agent.report.deep.schema import DeepAnalysisResult, ExecutionReport

    seen: dict[str, object] = {}

    def fake_ensure(plan, **kwargs):
        seen["ensure"] = True
        seen["facts_dir"] = kwargs.get("facts_dir")
        return SimpleNamespace(notes=["年报补齐 ok"], fact_count=1, facts_path=None)

    def fake_collect(*_a, **kwargs):
        seen["collect"] = True
        assert seen.get("ensure") is True  # 必须先补齐再盘点
        from ai_agent.report.deep.sources import SourceBundle

        return SourceBundle(notes=["盘点 ok"])

    def fake_run(*_a, **_k):
        return DeepAnalysisResult(
            target=_SUBJECT,
            period="2026Q3",
            mode="company",
            execution=ExecutionReport(),
        )

    monkeypatch.setattr(
        "ai_agent.report.ensure_deep_filings.ensure_deep_annuals", fake_ensure
    )
    monkeypatch.setattr("ai_agent.report.deep.bridge.collect_sources", fake_collect)
    monkeypatch.setattr("ai_agent.report.deep.bridge.run_deep_analysis", fake_run)

    result = SimpleNamespace(
        request="生成甲公司2026Q3财报深度分析",
        plan=_plan(),
        facts=[],
    )
    deep = attach_deep(
        result,
        DeepOptions(
            enabled=False,
            facts_dir=tmp_path,
            ensure_annuals=True,
            collect_external=True,
        ),
    )
    assert seen.get("ensure") and seen.get("collect")
    assert any("年报补齐 ok" in note for note in deep.execution.notes)


def test_attach_deep_skips_ensure_when_disabled(monkeypatch, tmp_path):
    from ai_agent.report.deep.bridge import DeepOptions, attach_deep
    from ai_agent.report.deep.schema import DeepAnalysisResult, ExecutionReport

    called = {"ensure": False}

    def mark_ensure(*_a, **_k):
        called["ensure"] = True
        return SimpleNamespace(notes=[])

    monkeypatch.setattr(
        "ai_agent.report.ensure_deep_filings.ensure_deep_annuals",
        mark_ensure,
    )
    monkeypatch.setattr(
        "ai_agent.report.deep.bridge.collect_sources",
        lambda *_a, **_k: __import__(
            "ai_agent.report.deep.sources", fromlist=["SourceBundle"]
        ).SourceBundle(),
    )
    monkeypatch.setattr(
        "ai_agent.report.deep.bridge.run_deep_analysis",
        lambda *_a, **_k: DeepAnalysisResult(
            target=_SUBJECT, period="2026Q3", mode="company",
            execution=ExecutionReport(),
        ),
    )
    attach_deep(
        SimpleNamespace(request="", plan=_plan(), facts=[]),
        DeepOptions(
            enabled=False,
            facts_dir=tmp_path,
            ensure_annuals=False,
            collect_external=True,
        ),
    )
    assert called["ensure"] is False
