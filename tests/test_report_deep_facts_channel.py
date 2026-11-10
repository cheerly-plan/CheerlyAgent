"""结构化事实清单数据通道：分类、覆盖度盘点、严格校验、入口挂载与零幻觉边界。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ai_agent.report.deep import (
    DeepConfig,
    SourceLedger,
    collect_sources,
    default_facts_dir,
    inspect_fact_row,
    lint_facts_dir,
    run_deep_analysis,
)
from ai_agent.report.deep.analyzer import numbers_allowed
from ai_agent.report.deep.sources import (
    LocalFactsProvider,
    classify_doc,
)


def _dir():
    return tempfile.TemporaryDirectory()


def _row(**overrides):
    row = {
        "item": "营业收入",
        "value": "1,234,567.89元",
        "source_file": "甲公司2025年第三季度报告（00000000-0000-0000-0000-000000000000）",
        "doc_type": "periodic",
        "level": "L1",
        "location": "第1页 一、主要财务数据",
        "quote": "单位：元 营业收入 1,234,567.89",
        "subject": "甲公司",
        "period": "2025Q3",
        "acquisition": "人工核对季报原文录入",
        "chapters": [3, 4],
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------- 文档分类


def test_semiannual_report_is_periodic_not_annual():
    """「半年度报告」含「年度报告」子串，旧顺序会把它误登记成年报。"""
    assert classify_doc("机器人2026年半年度报告") == "periodic"
    assert classify_doc("甲公司2026年中期报告") == "periodic"
    assert classify_doc("甲公司2024年年度报告.pdf") == "annual"
    assert classify_doc("甲公司2025年审计报告.pdf") == "audit"
    assert classify_doc("关于2025年审计报告问询函的回复.pdf") == "inquiry"


# ---------------------------------------------------------------- 覆盖度盘点


def test_coverage_report_separates_files_from_extracted_facts():
    """库里有 PDF 不等于数字已抽出：两列必须分开计数。"""
    rows = [{
        "file_id": "f1", "filename": "甲公司2026年半年度报告",
        "subject": "甲公司", "period": "2026H1", "status": "ready",
    }]
    with _dir() as directory:
        path = Path(directory) / "facts.json"
        path.write_text(json.dumps({"facts": [_row()]}, ensure_ascii=False), encoding="utf-8")
        bundle = collect_sources(
            "甲公司", "2026Q3", years=5,
            list_files=lambda: rows, facts_dir=Path(directory),
        )
    lines = "\n".join(bundle.coverage_report("company"))
    assert "定期报告（季报/半年报）" in lines
    assert "源文件 1 份 / 结构化事实 1 条" in lines
    assert "年度报告" in lines and "【缺】" in lines
    assert "源文件 0 份 / 结构化事实 0 条" in lines
    # 没有清单时同一批源文件应标「有源文件但未抽出数字」
    empty = collect_sources("甲公司", "2026Q3", years=5, list_files=lambda: rows)
    assert any("有源文件但未抽出数字" in line for line in empty.coverage_report("company"))
    assert any("无结构化事实" in gap.item for gap in empty.extraction_gaps())
    assert empty.unextracted_documents() == ["甲公司2026年半年度报告"]


def test_extraction_gaps_never_invent_facts_from_source_files():
    rows = [
        {"file_id": "f1", "filename": "甲公司2026年第一季度报告",
         "subject": "甲公司", "period": "2026Q1", "status": "ready"},
        {"file_id": "f2", "filename": "甲公司2025年年度报告",
         "subject": "甲公司", "period": "2025Y", "status": "ready"},
    ]
    bundle = collect_sources("甲公司", "2026Q3", years=5, list_files=lambda: rows)
    assert bundle.facts == []  # 只有源文件盘点，绝不顺手抽数
    assert len(bundle.documents) == 2
    items = [gap.item for gap in bundle.gaps]
    assert any("未抽出结构化事实" in item for item in items)
    assert any("但无结构化事实" in item for item in items)


# ---------------------------------------------------------------- 清单校验


def test_inspect_fact_row_runtime_keeps_ledger_as_only_gate():
    fact, errors, warnings = inspect_fact_row(_row())
    assert fact is not None and not errors and not warnings
    # 运行时只有三项必填会阻断，其余降级为警告，仍交给台账裁决
    fact, errors, warnings = inspect_fact_row(
        _row(level="L4", location="", quote="", backtrace_to="")
    )
    assert fact is not None and not errors
    assert any("缺 backtrace_to" in w for w in warnings)
    assert any("缺 location" in w for w in warnings)
    assert any("缺 quote" in w for w in warnings)
    assert fact.level == "L4"


def test_inspect_fact_row_strict_rejects_untraceable_rows():
    fact, errors, _ = inspect_fact_row(
        _row(level="L4", location="", quote="", backtrace_to=""), strict=True
    )
    assert fact is None
    assert any("backtrace_to" in e for e in errors)
    assert any("location" in e for e in errors)
    assert any("quote" in e for e in errors)
    _, errors, _ = inspect_fact_row({"item": "x"}, strict=True)
    assert any("缺 value" in e for e in errors) and any("缺 source_file" in e for e in errors)


def test_inspect_fact_row_checks_known_files_in_strict_mode():
    _, errors, _ = inspect_fact_row(
        _row(source_file="不存在的报告.pdf"), strict=True, known_files=["甲公司2025年第三季度报告"]
    )
    assert any("不在知识库" in e for e in errors)


def test_lint_facts_dir_reports_errors_duplicates_and_missing_dir():
    with _dir() as directory:
        root = Path(directory)
        assert not lint_facts_dir(root).ok
        (root / "facts.json").write_text(
            json.dumps({"facts": [
                _row(),
                _row(),  # 完全重复
                _row(item="毛利率", value="48.35%", level="L9",
                     location="", quote="", backtrace_to=""),
            ]}, ensure_ascii=False),
            encoding="utf-8",
        )
        (root / "broken.json").write_text("{not json", encoding="utf-8")
        strict = lint_facts_dir(root, strict=True)
        assert not strict.ok
        # 第三条在严格档被拦下，只有前两条（含重复那条）算可入账候选
        assert strict.accepted == 2
        assert any("完全重复" in d for d in strict.duplicates)
        assert any("解析失败" in e for e in strict.errors)
        assert any("level=L9" in e for e in strict.errors)
        loose = lint_facts_dir(root, strict=False)
        assert not any("level=L9" in e for e in loose.errors)  # 非法等级降级为警告
        assert any("解析失败" in e for e in loose.errors)  # 语法错误仍阻断


def test_local_facts_provider_exposes_rejections_and_warnings():
    with _dir() as directory:
        root = Path(directory)
        (root / "facts.json").write_text(
            json.dumps({"facts": [
                _row(),
                _row(),  # 重复，应被去重
                {"item": "缺出处", "value": "9"},
                _row(item="未经核对的条目", level="L9", location="", quote=""),
            ]}, ensure_ascii=False),
            encoding="utf-8",
        )
        (root / "bad.jsonl").write_text("{oops\n", encoding="utf-8")
        bundle = LocalFactsProvider(root).collect("甲公司", "2026Q3", years=5)
    assert len(bundle.facts) == 2  # 去重后只留两条不同事实
    notes = "\n".join(bundle.notes)
    assert "1 条缺 item/value/source_file，已丢弃" in notes
    assert "拒绝明细" in notes and "缺 source_file" in notes
    assert "可溯源警告" in notes
    assert any("解析失败" in gap.item for gap in bundle.gaps)
    assert any("未通过结构校验" in gap.item for gap in bundle.gaps)


def test_external_facts_reach_ledger_whitelist_verbatim():
    """清单数值逐字进白名单；模型心算值与未登记数字仍被拦。"""
    ledger = SourceLedger()
    assert ledger.add_external(**{
        k: v for k, v in _row().items() if k in {
            "item", "value", "source_file", "location", "level", "quote",
            "subject", "period", "acquisition", "chapters",
        }
    }) is True
    allowed = ledger.allowed_numbers
    assert numbers_allowed("营业收入 1,234,567.89元", allowed)
    assert not numbers_allowed("同比增长 34.31% 或 9,999,999.00元", allowed)


# ---------------------------------------------------------------- 入口与执行报告


def test_default_facts_dir_is_next_to_report_output_and_overridable(monkeypatch):
    from ai_agent.core.config import settings

    base = default_facts_dir()
    assert base.name == "deep_facts"
    assert base.parent == Path(settings.report_output_dir).parent
    monkeypatch.setenv("REPORT_DEEP_FACTS_DIR", "/tmp/custom-facts")
    assert default_facts_dir() == Path("/tmp/custom-facts")


def test_http_entry_mounts_default_facts_dir(monkeypatch):
    from ai_agent.api.routers import report as router
    from ai_agent.api.schemas import ReportRequest

    captured = {}

    def fake_run_report(request, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(router, "run_report", fake_run_report)
    router._run_request(ReportRequest(request="甲公司 2026Q3 财报分析"))
    assert captured["deep"].facts_dir == default_facts_dir()
    assert captured["deep"].chapter_concurrency >= 1


def test_run_deep_analysis_reports_external_admission_result():
    """候选 ≠ 入表：L4 未回溯被台账拒绝必须在执行报告里可见。"""
    sources = [
        {k: v for k, v in _row().items() if k != "doc_type"},
        {k: v for k, v in _row(
            item="市占率", value="18%", level="L4", backtrace_to="",
            source_file="某券商研报.pdf",
        ).items() if k != "doc_type"},
    ]
    result = run_deep_analysis(
        "甲公司", "2026Q3", [], config=DeepConfig(), external_sources=sources,
    )
    notes = "\n".join(result.execution.notes)
    assert "外部事实准入：2 条候选 / 1 条入表 / 1 条被台账拒绝" in notes
    assert "拒绝明细" in notes
    assert any("回溯" in gap.impact for gap in result.ledger_gaps)


# ---------------------------------------------------------------- 确定性导出


def test_external_fact_appears_in_chapter_prompt():
    """数据通道最后一公里：清单数值必须真的出现在章节 LLM 的授权事实台账里。"""
    from ai_agent.report.deep.analyzer import analyze_chapter
    from ai_agent.report.deep.chapters import spec_of

    ledger = SourceLedger()
    row = _row()
    assert ledger.add_external(**{
        k: v for k, v in row.items() if k in {
            "item", "value", "source_file", "location", "level", "quote",
            "subject", "period", "acquisition", "chapters",
        }
    }) is True
    captured: list[str] = []

    def stub(_system, user):
        captured.append(user)
        return json.dumps({
            "verdict": "仅作校验", "claims": [], "gaps": [],
            "tracking": [], "falsifiers": [],
        }, ensure_ascii=False)

    analyze_chapter(
        spec_of(4), ledger, target="甲公司", period="2026Q3",
        mode="company", prior="（无前序章节）", complete=stub,
    )
    assert captured
    assert "1,234,567.89元" in captured[0]
    assert row["source_file"] in captured[0]
    assert "第1页" in captured[0]


def test_render_for_keeps_external_facts_when_ledger_exceeds_limit():
    """清单变大后外部事实不能被事实层挤出证据包；未超限时行为逐字不变。"""
    from ai_agent.report.deep.ledger import SourceLedger, SourceRecord

    def record(index, *, external=False, chapters=()):
        return SourceRecord(
            source_id=f"e{index}", item=f"指标{index}", value=f"{index}元", unit="",
            subject="甲公司", period="2025Q3", filename=f"源文件{index}.pdf",
            file_id="", page=1, section="附注", quote=f"原文 {index}",
            external=external, chapters=chapters,
        )

    small = SourceLedger(records=[record(i) for i in range(3)])
    text = small.render_for([4])
    assert text.index("e0") < text.index("e1") < text.index("e2")
    assert "未展开" not in text

    many = SourceLedger(
        records=[record(i) for i in range(45)]
        + [record(100, external=True, chapters=(4,))]
    )
    text = many.render_for([4], limit=10)
    assert "e100" in text  # 声明服务第 4 章的外部事实必须可见
    assert "e0｜" in text  # 事实层也不能被完全挤掉
    assert "另有 36 条未展开" in text


def test_export_script_only_takes_original_amounts_with_provenance():
    from scripts.export_deep_facts import parse_provenance

    with _dir() as directory:
        path = Path(directory) / "x.provenance.md"
        path.write_text(
            "# 溯源附件\n\n"
            "| 公司 | 报告期 | 指标 | 数值 | 口径 | 文件 | 页码 | 章节 | 原文 |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| 甲公司 | 2026Q3 | 营业收入 | 1,234.56元 | consolidated | "
            "甲公司2026年第三季度报告（00000000-0000-0000-0000-000000000000） | 1 | "
            "一、主要财务数据 | 营业收入 \\| 1,234.56 |\n"
            "| 甲公司 | 2026Q3 | 毛利率 | 48.35% | consolidated | "
            "akshare.stock_financial_abstract（11111111-1111-1111-1111-111111111111） |  | "
            "财务摘要 | 毛利率 48.35 |\n"
            "| 甲公司 | 2026Q3 | 推算值 | 999.99元 | consolidated | 无出处文件 | 1 | 章节 | 原文 |\n",
            encoding="utf-8",
        )
        rows, skipped = parse_provenance(path)
    assert len(rows) == 1
    row = rows[0]
    assert row["value"] == "1,234.56元"  # 逐字搬运，不换算
    assert row["doc_type"] == "periodic"
    assert row["level"] == "L1"
    assert row["location"] == "第1页 一、主要财务数据"
    assert row["quote"] == "营业收入 | 1,234.56"
    assert row["chapters"] == [3, 4]
    assert skipped.get("第三方摘要") == 1
    assert skipped.get("无知识库 file_id") == 1
    fact, errors, warnings = inspect_fact_row(row, strict=True)
    assert fact is not None and not errors and not warnings
