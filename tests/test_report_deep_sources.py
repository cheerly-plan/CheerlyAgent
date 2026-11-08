"""外部数据源层：文档分类、近 5 年窗口、缺口声明、事实注入与 L4 回溯门禁。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_agent.report.deep.ledger import SourceLedger
from ai_agent.report.deep.sources import (
    COMPANY_REQUIRED,
    ExternalFact,
    IndustryStatProvider,
    KnowledgeBaseProvider,
    LocalFactsProvider,
    classify_doc,
    collect_sources,
    window_years,
    year_of,
)


@pytest.fixture
def facts_dir():
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def _rows() -> list[dict]:
    return [
        {
            "file_id": f"f{i}",
            "filename": name,
            "subject": subject,
            "subject_aliases": [subject],
            "period": period,
            "status": "ready",
            "doc_type": "",
            "description": "",
        }
        for i, (name, period, subject) in enumerate(
            [
                ("甲公司2022年年度报告.pdf", "2022Y", "甲公司"),
                ("甲公司2024年年度报告.pdf", "2024Y", "甲公司"),
                ("甲公司2025年审计报告.pdf", "2025Y", "甲公司"),
                ("甲公司2026年第三季度报告.pdf", "2026Q3", "甲公司"),
                ("甲公司2020年年度报告.pdf", "2020Y", "甲公司"),
                ("乙公司2025年年度报告.pdf", "2025Y", "乙公司"),
            ]
        )
    ]


def test_classify_doc_prioritizes_inquiry_over_audit():
    assert classify_doc("关于2025年审计报告问询函的回复.pdf") == "inquiry"
    assert classify_doc("甲公司2025年审计报告.pdf") == "audit"
    assert classify_doc("甲公司2024年年度报告.pdf") == "annual"
    assert classify_doc("甲公司2026年第三季度报告.pdf") == "periodic"
    assert classify_doc("半导体行业协会产能统计2025.pdf") == "industry_stat"
    assert classify_doc("随手记的笔记.txt") == "other"


def test_year_and_window():
    assert year_of("2026Q3") == 2026
    assert year_of("甲公司2024年年度报告.pdf") == 2024
    assert year_of("无年份") == 0
    assert window_years("2026Q3", 5) == (2022, 2026, False)
    assert window_years("", 5)[2] is True


def test_kb_provider_filters_subject_period_and_status():
    provider = KnowledgeBaseProvider(_rows)
    bundle = provider.collect("甲公司", "2026Q3", years=5)
    names = [doc.filename for doc in bundle.documents]
    assert "甲公司2022年年度报告.pdf" in names
    assert "甲公司2026年第三季度报告.pdf" in names
    assert "甲公司2020年年度报告.pdf" not in names  # 超出 5 年窗口
    assert "乙公司2025年年度报告.pdf" not in names  # 主体不匹配
    assert len(bundle.facts) == 0  # 有 PDF ≠ 已抽出可溯源事实，不能假装入账
    assert any("源文件仅代表可回溯出处" in note for note in bundle.notes)


def test_kb_provider_skips_not_ready_rows():
    rows = _rows()
    rows[0]["status"] = "parsing"
    bundle = KnowledgeBaseProvider(lambda: rows).collect("甲公司", "2026Q3", years=5)
    assert all(doc.filename != "甲公司2022年年度报告.pdf" for doc in bundle.documents)


def test_kb_provider_declares_missing_required_types():
    # 只有季报 → 年报/审计/问询三类缺口都要具体到章节
    rows = [_rows()[3]]
    bundle = KnowledgeBaseProvider(lambda: rows).collect("甲公司", "2026Q3", years=5)
    items = [gap.item for gap in bundle.gaps]
    for kind in COMPANY_REQUIRED:
        assert any(kind in item for item in items)
    audit_gap = next(gap for gap in bundle.gaps if "audit" in gap.item)
    assert 2 in audit_gap.chapters
    assert "审计" in audit_gap.handling


def test_kb_provider_no_missing_gap_when_all_types_present():
    rows = _rows()[:3]
    rows.append({
        "file_id": "f9", "filename": "甲公司2025年问询函回复.pdf",
        "subject": "甲公司", "period": "2025Y", "status": "ready",
    })
    bundle = KnowledgeBaseProvider(lambda: rows).collect("甲公司", "2026Q3", years=5)
    assert bundle.missing_doc_types("company") == []
    assert bundle.requirement_gaps("company") == []


def test_kb_provider_rejects_non_callable_rows():
    with pytest.raises(TypeError, match="callable"):
        KnowledgeBaseProvider(_rows())  # 误传 list 应当立即报错，不能静默记缺口


def test_kb_provider_db_failure_is_a_gap_not_a_crash():
    def boom():
        raise RuntimeError("db down")

    bundle = KnowledgeBaseProvider(boom).collect("甲公司", "2026Q3", years=5)
    assert bundle.documents == []
    assert any("盘点" in gap.item for gap in bundle.gaps)
    assert any("读取失败" in note for note in bundle.notes)


def test_kb_provider_industry_mode_only_requires_statistics():
    bundle = KnowledgeBaseProvider(lambda: _rows()).collect("半导体", "2026Q3", years=5)
    assert bundle.missing_doc_types("industry") == ["industry_stat"]
    assert [gap.chapters for gap in bundle.requirement_gaps("industry")] == [[6, 9]]


def test_local_facts_provider_reads_json_and_jsonl(facts_dir):
    (facts_dir / "annual.json").write_text(
        json.dumps({"facts": [{
            "item": "分部收入-存储芯片", "value": "1,234,567",
            "source_file": "甲公司2024年年度报告.pdf", "location": "第88页",
            "quote": "存储芯片分部收入 1,234,567 元", "period": "2024Y",
            "subject": "甲公司", "doc_type": "annual", "level": "L1",
        }]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (facts_dir / "audit.jsonl").write_text(
        json.dumps({
            "item": "关键审计事项数量", "value": "3",
            "source_file": "甲公司2025年审计报告.pdf", "period": "2025Y",
            "subject": "甲公司", "doc_type": "audit", "level": "L1",
        }, ensure_ascii=False) + "\n# 注释行\n",
        encoding="utf-8",
    )
    bundle = LocalFactsProvider(facts_dir).collect("甲公司", "2026Q3", years=5)
    assert len(bundle.facts) == 2
    assert {fact.doc_type for fact in bundle.facts} == {"annual", "audit"}
    assert any("2 条入账候选" in note for note in bundle.notes)


def test_local_facts_provider_filters_year_and_drops_bad_rows(facts_dir):
    (facts_dir / "mixed.json").write_text(
        json.dumps([
            {"item": "旧数据", "value": "9", "source_file": "甲公司2019年年度报告.pdf",
             "period": "2019Y", "subject": "甲公司"},
            {"item": "缺源文件", "value": "9", "period": "2025Y"},
            {"item": "新数据", "value": "8", "source_file": "甲公司2025年年度报告.pdf",
             "period": "2025Y", "subject": "甲公司", "level": "L9"},
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    bundle = LocalFactsProvider(facts_dir).collect("甲公司", "2026Q3", years=5)
    assert [fact.item for fact in bundle.facts] == ["新数据"]
    assert bundle.facts[0].level == "L4"  # 未知等级按最保守的二手数据处理
    assert any("1 条缺 item/value/source_file" in note for note in bundle.notes)


def test_local_facts_provider_missing_directory_is_a_gap(facts_dir):
    bundle = LocalFactsProvider(facts_dir / "nope").collect("甲公司", "2026Q3", years=5)
    assert bundle.facts == []
    assert any("外部结构化事实清单" in gap.item for gap in bundle.gaps)


def test_external_fact_ledger_kwargs_round_trip():
    fact = ExternalFact(
        item="产能", value="120万片", source_file="统计局原始发布.pdf",
        doc_type="industry_stat", level="L3", location="第3页", chapters=(6, 9),
    )
    kwargs = fact.ledger_kwargs()
    assert kwargs["chapters"] == [6, 9]
    ledger = SourceLedger()
    assert ledger.add_external(**kwargs) is True
    assert ledger.records[0].tier == "Tier 1"


def test_l4_without_backtrace_is_rejected_by_ledger():
    fact = ExternalFact(
        item="市占率", value="18%", source_file="某券商研报.pdf",
        level="L4", chapters=(6,),
    )
    ledger = SourceLedger()
    assert ledger.add_external(**fact.ledger_kwargs()) is False
    assert ledger.records == []
    assert any("未回溯" in item for item in ledger.rejected)
    assert ledger.gaps[0].chapters == [6]


def test_l4_with_backtrace_is_accepted():
    fact = ExternalFact(
        item="市占率", value="18%", source_file="某券商研报.pdf",
        level="L4", backtrace_to="统计局原始发布.pdf", chapters=(6,),
    )
    ledger = SourceLedger()
    assert ledger.add_external(**fact.ledger_kwargs()) is True
    assert ledger.records[0].backtrace_to == "统计局原始发布.pdf"


def test_industry_stat_without_fetcher_declares_gap():
    bundle = IndustryStatProvider().collect("半导体", "2026Q3", years=5)
    assert bundle.facts == []
    assert bundle.gaps[0].chapters == [6, 9]
    assert any("不内置第三方 API" in note for note in bundle.notes)


def test_industry_stat_with_fetcher_and_failure():
    rows = [{
        "item": "行业总产能", "value": "120万片",
        "source_file": "协会2025产能统计.pdf", "period": "2025Y", "level": "L3",
    }]
    bundle = IndustryStatProvider(lambda _t, _p, _y: rows).collect("半导体", "2026Q3")
    assert len(bundle.facts) == 1
    assert bundle.documents[0].doc_type == "industry_stat"
    assert bundle.coverage()[6] == ["协会2025产能统计.pdf"]

    def boom(_t, _p, _y):
        raise TimeoutError("gateway")

    failed = IndustryStatProvider(boom).collect("半导体", "2026Q3")
    assert failed.facts == []
    assert any("拉取失败" in gap.item for gap in failed.gaps)


def test_collect_sources_merges_providers_and_dedups_required_gaps(facts_dir):
    bundle = collect_sources(
        "甲公司", "2026Q3", years=5, mode="company",
        facts_dir=facts_dir, list_files=_rows,
    )
    # 各 provider 各自报的「原文未入库」在组合层按合并结果重算：
    # 库里已有年报与审计报告，只缺问询函，且只留一份不重复。
    required = [gap for gap in bundle.gaps if gap.item.endswith("原文未入库")]
    assert len(required) == 1
    assert "inquiry" in required[0].item
    assert any("本地事实清单：0 个文件" in note for note in bundle.notes)
    assert any("章节源文件覆盖" in note for note in bundle.notes)
    # 年报与季报都在库里 → 第 3/4 章有源文件
    assert 3 in bundle.coverage()


def test_collect_sources_isolates_broken_provider():
    class Broken:
        def collect(self, target, period, *, years=5):
            raise ValueError("bad provider")

    bundle = collect_sources(
        "甲公司", "2026Q3", providers=[Broken(), IndustryStatProvider()],
        mode="industry",
    )
    assert any("Broken 采集失败" in note for note in bundle.notes)
    assert bundle.gaps  # 其余 provider 的缺口仍保留
