"""溯源台账：Tier 分级、引用链回溯、数字白名单、缺口显式声明。"""

from __future__ import annotations

from types import SimpleNamespace

from ai_agent.report.deep.ledger import SourceLedger, normalize_number, numbers_in


def fact(**overrides):
    data = {
        "subject": "甲公司",
        "period": "2026Q3",
        "name": "营业收入",
        "value": "1,234,567",
        "unit": "元",
    }
    data.update(overrides)
    evidence = SimpleNamespace(
        file_id="7b75379b-1111-2222-3333-444444444444",
        filename="甲公司2026Q3季报.pdf",
        page=8,
        page_end=8,
        section="合并利润表",
        quote="营业收入 1,234,567 元，同比增长 12%",
    )
    data["evidence"] = evidence
    return SimpleNamespace(**data)


def test_from_facts_builds_tier1_record_with_location():
    ledger = SourceLedger.from_facts([fact()])
    assert len(ledger.records) == 1
    record = ledger.records[0]
    assert record.source_id == "e0"
    assert record.tier == "Tier 1"
    assert "第8页" in record.location
    assert "合并利润表" in record.location
    assert "7b75379b" not in record.render()  # 短锚点不进本层证据包


def test_number_whitelist_normalizes_thousands_and_percent():
    assert normalize_number("1,234,567") == "1234567"
    assert normalize_number("12%") == "12"
    assert normalize_number("1,234,567") == normalize_number("1234567")
    ledger = SourceLedger.from_facts([fact()])
    assert "1234567" in ledger.allowed_numbers
    assert "12" in ledger.allowed_numbers


def test_empty_facts_declare_gap():
    ledger = SourceLedger.from_facts([])
    assert ledger.records == []
    assert ledger.gaps
    assert "缺口" in ledger.render_for([2]) or "台账为空" in ledger.render_for([2])


def test_l4_secondary_data_without_backtrace_is_rejected():
    ledger = SourceLedger.from_facts([fact()])
    ok = ledger.add_external(
        item="行业CR5", value="68%", source_file="某券商研报.pdf", level="L4",
        chapters=[6],
    )
    assert ok is False
    assert len(ledger.records) == 1
    assert ledger.rejected
    assert ledger.gaps[-1].chapters == [6]


def test_l4_with_backtrace_and_l3_original_are_accepted():
    ledger = SourceLedger.from_facts([fact()])
    assert ledger.add_external(
        item="行业CR5", value="68%", source_file="某券商研报.pdf", level="L4",
        backtrace_to="中国半导体行业协会2025年鉴 第42页", chapters=[6],
    )
    assert ledger.add_external(
        item="晶圆产能", value="120万片", source_file="统计局原始发布", level="L3",
    )
    assert [record.tier for record in ledger.records[1:]] == ["Tier 2", "Tier 1"]
    assert len(ledger.source_rows()) == 3


def test_source_rows_limited_by_ids():
    ledger = SourceLedger.from_facts([fact(), fact(name="扣非净利润", value="98,765")])
    rows = ledger.source_rows(["e1"])
    assert len(rows) == 1
    assert "扣非净利润" in rows[0].item


def test_numbers_in_handles_chinese_comma():
    assert "1234567" in numbers_in("1，234，567 元")
