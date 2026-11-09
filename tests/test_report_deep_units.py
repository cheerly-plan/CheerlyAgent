"""台账金额的单位换算白名单：放行等价写法，绝不放行模型心算值。

真实端到端产物里，评审意见自己指出「作者未进行数据提取与单位换算，导致结论
出现 5 处数据空白」——台账原文是「元」，模型习惯写「亿元」，旧白名单把
16.14 亿元这类等价写法全遮蔽了。本文件钉死放行的边界：只有从 L1 台账数值
按固定倍率派生的金额写法可放行，同比/差值/编造值仍一律拦下。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from ai_agent.report.deep import DeepConfig, run_deep_analysis
from ai_agent.report.deep.analyzer import guard_numbers, numbers_allowed
from ai_agent.report.deep.ledger import (
    derived_number_forms,
    SourceLedger,
)


def fact(name: str, value: str, unit: str = "元", quote: str = ""):
    return SimpleNamespace(
        subject="甲公司", period="2026Q3", name=name, value=value, unit=unit,
        evidence=SimpleNamespace(
            file_id="f-1", filename="甲公司2026Q3.pdf", page=8,
            section="合并利润表", quote=quote or f"{name} {value}{unit}",
        ),
    )


def ledger_with_yuan() -> SourceLedger:
    return SourceLedger.from_facts([
        fact("营业收入", "1,614,059,366.89"),
        fact("扣非净利润", "-205,260,194.78"),
    ])


def test_yuan_value_allows_wan_and_yi_rewrites():
    allowed = ledger_with_yuan().allowed_numbers
    assert numbers_allowed("营业收入 16.14 亿元", allowed)
    assert numbers_allowed("营业收入 16.1406 亿元", allowed)
    assert numbers_allowed("营业收入 161,405.94 万元", allowed)
    assert numbers_allowed("营业收入 1,614,059,366.89 元", allowed)
    # 台账里没有的金额，换个单位也照样不许出现。
    assert not numbers_allowed("扣非净利润 1.17 亿元", allowed)
    assert not numbers_allowed("合同负债 9.99 亿元", allowed)
    # 计算派生（同比、差值）不是单位换算，继续拦。
    assert not numbers_allowed("同比 +227.60%", allowed)


def test_negative_amount_derives_negative_yi():
    allowed = ledger_with_yuan().allowed_numbers
    assert numbers_allowed("扣非净利润亏损 -2.05 亿元", allowed)
    assert not numbers_allowed("扣非净利润亏损 -2.99 亿元", allowed)


def test_negative_amount_allows_absolute_value_wording():
    """真实产物里评审指出：台账是 -205,260,194.78 元，模型写「亏损 2.05 亿元」。"""
    allowed = ledger_with_yuan().allowed_numbers
    assert numbers_allowed("扣非净利润亏损 2.05 亿元", allowed)
    assert numbers_allowed("经营活动现金流净流出 2.05 亿元", allowed)
    assert not numbers_allowed("扣非净利润亏损 2.99 亿元", allowed)


def test_percent_and_non_amount_units_are_not_derived():
    assert derived_number_forms("48.35", "%") == set()
    assert derived_number_forms("12", "人") == set()
    assert derived_number_forms("约 16 亿元", "元") == set()  # 非纯数字 value
    allowed = SourceLedger.from_facts([fact("毛利率", "48.35", "%")]).allowed_numbers
    assert numbers_allowed("毛利率 48.35%", allowed)
    assert not numbers_allowed("写成小数 0.4835", allowed)
    assert not numbers_allowed("换算成 4,835 万元", allowed)


def test_value_with_embedded_unit_derives_other_units():
    ledger = SourceLedger.from_facts([fact("营业收入", "16.14亿元", "")])
    assert numbers_allowed("营业收入 16.14 亿元", ledger.allowed_numbers)
    assert numbers_allowed("营业收入 1,614,000,000 元", ledger.allowed_numbers)
    assert not numbers_allowed("营业收入 16.15 亿元", ledger.allowed_numbers)


def test_unit_conversions_only_lists_amount_records():
    ledger = SourceLedger.from_facts([
        fact("营业收入", "1,614,059,366.89"),
        fact("毛利率", "48.35", "%"),
    ])
    rows = ledger.unit_conversions
    assert len(rows) == 1 and "营业收入" in rows[0]


def test_guard_numbers_keeps_derived_value_and_masks_invented_one():
    allowed = ledger_with_yuan().allowed_numbers
    guarded, blocked = guard_numbers(
        "营收 16.14 亿元；扣非净利润 1.17 亿元；同比 +227.60%", allowed
    )
    assert "16.14" in guarded
    assert "〔未溯源〕" in guarded
    assert "1.17" not in guarded and "227.60" not in guarded
    assert len(blocked) == 2


def _stub(verdict: str):
    def complete(system: str, _user: str) -> str:
        if system.startswith("你是章节评审专家组"):
            return json.dumps({"opinions": []}, ensure_ascii=False)
        if system.startswith("你是核心链条收敛"):
            return json.dumps({
                "final_chain": "营收增长由订单驱动", "decisions": [],
                "delta": "未推翻", "moat_rating": "中等", "moat_sources": {},
                "falsifiers": [], "tracking": [], "scenarios": [],
            }, ensure_ascii=False)
        return json.dumps({
            "verdict": verdict, "claims": [], "gaps": [],
            "tracking": [], "falsifiers": [],
        }, ensure_ascii=False)
    return complete


def test_deep_pipeline_keeps_unit_rewrites_but_masks_invented_numbers():
    result = run_deep_analysis(
        "甲公司", "2026Q3",
        [fact("营业收入", "1,614,059,366.89"),
         fact("扣非净利润", "-205,260,194.78")],
        config=DeepConfig(enabled=True, chapters=(4, 9), review_rounds=1),
        complete=_stub(
            "营收 16.14 亿元，扣非净利润亏损 2.05 亿元，但另有 1.17 亿元缺口"
        ),
    )
    verdicts = " ".join(chapter.verdict for chapter in result.chapters)
    assert "16.14 亿元" in verdicts
    assert "亏损 2.05 亿元" in verdicts
    assert "1.17" not in verdicts
    assert "〔未溯源〕" in verdicts
    assert any("单位换算白名单" in note for note in result.execution.notes)
