"""分析器硬门禁：未授权数字、无证据推断、估值建议、空驳回一律丢弃。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from ai_agent.report.deep.analyzer import (
    _claims_from,
    analyze_chapter,
    converge,
    numbers_allowed,
    review_chapter,
)
from ai_agent.report.deep.chapters import spec_of
from ai_agent.report.deep.ledger import SourceLedger


def fact():
    evidence = SimpleNamespace(
        file_id="7b75379b-0000", filename="甲公司2026Q3.pdf", page=8,
        page_end=8, section="合并利润表", quote="营业收入 1,234,567 元",
    )
    return SimpleNamespace(
        subject="甲公司", period="2026Q3", name="营业收入",
        value="1,234,567", unit="元", evidence=evidence,
    )


def ledger() -> SourceLedger:
    return SourceLedger.from_facts([fact()])


def test_numbers_allowed_rejects_unauthorized_and_exempts_years():
    allowed = ledger().allowed_numbers
    assert numbers_allowed("营业收入为 1,234,567 元，2026Q3 第 4 章", allowed)
    assert not numbers_allowed("毛利率 42.5%", allowed)


def test_claims_gate_drops_unauthorized_numbers_inference_and_valuation():
    payload = {
        "claims": [
            {"text": "营业收入为 1,234,567 元", "claim_type": "known",
             "tier": "Tier 1", "evidence_ids": ["e0"], "confidence": 0.95},
            {"text": "毛利率达 42.5%", "claim_type": "known",
             "tier": "Tier 1", "evidence_ids": ["e0"]},
            {"text": "AI 订单驱动增长", "claim_type": "inferred",
             "tier": "Tier 2", "evidence_ids": [], "confidence": 0.9},
            {"text": "目标价 88 元，建议买入", "claim_type": "known",
             "tier": "Tier 1", "evidence_ids": ["e0"]},
            {"text": "产能受限支撑价格", "claim_type": "inferred",
             "tier": "Tier 2", "evidence_ids": ["e0"], "confidence": 0.95},
            {"text": "审计意见未披露", "claim_type": "uncertain",
             "tier": "Tier 5", "evidence_ids": [], "confidence": 0.9},
        ],
        "gaps": [{"item": "近5年审计报告", "impact": "可信度无法评级"}],
    }
    claims, gaps = _claims_from(payload, ledger(), 6, 2)
    texts = [claim.text for claim in claims]
    assert "营业收入为 1,234,567 元" in texts
    assert "毛利率达 42.5%" not in texts
    assert "AI 订单驱动增长" not in texts
    assert not any("目标价" in text for text in texts)
    inferred = next(claim for claim in claims if claim.claim_type == "inferred")
    assert inferred.confidence <= 0.7
    uncertain = next(claim for claim in claims if claim.claim_type == "uncertain")
    assert uncertain.confidence <= 0.4
    assert gaps[0].chapters == [2]


def test_analyze_chapter_parses_and_attaches_source_table():
    def complete(_system, _user):
        return json.dumps({
            "verdict": "可信度：中",
            "claims": [{"text": "营业收入为 1,234,567 元", "claim_type": "known",
                        "tier": "Tier 1", "evidence_ids": ["e0"], "confidence": 0.9}],
            "gaps": [], "tracking": [], "falsifiers": [],
        }, ensure_ascii=False)

    result = analyze_chapter(
        spec_of(2), ledger(), target="甲公司", period="2026Q3", mode="company",
        prior="", complete=complete,
    )
    assert result.status == "draft"
    assert result.verdict == "可信度：中"
    assert len(result.claims) == 1
    assert result.sources[0].tier == "Tier 1"


def test_analyze_chapter_degrades_on_llm_error():
    def boom(_system, _user):
        raise RuntimeError("网关 500")

    result = analyze_chapter(
        spec_of(3), ledger(), target="甲公司", period="2026Q3", mode="company",
        prior="", complete=boom,
    )
    assert result.status == "fallback"
    assert result.claims == []
    assert result.gaps


def test_review_requires_real_disposition_and_known_expert():
    from ai_agent.report.deep.schema import ChapterResult

    chapter = ChapterResult(number=4, title="ROE", question="?", method="杜邦",
                            verdict="中等", status="draft")
    def complete(_system, _user):
        return json.dumps({"opinions": [
            {"expert": "财务专家", "issue": "未分析账龄", "disposition": "adopted",
             "reason": "", "revision": "补充账龄"},
            {"expert": "批评家", "issue": "无异议", "disposition": "rejected",
             "reason": "我认为没问题", "revision": ""},
            {"expert": "路人甲", "issue": "越权意见", "disposition": "adopted"},
            {"expert": "哲学家", "issue": "前提未验证", "disposition": "rejected",
             "reason": "该数据另有年报 Tier 1 来源可回溯", "revision": ""},
        ]}, ensure_ascii=False)

    opinions = review_chapter(
        chapter, ledger(), experts=("财务专家", "批评家", "哲学家"),
        rounds=1, prior="", complete=complete,
    )
    # 每轮会跑 2 次（轮 1 + 终审合并轮），意见会重复出现，按专家集合断言
    experts = {item.expert for item in opinions}
    assert "财务专家" in experts
    assert "哲学家" in experts
    assert "批评家" not in experts
    assert "路人甲" not in experts


def test_converge_parses_chain_moat_falsifiers_and_scenarios():
    def complete(_system, _user):
        return json.dumps({
            "final_chain": "毛利率←产品结构←工艺时间壁垒",
            "decisions": [{"mainline": "甲公司", "decision": "修正", "reason": "事实部分不符"}],
            "delta": "候选高估了品牌作用",
            "moat_rating": "中等",
            "moat_sources": {"品牌": "★★", "成本优势": "未找到实证"},
            "falsifiers": [{"judgment": "成本优势", "signal": "单位成本连续两季高于同行",
                            "severity": "高"}],
            "tracking": [{"judgment": "成本优势", "indicator": "单位成本",
                          "direction": "高于同行=证伪", "source": "季报", "frequency": "季度"}],
            "scenarios": [
                {"name": "基准", "probability": 0.55, "assumption": "需求持平",
                 "trigger": "毛利率稳定", "impact": "维持"},
                {"name": "乐观", "probability": 0.25, "assumption": "需求上行",
                 "trigger": "毛利率上行", "impact": "上调"},
                {"name": "悲观", "probability": 0.20, "assumption": "需求下行",
                 "trigger": "毛利率下滑", "impact": "下调"},
            ],
        }, ensure_ascii=False)

    convergence, tracking, scenarios, falsifiers = converge(
        target="甲公司", period="2026Q3", mainlines_text="- 甲公司（规则①）",
        prior="第4章：ROE 中等", ledger=ledger(), complete=complete,
    )
    assert convergence is not None
    assert convergence.decisions[0]["decision"] == "修正"
    assert convergence.moat_sources["成本优势"] == "未找到实证"
    assert len(scenarios) == 3
    assert falsifiers[0]["severity"] == "高"
    assert tracking[0].indicator == "单位成本"
