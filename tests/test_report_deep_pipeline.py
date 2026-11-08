"""编排层：默认零 token、行业章节裁剪、预算闸、Gate 4/5 产物完整。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from ai_agent.report.deep import DeepConfig, chapters_for, run_deep_analysis


def facts(count: int = 2):
    out = []
    for index, (name, value) in enumerate(
        [("营业收入", "1,234,567"), ("扣非净利润", "98,765"), ("研发投入", "12,345")][:count]
    ):
        evidence = SimpleNamespace(
            file_id=f"7b75379b-{index:04d}", filename="甲公司2026Q3.pdf", page=8 + index,
            page_end=8 + index, section="合并利润表",
            quote=f"{name} {value} 元",
        )
        out.append(SimpleNamespace(
            subject="甲公司", period="2026Q3", name=name, value=value,
            unit="元", evidence=evidence,
        ))
    return out


def stub_complete(system: str, user: str) -> str:
    if system.startswith("你是章节评审专家组"):
        return json.dumps({"opinions": [{
            "expert": "财务专家", "issue": "现金流与利润匹配未核对",
            "disposition": "adopted", "reason": "", "revision": "补核对现金流量表",
        }]}, ensure_ascii=False)
    if system.startswith("你是核心链条收敛"):
        return json.dumps({
            "final_chain": "毛利率←产品结构←工艺壁垒（时间实证）",
            "decisions": [{"mainline": "甲公司", "decision": "确认", "reason": "与第4章一致"}],
            "delta": "候选与终稿一致，未发现推翻证据",
            "moat_rating": "中等",
            "moat_sources": {"品牌": "★★", "成本优势": "未找到实证"},
            "falsifiers": [{"judgment": "成本优势", "signal": "单位成本连续两季高于同行",
                            "severity": "高"}],
            "tracking": [{"judgment": "成本优势", "indicator": "单位成本",
                          "direction": "高于同行=证伪", "source": "季报", "frequency": "季度"}],
            "scenarios": [
                {"name": "基准", "probability": 0.55, "assumption": "持平",
                 "trigger": "毛利率稳定", "impact": "维持"},
                {"name": "乐观", "probability": 0.25, "assumption": "上行",
                 "trigger": "毛利率上行", "impact": "上调"},
                {"name": "悲观", "probability": 0.20, "assumption": "下行",
                 "trigger": "毛利率下滑", "impact": "下调"},
            ],
        }, ensure_ascii=False)
    return json.dumps({
        "verdict": "评级：中（台账内事实可支撑）",
        "claims": [
            {"text": "营业收入为 1,234,567 元", "claim_type": "known",
             "tier": "Tier 1", "evidence_ids": ["e0"], "confidence": 0.9},
            {"text": "产能受限可能支撑价格", "claim_type": "inferred",
             "tier": "Tier 2", "evidence_ids": ["e0"], "confidence": 0.6},
        ],
        "gaps": [{"item": "近5年审计报告", "impact": "可信度评级不完整"}],
        "tracking": [], "falsifiers": [],
    }, ensure_ascii=False)


def test_default_mode_is_zero_token_skeleton():
    result = run_deep_analysis("甲公司", "2026Q3", facts())
    assert result.execution.llm_calls == 0
    assert result.execution.estimated_input_tokens == 0
    assert all(chapter.status == "skipped" for chapter in result.chapters)
    assert result.mainlines  # 规则化候选主线不依赖 LLM
    assert result.checklist
    assert result.cross_checks
    json.loads(result.to_json())


def test_company_and_industry_chapter_sets():
    assert chapters_for("company") == tuple(range(1, 11))
    assert chapters_for("industry") == (1, 4, 5, 6, 8, 9)
    result = run_deep_analysis("机器人行业", "2026Q3", facts(), mode="industry")
    assert result.execution.enabled_chapters == [1, 4, 5, 6, 8, 9]
    assert {2, 3, 7, 10}.issubset(set(result.execution.skipped_chapters))
    assert any("产能/份额" in gap.item for gap in result.all_gaps)


def test_injected_complete_runs_full_loop_and_emits_report():
    result = run_deep_analysis(
        "甲公司", "2026Q3", facts(),
        config=DeepConfig(enabled=True, review_rounds=1),
        complete=stub_complete,
    )
    assert result.execution.llm_calls > 0
    assert result.execution.estimated_input_tokens > 0
    assert any(chapter.number == 2 and chapter.status == "reviewed" for chapter in result.chapters)
    assert any(chapter.review for chapter in result.chapters)
    assert result.convergence is not None
    assert result.convergence.delta
    assert result.falsifiers
    assert len(result.scenarios) == 3
    assert result.tracking
    text = result.to_markdown()
    for section in ("执行报告", "候选主线", "评审记录", "跟踪与证伪", "跨章一致性审查"):
        assert section in text
    json.loads(result.to_json())


def test_budget_gate_marks_remaining_chapters_skipped():
    result = run_deep_analysis(
        "甲公司", "2026Q3", facts(),
        config=DeepConfig(enabled=True, max_llm_calls=2, review_rounds=1),
        complete=stub_complete,
    )
    assert result.execution.llm_calls <= 2
    assert any("预算闸" in note for note in result.execution.notes)
    assert any(chapter.status == "skipped" for chapter in result.chapters)


def test_invalid_mode_rejected():
    try:
        run_deep_analysis("甲公司", "2026Q3", facts(), mode="macro")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 mode 应当被拒绝")


def test_unbacktraced_secondary_source_is_declared_as_ledger_gap():
    result = run_deep_analysis(
        "甲公司", "2026Q3", facts(),
        external_sources=[{
            "item": "行业CR5", "value": "68%", "source_file": "某券商研报.pdf",
            "level": "L4", "chapters": [6],
        }],
    )
    assert result.execution.llm_calls == 0
    assert result.chapters  # 骨架仍产出章节对象
    assert any("行业CR5" in gap.item for gap in result.ledger_gaps)
    assert any("行业CR5" in gap.item for gap in result.all_gaps)
