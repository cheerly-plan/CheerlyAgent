"""深度分析结构化结论对象：可序列化、可渲染、事实与推断分层。"""

from __future__ import annotations

import json

from ai_agent.report.deep.schema import (
    ChapterResult,
    Claim,
    Convergence,
    DataGap,
    DeepAnalysisResult,
    ExecutionReport,
    Falsifier,
    Mainline,
    Scenario,
    TrackingItem,
)


def _result() -> DeepAnalysisResult:
    chapter = ChapterResult(
        number=4,
        title="ROE 质量与来源",
        question="股东回报的质量和来源？",
        method="杜邦三分解",
        verdict="ROE 成色中等",
        claims=[Claim("净利率为 12%", "known", "Tier 1", ["e0"], 0.9)],
        gaps=[DataGap("近5年分部数据", [4], "无法拆解周转率")],
    )
    return DeepAnalysisResult(
        target="甲公司",
        period="2026Q3",
        mode="company",
        mainlines=[Mainline("存储芯片", "规则①收入主线", "毛利率←产品结构")],
        chapters=[chapter],
        convergence=Convergence(final_chain="链条终稿", delta="候选被修正"),
        tracking=[TrackingItem("净利率企稳", "毛利率", ">30%=验证")],
        falsifiers=[Falsifier("护城河", "毛利率连续两季下滑", "高")],
        scenarios=[Scenario("基准", 0.55, "需求持平", "毛利率稳定", "维持结论")],
        execution=ExecutionReport(target="甲公司", period="2026Q3", llm_calls=12),
    )


def test_result_is_json_serializable():
    result = _result()
    payload = json.loads(result.to_json())
    assert payload["target"] == "甲公司"
    assert payload["chapters"][0]["claims"][0]["evidence_ids"] == ["e0"]
    assert payload["convergence"]["delta"] == "候选被修正"
    assert payload["execution"]["llm_calls"] == 12


def test_markdown_has_required_sections_and_claim_labels():
    text = _result().to_markdown()
    for section in ("执行报告", "候选主线", "跨章一致性审查", "数据缺口总表", "跟踪与证伪"):
        assert section in text
    assert "【已知】" in text
    assert "分析推断层" in text
    assert "不构成估值或买卖建议" in text
    assert "毛利率连续两季下滑" in text  # 证伪信号表确实渲染出来


def test_gaps_and_sources_deduplicated():
    result = _result()
    result.chapters.append(
        ChapterResult(
            number=5, title="资产负债表叙事", question="钱从哪来", method="三流匹配",
            gaps=[DataGap("近5年分部数据", [4], "无法拆解周转率")],
        )
    )
    assert len(result.all_gaps) == 1


def test_table_escapes_pipe():
    from ai_agent.report.deep.schema import _table

    rows = _table(["项"], [["A|B"]])
    assert "A\\|B" in rows[2]
