"""决策 ④：分章模型路由 + 第 9 章收敛质量门禁。

全部用例零真实 token：要么跑零调用骨架，要么注入 stub / 打桩 default_complete。
重点覆盖「默认行为不变」「模型不再静默漂移」「主模型只用于收敛」「判定对外可见」。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import ai_agent.report.deep.pipeline as deep_pipeline
from ai_agent.report.deep import (
    DeepConfig,
    assess_convergence,
    normalize_moat_rating,
    pick_industry_sample,
    render_deep_appendix_html,
    resolve_model_plan,
    run_deep_analysis,
)
from ai_agent.report.deep.ledger import SourceLedger
from ai_agent.report.deep.schema import (
    Convergence,
    DeepAnalysisResult,
    ExecutionReport,
    Falsifier,
    Mainline,
    Scenario,
    TrackingItem,
)


def facts():
    evidence = SimpleNamespace(
        file_id="7b75379b-0001", filename="甲公司2026Q3.pdf", page=8,
        page_end=8, section="合并利润表", quote="营业收入 1,234,567 元",
    )
    return [SimpleNamespace(
        subject="甲公司", period="2026Q3", name="营业收入", value="1,234,567",
        unit="元", evidence=evidence,
    )]


_REVIEW = json.dumps({"opinions": [{
    "expert": "财务专家", "issue": "现金流与利润匹配未核对",
    "disposition": "adopted", "reason": "", "revision": "补核对现金流量表",
}]}, ensure_ascii=False)

_CONVERGENCE = json.dumps({
    "final_chain": "毛利率←产品结构←工艺壁垒，因产能爬坡周期形成时间实证",
    "decisions": [
        {"mainline": "甲公司", "decision": "确认", "reason": "与第4章一致"},
        {"mainline": "备用主线", "decision": "推翻", "reason": "台账无支撑"},
        {"mainline": "第三条", "decision": "修正", "reason": "口径需调整"},
    ],
    "delta": "候选被修正一条、推翻一条",
    "moat_rating": "中等",
    "moat_sources": {
        "品牌": "★★", "成本优势": "★★★", "网络效应": "未找到实证",
        "转换成本": "★", "特许/牌照": "未找到实证",
    },
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

_CHAPTER = json.dumps({
    "verdict": "评级：中（台账内事实可支撑）",
    "claims": [{"text": "营业收入为 1,234,567 元", "claim_type": "known",
                "tier": "Tier 1", "evidence_ids": ["e0"], "confidence": 0.9}],
    "gaps": [], "tracking": [], "falsifiers": [],
}, ensure_ascii=False)


def stub(log: list, name: str):
    """记录「哪条通道、哪种 system prompt」被调用。"""
    def call(system: str, user: str) -> str:
        log.append((name, system.split("\n", 1)[0]))
        if system.startswith("你是章节评审专家组"):
            return _REVIEW
        if system.startswith("你是核心链条收敛"):
            return _CONVERGENCE
        return _CHAPTER
    return call


def solid_convergence() -> Convergence:
    return Convergence(
        final_chain="毛利率←产品结构←工艺壁垒，因产能爬坡周期形成时间实证",
        decisions=[{"mainline": "甲", "decision": "确认"}],
        moat_sources={
            "品牌": "★★", "成本优势": "★★★", "网络效应": "未找到实证",
            "转换成本": "★", "特许": "未找到实证",
        },
    )


def solid_parts():
    return (
        [Mainline("甲", "规则①", "链条")],
        [Scenario("基准", 0.55, "", "", ""), Scenario("乐观", 0.25, "", "", ""),
         Scenario("悲观", 0.20, "", "", "")],
        [Falsifier("成本优势", "单位成本高于同行")],
        [TrackingItem("成本优势", "单位成本", "高于同行=证伪")],
    )


# ——— 默认行为与路由解析（纯函数，零调用）———

def test_zero_token_skeleton_never_reads_model_config():
    result = run_deep_analysis("甲公司", "2026Q3", facts())
    assert result.execution.llm_calls == 0
    assert result.execution.model_route == "none"
    assert result.execution.chapter_model == ""
    assert result.execution.convergence_model == ""
    assert result.execution.convergence_quality == ""
    assert any("未产出收敛" in item for item in result.execution.convergence_checks)


def test_default_plan_small_chapters_primary_convergence():
    plan = resolve_model_plan(
        DeepConfig(), small_model="qwen-small", primary_model="gpt-main"
    )
    assert plan.route == "small+primary"
    assert plan.chapter_label == "qwen-small"
    assert plan.convergence_label == "gpt-main"
    assert plan.chapter_model == ""  # 交给 lightweight_llm("") 走本地网关
    assert plan.convergence_model == "gpt-main"  # 显式模型名 → 主网关
    assert plan.warnings == ()


def test_missing_small_model_warns_instead_of_silent_fallback():
    plan = resolve_model_plan(DeepConfig(), small_model="", primary_model="gpt-main")
    assert plan.route == "all-primary"
    assert any("SMALL_MODEL 未配置" in warning for warning in plan.warnings)


def test_explicit_convergence_model_can_return_to_all_small():
    plan = resolve_model_plan(
        DeepConfig(convergence_model="qwen-small"),
        small_model="qwen-small", primary_model="gpt-main",
    )
    assert plan.route == "all-small"


def test_explicit_chapter_model_marks_override():
    plan = resolve_model_plan(
        DeepConfig(model="special-x"),
        small_model="qwen-small", primary_model="gpt-main",
    )
    assert plan.route == "override"
    assert plan.chapter_model == "special-x"


def test_injected_stub_is_marked_as_injected():
    log: list = []
    result = run_deep_analysis(
        "甲公司", "2026Q3", facts(),
        config=DeepConfig(enabled=True, review_rounds=1),
        complete=stub(log, "small"),
    )
    assert result.execution.model_route == "injected"
    assert result.execution.chapter_model == "外部注入"
    assert result.execution.convergence_model == "外部注入"


# ——— 真实通道（default_complete 打桩，不触网）———

def test_convergence_and_its_final_review_use_primary_channel():
    log: list = []
    result = run_deep_analysis(
        "甲公司", "2026Q3", facts(),
        config=DeepConfig(enabled=True, review_rounds=1),
        complete=stub(log, "small"),
        convergence_complete=stub(log, "primary"),
    )
    assert result.convergence is not None
    convergence_calls = [row for row in log if row[1].startswith("你是核心链条收敛")]
    assert convergence_calls
    assert all(row[0] == "primary" for row in convergence_calls)
    # 第 9 章的终审也要用主模型：它负责判断有没有停在断言层
    assert any(
        row[0] == "primary" and row[1].startswith("你是章节评审") for row in log
    )
    # 章节分析与第 2-8 章评审仍走小模型通道
    assert any(row[0] == "small" and row[1].startswith("你是上市公司") for row in log)
    assert any(row[0] == "small" and row[1].startswith("你是章节评审") for row in log)
    assert all(
        row[0] == "small" for row in log if row[1].startswith("你是上市公司")
    )


def test_real_route_is_recorded_in_execution_report(monkeypatch):
    seen: list[str] = []
    log: list = []

    def fake_default(max_tokens, temperature, timeout, model):
        seen.append(model)
        return stub(log, "fake")

    monkeypatch.setattr(
        deep_pipeline, "settings",
        SimpleNamespace(model="gpt-main", small_model="qwen-small"),
    )
    monkeypatch.setattr(deep_pipeline, "default_complete", fake_default)
    result = run_deep_analysis(
        "甲公司", "2026Q3", facts(),
        config=DeepConfig(enabled=True, review_rounds=1),
    )
    assert "" in seen and "gpt-main" in seen  # 两条通道传了不同的业务模型名
    assert result.execution.model_route == "small+primary"
    assert result.execution.chapter_model == "qwen-small"
    assert result.execution.convergence_model == "gpt-main"
    assert any("模型路由：small+primary" in note for note in result.execution.notes)


def test_missing_small_model_warning_reaches_execution_notes(monkeypatch):
    monkeypatch.setattr(
        deep_pipeline, "settings", SimpleNamespace(model="gpt-main", small_model=""),
    )
    monkeypatch.setattr(
        deep_pipeline, "default_complete",
        lambda *args: stub([], "fake"),
    )
    result = run_deep_analysis(
        "甲公司", "2026Q3", facts(),
        config=DeepConfig(enabled=True, review_rounds=1),
    )
    assert result.execution.model_route == "all-primary"
    assert any("SMALL_MODEL 未配置" in note for note in result.execution.notes)


# ——— 收敛质量门禁 ———

def test_quality_solid_on_complete_convergence():
    quality, checks = assess_convergence(solid_convergence(), *solid_parts())
    assert quality == "solid"
    assert all(item.startswith("✅") for item in checks)


def test_quality_shallow_when_chain_stops_at_assertion_layer():
    convergence = Convergence(final_chain="品牌强、技术领先")
    quality, checks = assess_convergence(convergence, *solid_parts())
    assert quality == "shallow"
    assert any("断言层" in item for item in checks)


def test_quality_incomplete_when_required_pieces_are_missing():
    convergence = Convergence(
        final_chain="毛利率←产品结构", moat_sources={"品牌": "★★"}, decisions=[]
    )
    quality, checks = assess_convergence(convergence, solid_parts()[0], [], [], [])
    assert quality == "incomplete"
    assert any("护城河分项" in item for item in checks)
    assert any("概率合计" in item for item in checks)


def test_moat_rating_downgrades_when_evidence_thin():
    sources = {
        "品牌": "未找到实证",
        "成本优势": "★★☆",
        "网络效应": "无",
        "转换成本": "★★",
        "特许/牌照": "未找到实证",
    }
    assert normalize_moat_rating("中等", sources) == "暂无法评级"
    assert normalize_moat_rating("宽", {
        "品牌": "★★★★", "成本优势": "★★★", "网络效应": "★★",
        "转换成本": "★★★", "特许/牌照": "★★",
    }) == "宽"


def test_industry_sample_is_declared_in_target_and_notes():
    evidence_a = SimpleNamespace(
        file_id="a", filename="鸣志.pdf", page=1, page_end=1,
        section="", quote="营业收入 1 元",
    )
    evidence_b = SimpleNamespace(
        file_id="b", filename="汇川.pdf", page=1, page_end=1,
        section="", quote="营业收入 2 元",
    )
    rows = [
        SimpleNamespace(
            subject="鸣志电器", period="2026Q3", name="营业收入",
            value="1", unit="元", evidence=evidence_a,
        ),
        SimpleNamespace(
            subject="鸣志电器", period="2026Q3", name="毛利率",
            value="38", unit="%", evidence=evidence_a,
        ),
        SimpleNamespace(
            subject="汇川技术", period="2026Q3", name="营业收入",
            value="2", unit="元", evidence=evidence_b,
        ),
    ]
    result = run_deep_analysis(
        "机器人", "2026Q3", rows, mode="industry",
        config=DeepConfig(enabled=False),
    )
    assert "样本：鸣志电器" in result.target
    assert any("样本公司：鸣志电器" in note for note in result.execution.notes)
    sample, note = pick_industry_sample(SourceLedger.from_facts(rows))
    assert sample == "鸣志电器"
    assert "对照池" in note


def test_quality_is_propagated_to_cross_checks_and_checklist():
    log: list = []
    result = run_deep_analysis(
        "甲公司", "2026Q3", facts(),
        config=DeepConfig(enabled=True, review_rounds=1),
        complete=stub(log, "small"),
    )
    assert result.execution.convergence_quality in {"solid", "incomplete"}
    assert any("收敛质量门禁" in note for note in result.cross_checks)
    assert any(item.item == "第 9 章收敛未停在断言层" for item in result.checklist)


# ——— 对外可见性（Markdown 与合并 PDF 附录）———

def _visible_result() -> DeepAnalysisResult:
    return DeepAnalysisResult(
        target="甲公司", period="2026Q3", mode="company",
        convergence=Convergence(final_chain="毛利率←产品结构"),
        execution=ExecutionReport(
            chapter_model="qwen-small", convergence_model="gpt-main",
            model_route="small+primary", convergence_quality="shallow",
            convergence_checks=["❌ 核心链条停在断言层"],
            notes=["模型路由：small+primary"],
        ),
    )


def test_models_and_quality_visible_in_markdown():
    text = _visible_result().to_markdown()
    assert "章节/评审模型" in text
    assert "qwen-small" in text
    assert "第 9 章收敛模型" in text
    assert "gpt-main" in text
    assert "small+primary" in text
    assert "shallow · 停在断言层" in text
    assert "❌ 核心链条停在断言层" in text


def test_models_and_quality_visible_in_pdf_appendix():
    rendered = render_deep_appendix_html(_visible_result())
    assert "章节/评审模型" in rendered
    assert "qwen-small" in rendered
    assert "第 9 章收敛模型" in rendered
    assert "gpt-main" in rendered
    assert "small+primary" in rendered
    assert "shallow · 停在断言层" in rendered
    assert "❌ 核心链条停在断言层" in rendered
    assert "附录 D · 核心链条收敛与护城河" in rendered
