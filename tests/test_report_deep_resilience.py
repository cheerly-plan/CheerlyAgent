"""P0 韧性回归：空 content 自动升级、结论数字门禁、降级诊断可见。

全部用例零真实 token（注入 stub / 打桩 complete_chat），固化 2026-11-09
实跑发现的故障：本地 qwen3 思考型小模型把 max_tokens 全耗在 reasoning 上，
返回空 content，导致 4/5 个章节降级且事后无法定位。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import ai_agent.report.deep.analyzer as analyzer
from ai_agent.report.deep import DeepConfig, run_deep_analysis
from ai_agent.report.deep.analyzer import (
    analyze_chapter,
    default_complete,
    guard_conclusion,
    guard_numbers,
)
from ai_agent.report.deep.chapters import spec_of
from ai_agent.report.deep.ledger import SourceLedger
from ai_agent.report.deep.pdf_appendix import render_deep_appendix_html
from ai_agent.report.deep.schema import ChapterResult, DeepAnalysisResult


def fact():
    evidence = SimpleNamespace(
        file_id="7b75379b-0002", filename="甲公司2026Q3.pdf", page=8,
        page_end=8, section="合并利润表", quote="营业收入 1,234,567 元",
    )
    return SimpleNamespace(
        subject="甲公司", period="2026Q3", name="营业收入",
        value="1,234,567", unit="元", evidence=evidence,
    )


def ledger() -> SourceLedger:
    return SourceLedger.from_facts([fact()])


def facts():
    return [fact()]


def chapter_payload(verdict: str) -> str:
    return json.dumps({
        "verdict": verdict,
        "claims": [], "gaps": [], "tracking": [], "falsifiers": [],
    }, ensure_ascii=False)


def analyze(verdict_or_raw: str, number: int = 4) -> ChapterResult:
    return analyze_chapter(
        spec_of(number), ledger(), target="甲公司", period="2026Q3",
        mode="company", prior="", complete=lambda _s, _u: verdict_or_raw,
    )


# ——— 数字台账门禁：verdict / final_chain 不再绕过 ———

def test_guard_numbers_keeps_ledger_values_and_years():
    safe, blocked = guard_numbers(
        "营业收入 1,234,567 元，毛利率 42.5%，2026Q3 第 4 章",
        ledger().allowed_numbers,
    )
    assert "1,234,567" in safe
    assert "2026" in safe and "第 4 章" in safe
    assert "42.5%" not in safe
    assert "〔未溯源〕" in safe
    assert blocked == ["42.5%"]


def test_guard_conclusion_records_gap_only_when_blocked():
    text, gap = guard_conclusion("评级：中", ledger(), 5)
    assert text == "评级：中" and gap is None
    masked, gap = guard_conclusion("净利润 1.17 亿元，同比 +227.60%", ledger(), 5)
    assert gap is not None and gap.chapters == [5]
    assert "台账外数字" in gap.item
    assert "1.17" not in masked and "227.60" not in masked
    assert masked.count("〔未溯源〕") == 2


def test_chapter_verdict_passes_number_gate():
    result = analyze(chapter_payload("扣非净利润 1.17 亿元，同比 +227.60%"))
    assert "1.17" not in result.verdict and "227.60" not in result.verdict
    assert "〔未溯源〕" in result.verdict
    assert any("台账外数字" in gap.item for gap in result.gaps)


# ——— 降级原因分类：空响应 / 非 JSON / 调用异常 ———

def test_empty_content_fallback_reason():
    result = analyze("")
    assert result.status == "fallback"
    assert "空 content" in result.fallback_reason
    assert "reasoning" in result.fallback_reason


def test_bad_json_fallback_reason_contains_raw_excerpt():
    result = analyze("抱歉，我无法完成该分析", number=5)
    assert result.status == "fallback"
    assert "未解析成 JSON 对象" in result.fallback_reason
    assert "抱歉" in result.fallback_reason


def test_llm_exception_fallback_reason():
    def boom(_s, _u):
        raise RuntimeError("gateway down")

    result = analyze_chapter(
        spec_of(6), ledger(), target="甲公司", period="2026Q3",
        mode="company", prior="", complete=boom,
    )
    assert result.status == "fallback"
    assert "LLM 调用异常：RuntimeError" in result.fallback_reason


# ——— 空响应自动升级主模型（小模型优先，失败才升级）———

def _install_fake_gateway(monkeypatch, handler, calls: list[str]):
    monkeypatch.setattr(
        analyzer, "lightweight_llm", lambda model: (model or "qwen-small", object())
    )

    def fake_chat(_messages, *, diagnostics=None, **kwargs):
        model = kwargs.get("model", "")
        calls.append(model)
        if diagnostics is not None:
            diagnostics.update({
                "finish_reason": "length", "content_chars": 0,
                "reasoning_chars": 851, "completion_tokens": 1600,
            })
        return handler(model)

    monkeypatch.setattr(analyzer, "complete_chat", fake_chat)


def test_default_complete_escalates_on_empty_content(monkeypatch):
    calls: list[str] = []
    _install_fake_gateway(
        monkeypatch,
        lambda model: "" if model == "qwen-small" else '{"verdict":"ok"}',
        calls,
    )
    events: list[str] = []
    complete = default_complete(
        1600, 0.1, 90, "", escalation_model="gpt-main", events=events
    )
    assert complete("sys", "user") == '{"verdict":"ok"}'
    assert calls == ["qwen-small", "gpt-main"]
    assert len(events) == 1
    assert "自动升级" in events[0] and "reasoning_chars=851" in events[0]


def test_default_complete_records_event_without_escalation_model(monkeypatch):
    calls: list[str] = []
    _install_fake_gateway(monkeypatch, lambda _model: "", calls)
    events: list[str] = []
    complete = default_complete(1600, 0.1, 90, "", events=events)
    assert complete("sys", "user") == ""
    assert calls == ["qwen-small"]
    assert len(events) == 1 and "无可升级模型" in events[0]


def test_circuit_breaker_skips_repeated_empty_small_model(monkeypatch):
    calls: list[str] = []
    _install_fake_gateway(
        monkeypatch,
        lambda model: "" if model == "qwen-small" else '{"verdict":"ok"}',
        calls,
    )
    events: list[str] = []
    complete = default_complete(
        1600, 0.1, 90, "", escalation_model="gpt-main", events=events
    )
    for _ in range(3):
        assert complete("sys", "user") == '{"verdict":"ok"}'
    # 前两次：小模型空 → 升级主模型；第三次起熔断，不再空跑小模型。
    assert calls == [
        "qwen-small", "gpt-main", "qwen-small", "gpt-main", "gpt-main",
    ]
    assert any("熔断" in event for event in events)


def test_nonempty_response_never_escalates(monkeypatch):
    calls: list[str] = []
    _install_fake_gateway(monkeypatch, lambda _model: '{"verdict":"ok"}', calls)
    events: list[str] = []
    complete = default_complete(
        1600, 0.1, 90, "", escalation_model="gpt-main", events=events
    )
    assert complete("sys", "user") == '{"verdict":"ok"}'
    assert calls == ["qwen-small"] and events == []


# ——— 编排层：收敛链与章节 verdict 同过门禁，诊断对外可见 ———

def convergence_stub(system: str, _user: str) -> str:
    if system.startswith("你是章节评审专家组"):
        # 真实产物里评审意见曾原样引用台账外数字，这里固化成回归样例。
        return json.dumps({"opinions": [{
            "expert": "财务专家",
            "issue": "扣非净利润 1.17 亿元与合同负债 3.92 亿元缺少台账溯源",
            "disposition": "adopted", "reason": "",
            "revision": "补核对利润表与资产负债表附注",
        }]}, ensure_ascii=False)
    if system.startswith("你是核心链条收敛"):
        return json.dumps({
            "final_chain": "扣非净利润 1.17 亿元，同比 +227.60%，因产能爬坡形成机制",
            "decisions": [{"mainline": "甲公司", "decision": "确认",
                           "reason": "营业收入 9.99 亿元提供支撑"}],
            "delta": "候选主线中 3 条被修正",
            "moat_rating": "暂无法评级",
            "moat_sources": {
                "品牌": "未找到实证", "成本优势": "未找到实证",
                "网络效应": "未找到实证", "转换成本": "未找到实证",
                "特许/牌照": "未找到实证",
            },
            "falsifiers": [{"judgment": "成本", "signal": "单位成本上行", "severity": "高"}],
            "tracking": [{"judgment": "成本", "indicator": "单位成本",
                          "direction": "上行=证伪", "source": "季报", "frequency": "季度"}],
            "scenarios": [
                {"name": "基准", "probability": 0.55, "assumption": "持平",
                 "trigger": "稳定", "impact": "维持"},
                {"name": "乐观", "probability": 0.25, "assumption": "上行",
                 "trigger": "改善", "impact": "上调"},
                {"name": "悲观", "probability": 0.20, "assumption": "下行",
                 "trigger": "恶化", "impact": "下调"},
            ],
        }, ensure_ascii=False)
    return chapter_payload("评级：中")


def test_convergence_final_chain_and_verdict_both_guarded():
    result = run_deep_analysis(
        "甲公司", "2026Q3", facts(),
        config=DeepConfig(enabled=True, review_rounds=1),
        complete=convergence_stub,
    )
    assert result.convergence is not None
    assert "1.17" not in result.convergence.final_chain
    assert "227.60" not in result.convergence.final_chain
    ch9 = next(chapter for chapter in result.chapters if chapter.number == 9)
    assert "1.17" not in ch9.verdict
    assert any("台账外数字" in gap.item for gap in ch9.gaps)
    markdown = result.to_markdown()
    for leaked in ("1.17", "227.60", "3.92", "9.99"):
        assert leaked not in markdown, f"推断层仍出现台账外数字 {leaked}"


def test_all_model_supplied_text_has_no_untraceable_numbers():
    """门禁边界：模型给出的自由文本零台账外数字；工程计量与台账自带表不在此列。"""
    from ai_agent.report.deep.analyzer import numbers_allowed

    result = run_deep_analysis(
        "甲公司", "2026Q3", facts(),
        config=DeepConfig(enabled=True, review_rounds=1),
        complete=convergence_stub,
    )
    allowed = ledger().allowed_numbers
    texts: list[str] = []
    for chapter in result.chapters:
        texts.extend([chapter.verdict, chapter.fallback_reason])
        texts.extend(gap.item for gap in chapter.gaps)
        texts.extend(claim.text for claim in chapter.claims)
        texts.extend(opinion.issue for opinion in chapter.review)
    if result.convergence is not None:
        texts.extend([result.convergence.final_chain, result.convergence.delta])
        texts.extend(str(row.get("reason", "")) for row in result.convergence.decisions)
    texts.extend(item.signal for item in result.falsifiers)
    texts.extend(item.indicator for item in result.tracking)
    bad = [text for text in texts if not numbers_allowed(text, allowed)]
    assert not bad, f"仍有台账外数字：{bad[:3]}"


def test_fallback_reason_visible_in_markdown_and_pdf():
    result = DeepAnalysisResult(
        target="甲公司", period="2026Q3", mode="company",
        chapters=[ChapterResult(
            number=4, title="ROE 质量与来源", question="?", method="杜邦",
            status="fallback", verdict="本章输出无法解析，结论从略",
            fallback_reason="模型返回空 content：reasoning 耗尽 token",
        )],
    )
    assert "模型返回空 content" in result.to_markdown()
    html = render_deep_appendix_html(result, standalone=True)
    assert "降级原因" in html and "reasoning 耗尽 token" in html
