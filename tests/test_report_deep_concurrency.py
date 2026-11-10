"""第 2-8 章并行：真的并发、结果确定性、预算闸不失效、串行语义不回退。"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ai_agent.report.deep import DeepConfig, run_deep_analysis
from ai_agent.report.deep.analyzer import default_complete
from ai_agent.report.deep.pipeline import ENTRY_CHAPTER_CONCURRENCY


def facts():
    return [
        SimpleNamespace(
            subject="甲公司", period="2026Q3", name=name, value=value, unit="元",
            evidence=SimpleNamespace(
                file_id=f"f-{i}", filename="甲公司2026Q3.pdf", page=8 + i,
                section="合并利润表", quote=f"{name} {value} 元",
            ),
        )
        for i, (name, value) in enumerate(
            [("营业收入", "1,614,059,366.89"), ("研发投入", "205,260,194.78")]
        )
    ]


def payload(system: str) -> str:
    if system.startswith("你是章节评审专家组"):
        return json.dumps({"opinions": [{
            "expert": "财务专家", "issue": "现金流未与利润交叉核对",
            "disposition": "adopted", "reason": "", "revision": "补核对现金流量表",
        }]}, ensure_ascii=False)
    if system.startswith("你是核心链条收敛"):
        return json.dumps({
            "final_chain": "毛利率←产品结构←工艺壁垒", "decisions": [],
            "delta": "未推翻", "moat_rating": "中等", "moat_sources": {},
            "falsifiers": [], "tracking": [], "scenarios": [],
        }, ensure_ascii=False)
    return json.dumps({
        "verdict": "评级：中（台账内事实可支撑）",
        "claims": [], "gaps": [], "tracking": [], "falsifiers": [],
    }, ensure_ascii=False)


def test_parallel_chapters_really_run_together_and_stay_ordered():
    # 三章必须同时进入章节分析；若退回串行，barrier 超时会让用例明确失败。
    barrier = threading.Barrier(3, timeout=4)
    users: list[str] = []
    lock = threading.Lock()

    def complete(system: str, user: str) -> str:
        if system.startswith("你是上市公司结构化深度投资分析助手"):
            barrier.wait()
            with lock:
                users.append(user)
        return payload(system)

    try:
        result = run_deep_analysis(
            "甲公司", "2026Q3", facts(),
            config=DeepConfig(
                enabled=True, chapters=(2, 3, 4, 9), review_rounds=1,
                chapter_concurrency=3,
            ),
            complete=complete,
        )
    except Exception as exc:  # barrier 破裂会原样冒泡，转成可读断言
        pytest.fail(f"章节没有并行执行：{type(exc).__name__}: {exc}")

    assert [chapter.number for chapter in result.chapters] == [2, 3, 4, 9]
    assert [chapter.status for chapter in result.chapters] == ["reviewed"] * 4
    assert len(users) == 3
    # 并行时每章拿到同一份确定性基线，不含别的章节结论。
    assert all("候选主线" in user and "台账要点" in user for user in users)
    assert all("第2章 " not in user for user in users)
    assert any("章节并行" in note for note in result.execution.notes)


def test_serial_mode_keeps_prior_chapter_chain():
    seen: list[str] = []

    def complete(system: str, user: str) -> str:
        if system.startswith("你是上市公司结构化深度投资分析助手"):
            seen.append(user)
        return payload(system)

    run_deep_analysis(
        "甲公司", "2026Q3", facts(),
        config=DeepConfig(
            enabled=True, chapters=(2, 3, 4, 9), review_rounds=1,
            chapter_concurrency=1,
        ),
        complete=complete,
    )
    assert len(seen) == 3
    assert "第2章" in seen[1] and "第3章" in seen[2]
    assert not any("候选主线：" in user for user in seen)


def test_library_default_is_still_serial():
    assert DeepConfig().chapter_concurrency == 1
    assert ENTRY_CHAPTER_CONCURRENCY > 1


def test_parallel_reduces_wall_clock_with_slow_completion():
    """受控量化：同样的章节与调用次数，并发把章节段耗时压到接近 1/N。"""
    import time

    def slow(system: str, _user: str) -> str:
        time.sleep(0.08)
        return payload(system)

    def elapsed(concurrency: int) -> float:
        start = time.perf_counter()
        run_deep_analysis(
            "甲公司", "2026Q3", facts(),
            config=DeepConfig(
                enabled=True, chapters=(2, 3, 4, 9), review_rounds=1,
                chapter_concurrency=concurrency,
            ),
            complete=slow,
        )
        return time.perf_counter() - start

    serial, parallel = elapsed(1), elapsed(3)
    assert parallel < serial * 0.75, f"串行 {serial:.3f}s / 并发 {parallel:.3f}s"


def test_parallel_progress_follows_completion_order():
    """第 2 章慢时，先完成的章节要立刻推进度，不能被按章号阻塞压住。"""
    import time

    from ai_agent.tools.progress import reset_progress, set_progress

    messages: list[str] = []
    token = set_progress(messages.append)

    def slow(system: str, user: str) -> str:
        # 只有章节分析（不是评审）里能定位章号；让第 2 章明显最慢
        if (
            system.startswith("你是上市公司结构化深度投资分析助手")
            and "第 2 章" in user
        ):
            time.sleep(0.2)
        return payload(system)

    try:
        result = run_deep_analysis(
            "甲公司", "2026Q3", facts(),
            config=DeepConfig(
                enabled=True, chapters=(2, 3, 4, 9), review_rounds=1,
                chapter_concurrency=3,
            ),
            complete=slow,
        )
    finally:
        reset_progress(token)

    progress = [m for m in messages if "深度分析章节进度" in m]
    assert len(progress) == 3
    assert "第2章" not in progress[0], progress
    # 进度按完成顺序推，结果汇总仍按章节号
    assert [c.number for c in result.chapters] == [2, 3, 4, 9]


def test_parallel_budget_gate_never_exceeds_limit():
    result = run_deep_analysis(
        "甲公司", "2026Q3", facts(),
        config=DeepConfig(
            enabled=True, chapters=(2, 3, 4, 9), review_rounds=1,
            chapter_concurrency=3, max_llm_calls=4,
        ),
        complete=lambda system, _user: payload(system),
    )
    assert result.execution.llm_calls <= 4
    assert any("预算闸" in note for note in result.execution.notes)
    assert any(chapter.status == "skipped" for chapter in result.chapters)


def test_default_complete_survives_concurrent_empty_responses(monkeypatch):
    """共享补全函数在多线程下不能把熔断状态或事件列表搞乱。"""
    from ai_agent.report.deep import analyzer

    calls: list[str] = []
    lock = threading.Lock()

    def fake_chat(messages, *, model, **_kwargs):
        with lock:
            calls.append(model)
        diagnostics = _kwargs.get("diagnostics")
        if isinstance(diagnostics, dict) and model == "qwen-small":
            diagnostics.update({"finish_reason": "length", "content_chars": 0,
                                "reasoning_chars": 851, "completion_tokens": 1600})
        return "" if model == "qwen-small" else '{"verdict":"ok"}'

    monkeypatch.setattr(analyzer, "lightweight_llm", lambda name: (name, None))
    monkeypatch.setattr(analyzer, "complete_chat", fake_chat)

    events: list[str] = []
    complete = default_complete(
        1600, 0.1, 90, "qwen-small",
        escalation_model="gpt-main", events=events,
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda i: complete("sys", f"u{i}"), range(8)))
    assert outputs == ['{"verdict":"ok"}'] * 8
    assert events
    assert any("自动升级" in event for event in events)
    # 并发下熔断事件只能有一条，不能刷成「连续 2 次 / 3 次」。
    assert sum("熔断" in event for event in events) == 1
    # 熔断后不再每次空跑小模型：总调用数必然小于 2×8。
    assert len(calls) < 16
