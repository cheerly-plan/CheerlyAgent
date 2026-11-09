"""单章评审轮次并发：真的并发、合并同序、调用数不变、预算闸不失效。

起因：3 路章节并行下，单章仍是「1 次分析 + 3 轮评审」串行 4 段，深度层
10–15 分钟里大头是评审轮次。轮次之间互不依赖（prompt 只带章节正文与台账，
不带上一轮意见），并发只是把 4 段压成 2 段，不改评审语义。
"""

from __future__ import annotations

import json
import re
import threading
import time
from types import SimpleNamespace

from ai_agent.report.deep import DeepConfig, run_deep_analysis
from ai_agent.report.deep.analyzer import (
    planned_review_rounds,
    review_chapter,
    review_round_count,
)
from ai_agent.report.deep.chapters import DEFAULT_EXPERTS, spec_of
from ai_agent.report.deep.ledger import SourceLedger
from ai_agent.report.deep.schema import ChapterResult

ROUND_RE = re.compile(r"【评审轮次】第 (\d+) 轮")


def ledger() -> SourceLedger:
    evidence = SimpleNamespace(
        file_id="f-1", filename="甲公司2026Q3.pdf", page=8, page_end=8,
        section="合并利润表", quote="营业收入 1,234,567 元",
    )
    return SourceLedger.from_facts([
        SimpleNamespace(
            subject="甲公司", period="2026Q3", name="营业收入",
            value="1,234,567", unit="元", evidence=evidence,
        )
    ])


def chapter() -> ChapterResult:
    return ChapterResult(
        number=4, title="ROE", question="?", method="杜邦",
        verdict="中等", status="draft",
    )


def review_payload(round_no: int) -> str:
    return json.dumps({"opinions": [{
        "expert": "财务专家",
        "issue": f"第{round_no}轮意见",
        "disposition": "adopted",
        "reason": "",
        "revision": "补充核对",
    }]}, ensure_ascii=False)


def test_planned_rounds_matches_schedule():
    assert planned_review_rounds(2) == [1, 2, 5]
    assert planned_review_rounds(5) == [1, 2, 3, 4, 5]
    assert review_round_count(1) == 2
    assert review_round_count(2) == 3


def test_rounds_are_really_concurrent():
    """三轮回调必须同时在飞；退回串行时 barrier 超时，用例明确失败。"""
    barrier = threading.Barrier(3, timeout=5)
    seen: list[int] = []
    lock = threading.Lock()

    def complete(_system: str, user: str) -> str:
        round_no = int(ROUND_RE.search(user).group(1))
        with lock:
            seen.append(round_no)
        barrier.wait()
        return review_payload(round_no)

    opinions = review_chapter(
        chapter(), ledger(), experts=DEFAULT_EXPERTS, rounds=2, prior="",
        complete=complete, round_concurrency=0,
    )
    assert sorted(seen) == [1, 2, 5]
    # 合并顺序固定按轮次，不受完成顺序影响
    assert [item.round_no for item in opinions] == [1, 2, 5]


def test_default_is_still_serial():
    """库级默认 round_concurrency=1：一次只有一轮在飞，调用数不变。"""
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def complete(_system: str, _user: str) -> str:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return review_payload(1)

    # 走库级默认（不传 round_concurrency）
    review_chapter(
        chapter(), ledger(), experts=DEFAULT_EXPERTS, rounds=2, prior="",
        complete=complete,
    )
    assert peak == 1


def test_concurrent_and_serial_produce_same_opinions_and_call_count():
    def build(round_concurrency: int) -> tuple[list[tuple], int]:
        calls = 0
        lock = threading.Lock()

        def complete(_system: str, user: str) -> str:
            nonlocal calls
            round_no = int(ROUND_RE.search(user).group(1))
            with lock:
                calls += 1
            time.sleep(0.01)  # 让完成顺序与轮次顺序错开
            return review_payload(round_no)

        opinions = review_chapter(
            chapter(), ledger(), experts=DEFAULT_EXPERTS, rounds=2, prior="",
            complete=complete, round_concurrency=round_concurrency,
        )
        rows = [
            (item.round_no, item.expert, item.issue, item.disposition, item.revision)
            for item in opinions
        ]
        return rows, calls

    serial_rows, serial_calls = build(1)
    concurrent_rows, concurrent_calls = build(0)
    assert serial_rows == concurrent_rows
    assert serial_calls == concurrent_calls == 3


def test_concurrent_rounds_shorten_wall_clock():
    def slow(round_concurrency: int) -> float:
        def complete(_system: str, _user: str) -> str:
            time.sleep(0.09)
            return review_payload(1)

        start = time.perf_counter()
        review_chapter(
            chapter(), ledger(), experts=DEFAULT_EXPERTS, rounds=2, prior="",
            complete=complete, round_concurrency=round_concurrency,
        )
        return time.perf_counter() - start

    serial, concurrent = slow(1), slow(0)
    # 3 轮串行 ≈ 3t，并发 ≈ t；留足 CI 抖动余量
    assert concurrent < serial * 0.7, f"串行 {serial:.3f}s / 并发 {concurrent:.3f}s"


def test_on_round_callback_counts_up_in_caller_thread():
    seen: list[tuple[int, int]] = []
    opinions = review_chapter(
        chapter(), ledger(), experts=DEFAULT_EXPERTS, rounds=2, prior="",
        complete=lambda _system, _user: review_payload(1),
        round_concurrency=0,
        on_round=lambda done, total: seen.append((done, total)),
    )
    assert opinions
    # 回调串行递增，进度文案不会出现 2/3 早于 1/3
    assert seen == [(1, 3), (2, 3), (3, 3)]


def _deep_payload(system: str) -> str:
    if system.startswith("你是章节评审专家组"):
        return review_payload(1)
    if system.startswith("你是核心链条收敛"):
        return json.dumps({
            "final_chain": "毛利率←产品结构", "decisions": [], "delta": "未推翻",
            "moat_rating": "中等", "moat_sources": {}, "falsifiers": [],
            "tracking": [], "scenarios": [],
        }, ensure_ascii=False)
    return json.dumps({
        "verdict": "评级：中（台账内事实可支撑）",
        "claims": [], "gaps": [], "tracking": [], "falsifiers": [],
    }, ensure_ascii=False)


def _facts():
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


def test_pipeline_emits_finer_progress_from_workers():
    from ai_agent.tools.progress import reset_progress, set_progress

    messages: list[str] = []
    token = set_progress(messages.append)
    try:
        result = run_deep_analysis(
            "甲公司", "2026Q3", _facts(),
            config=DeepConfig(
                enabled=True, chapters=(2, 3, 9), review_rounds=2,
                chapter_concurrency=2, review_round_concurrency=0,
            ),
            complete=lambda system, _user: _deep_payload(system),
        )
    finally:
        reset_progress(token)

    assert [chapter.number for chapter in result.chapters] == [2, 3, 9]
    analyzed = [m for m in messages if "已撰写，进入评审" in m]
    rounds = [m for m in messages if "评审进度" in m]
    assert len(analyzed) == 2, messages
    assert len(rounds) == 6, messages
    assert all("3 轮并发" in m for m in analyzed)
    assert any("章节并行" in note for note in result.execution.notes)
    assert any("评审轮次并行" in note for note in result.execution.notes)


def test_budget_gate_still_holds_with_concurrent_rounds():
    result = run_deep_analysis(
        "甲公司", "2026Q3", _facts(),
        config=DeepConfig(
            enabled=True, chapters=(2,), review_rounds=1,
            chapter_concurrency=1, review_round_concurrency=0,
            max_llm_calls=2,
        ),
        complete=lambda system, _user: _deep_payload(system),
    )
    assert result.execution.llm_calls <= 2
    assert any("预算闸" in note for note in result.execution.notes)
    assert result.chapters[0].status == "skipped"


def test_spec_of_chapter_four_title_is_stable():
    # 并发用例依赖章号定位；标题变了说明章节表被改，评审 prompt 也要跟着回归
    assert spec_of(4).title
