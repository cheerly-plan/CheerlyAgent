"""研报进度：文案构造与阶段百分比同源、单调。"""

from __future__ import annotations

from ai_agent.report.progress import (
    MAX_RUNNING_PROGRESS,
    advance_progress,
    msg_compare_and_compose,
    msg_deep_bypass,
    msg_deep_chapter_progress,
    msg_deep_chapter_serial,
    msg_deep_chapters_batch,
    msg_deep_converge_ch9,
    msg_deep_facts_ready,
    msg_ensure_deep_annuals,
    msg_ensure_filings,
    msg_entry_with_deep,
    msg_extract_deep_facts,
    msg_extract_indicators,
    msg_merge_deep,
    msg_render_pdf,
    msg_report_done,
    msg_separate_deep_pdf,
    msg_understanding,
    msg_validate,
    msg_write_artifacts,
    stage_percent,
)


def test_stage_percent_maps_msg_builders():
    """msg_* 产出必须能被 STAGE_NEEDLES 命中，且深度细档离开抽数 30%。"""
    assert stage_percent(msg_understanding()) == 5
    assert stage_percent(msg_ensure_filings()) == 12
    assert stage_percent(msg_ensure_deep_annuals(years=5)) == 18
    assert stage_percent(msg_extract_indicators()) == 30
    assert stage_percent(msg_compare_and_compose()) == 45
    assert stage_percent(msg_validate()) == 58
    assert stage_percent(msg_render_pdf()) == 70
    assert stage_percent(msg_deep_bypass()) == 71
    assert stage_percent(msg_extract_deep_facts()) == 72
    assert stage_percent(msg_deep_facts_ready(fact_count=12)) == 74
    assert stage_percent(msg_deep_chapters_batch(workers=3, parallel=True)) == 76
    assert stage_percent(
        msg_deep_chapter_serial(number=3, title="护城河", index=2, total=7)
    ) == 80
    assert stage_percent(
        msg_deep_chapter_progress(done=3, total=7, number=4, title="估值")
    ) == 86
    assert stage_percent(msg_deep_converge_ch9()) == 92
    assert stage_percent(msg_merge_deep()) == 93
    assert stage_percent(msg_separate_deep_pdf()) == 94
    assert stage_percent(msg_write_artifacts()) == 95
    assert stage_percent(msg_report_done()) == 96
    assert "10–15 分钟" in msg_entry_with_deep()


def test_deep_stage_progress_is_monotonic_and_leaves_extract_tier():
    messages = [
        msg_understanding(),
        msg_extract_indicators(),
        msg_validate(),
        msg_render_pdf(),
        msg_deep_bypass(),
        msg_ensure_deep_annuals(years=5),
        msg_extract_deep_facts(),
        msg_deep_facts_ready(fact_count=8),
        msg_deep_chapters_batch(workers=3, parallel=True),
        msg_deep_chapter_progress(done=1, total=7, number=2, title="收入"),
        msg_deep_chapter_progress(done=7, total=7, number=8, title="风险"),
        msg_deep_converge_ch9(),
        msg_merge_deep(),
        msg_separate_deep_pdf(),
        msg_write_artifacts(),
        msg_report_done(),
    ]
    progress = 0
    seen: list[int] = []
    for message in messages:
        progress = advance_progress(progress, message)
        seen.append(progress)
    assert seen == sorted(seen)
    assert any(p == 30 for p in seen)
    assert any(p > 30 for p in seen)
    assert max(seen) == MAX_RUNNING_PROGRESS
    assert 72 in seen or 74 in seen
    assert 86 in seen or 92 in seen


def test_merge_deep_not_stolen_by_generic_deep_needle():
    """「合并深度分析附录」不得被笼统「深度分析」抢成 86。"""
    assert stage_percent(msg_merge_deep()) == 93
    assert stage_percent(msg_separate_deep_pdf()) == 94
    assert stage_percent(msg_deep_bypass()) == 71
