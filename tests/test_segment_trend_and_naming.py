"""赛道报告：同比／不可比趋势 + 财报对比底稿命名与下期数据源清单。"""

from ai_agent.report.compare import Measure, SeriesCompare, compare_measures
from ai_agent.report.compose_common import segment_reading_note
from ai_agent.report.compose_lead import change_lines
from ai_agent.report.industry import notes_for_plan
from ai_agent.report.parse import OutlineItem, ReportPlan, parse_request
from ai_agent.report.skills import get_skill, reload_skills


def test_segment_title_is_draft_not_deep_dive(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [],
    )
    plan = parse_request("生成机器人行业2026Q3财报对比分析")
    assert "财报对比底稿" in plan.title
    assert "深度分析" not in plan.title
    assert "赛道分组版" in plan.title


def test_segment_reading_note_calls_draft():
    plan = ReportPlan(
        title="机器人行业 2026Q3 财报对比底稿（赛道分组版）",
        kind="industry",
        deliverable="markdown",
        skill_id="robotics_industry_report",
        industry="机器人",
        subjects=["汇川技术"],
        period="2026Q3",
        indicators=["毛利率"],
        outline=[OutlineItem("阅读说明", "")],
    )
    text = "\n".join(segment_reading_note(plan))
    assert "财报对比底稿" in text


def test_change_lines_states_incomparable_instead_of_silence():
    plan = ReportPlan(
        title="机器人行业 2026Q3 财报对比底稿（赛道分组版）",
        kind="industry",
        deliverable="markdown",
        skill_id="robotics_industry_report",
        industry="机器人",
        subjects=["汇川技术", "新松机器人"],
        period="2026Q3",
        dimensions=["同比", "环比"],
        indicators=["营业收入"],
        outline=[OutlineItem("同比与趋势", "", "change")],
    )
    series = [
        SeriesCompare(
            subject="汇川技术",
            name="营业收入",
            scope="",
            unit="元",
            points=[("2026Q3", "10,000")],
            gaps=["2026Q3 与 2025Q3 期间口径不同，不算同比"],
        ),
        SeriesCompare(
            subject="新松机器人",
            name="营业收入",
            scope="",
            unit="元",
            points=[("2026Q3", "8,000")],
            gaps=[],
        ),
    ]
    lines = change_lines(plan, series)
    text = "\n".join(lines)
    assert "没有可写入的同比或环比" not in text
    assert "不可比或缺基期" in text or "缺同比／环比基期" in text
    assert "汇川技术" in text
    assert "新松机器人" in text


def test_change_lines_keeps_real_yoy_and_notes_remainder():
    measures = [
        Measure(
            subject="汇川技术",
            name="营业收入",
            period="2026Q3",
            value="120",
            unit="亿元",
            basis="cumulative",
            coverage_months=9,
        ),
        Measure(
            subject="汇川技术",
            name="营业收入",
            period="2025Q3",
            value="100",
            unit="亿元",
            basis="cumulative",
            coverage_months=9,
        ),
        Measure(
            subject="新松机器人",
            name="营业收入",
            period="2026Q3",
            value="80",
            unit="亿元",
            basis="cumulative",
            coverage_months=9,
        ),
    ]
    series = compare_measures(measures)
    plan = ReportPlan(
        title="t",
        kind="industry",
        deliverable="markdown",
        skill_id="robotics_industry_report",
        industry="机器人",
        subjects=["汇川技术", "新松机器人"],
        period="2026Q3",
        dimensions=["同比"],
        indicators=["营业收入"],
        outline=[OutlineItem("同比与趋势", "", "change")],
    )
    text = "\n".join(change_lines(plan, series))
    assert "汇川技术" in text and "同比" in text
    assert "缺基期或口径不可比" in text or "新松机器人" in text


def test_robotics_next_period_lists_external_sources():
    plan = ReportPlan(
        title="t",
        kind="industry",
        deliverable="markdown",
        skill_id="robotics_industry_report",
        industry="机器人",
        subjects=["汇川技术"],
        period="2026Q3",
        indicators=["毛利率"],
        outline=[],
    )
    notes = notes_for_plan(plan)
    text = "\n".join(notes.next_period)
    assert "建议接入数据源" in text
    assert "国家统计局" in text
    assert "对比底稿" in text
    assert "不构成行业深度研究" in text


def test_segment_skills_include_change_section():
    reload_skills()
    for skill_id in (
        "robotics_industry_report",
        "semiconductor_industry_report",
        "bank_industry_report",
    ):
        skill = get_skill(skill_id)
        assert skill is not None
        assert "change" in skill.section_ids
        assert "财报对比底稿" in skill.name
