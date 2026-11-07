"""新增行业解耦的架构护栏。

守住四条：行业档案与 skill 自洽、赛道都有映射、行业文案不串味、
渲染层不再出现 ``skill_id == "xxx_industry_report"`` 这类行业专属分支。
新增第三个行业时这些测试会自动覆盖它，不需要补断言。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_agent.report import industry, registry, skills

_SEGMENT_SKILLS = tuple(
    skill
    for skill in skills.SKILLS
    if skill.kind == "industry" and skill.companies
)
_REPORT_ROOT = Path(__file__).resolve().parents[1] / "src" / "ai_agent" / "report"


@pytest.mark.parametrize("skill", _SEGMENT_SKILLS, ids=lambda s: s.skill_id)
def test_industry_skill_profile_is_consistent(skill):
    """indicator_set 必须指向已登记 profile，且指标集与该 profile 一致。"""
    assert skill.indicator_set
    assert skill.indicator_set in industry.known_company_types()
    assert tuple(industry.default_indicators(skill.indicator_set)) == skill.indicators


@pytest.mark.parametrize("skill", _SEGMENT_SKILLS, ids=lambda s: s.skill_id)
def test_roster_segments_all_have_type_mapping(skill):
    """花名册赛道漏配 segment_to_type 会静默跨行业污染，必须由 lint 提前拦住。"""
    known = dict(registry.segment_map())
    for name, _code, segment, _aka in skill.companies:
        seg = (segment or "").strip()
        assert seg in known, f"{skill.skill_id} 的 {name} 赛道 {seg!r} 未登记"


@pytest.mark.parametrize("skill", _SEGMENT_SKILLS, ids=lambda s: s.skill_id)
def test_notes_for_plan_returns_own_industry_notes(skill):
    """每个行业只拿到自己的定性文案，不靠 skill_id 分支。"""
    plan = SimpleNamespace(
        skill_id=skill.skill_id,
        industry=skill.industries[0] if skill.industries else "",
    )
    assert industry.notes_for_plan(plan) == industry.profile_of(
        skill.indicator_set
    ).notes


def test_next_period_head_is_industry_specific():
    """行业专属首句各不相同（同一句通用收尾可以共用）。"""
    heads = [
        industry.profile_of(skill.indicator_set).notes.next_period[0]
        for skill in _SEGMENT_SKILLS
    ]
    assert heads
    assert len(set(heads)) == len(heads)


def test_unknown_segment_never_falls_back_to_another_industry():
    assert industry.type_for_segment("完全未登记的赛道") == industry.GENERIC


def test_render_layer_has_no_industry_specific_branches():
    """静态护栏：行业专属 if 不许在渲染/分析/路由层长回来。"""
    for name in (
        "compose.py",
        "analyze.py",
        "parse.py",
        "pdf.py",
        "market_finance.py",
    ):
        src = (_REPORT_ROOT / name).read_text(encoding="utf-8")
        for banned in (
            "semiconductor_industry_report",
            "robotics_industry_report",
            "_NEXT_PERIOD_NOTE_SEMICONDUCTOR",
            "_NEXT_PERIOD_NOTE_ROBOTICS",
        ):
            assert banned not in src, f"{name} 又出现了行业专属分支 {banned}"


def test_classification_defaults_is_generated_not_handwritten():
    from ai_agent.report import classification_snapshot

    path = classification_snapshot.DEFAULTS_PY
    assert path.exists()
    assert "自动生成，请勿手工编辑" in path.read_text(encoding="utf-8")
    expected = classification_snapshot.render_defaults(
        _REPORT_ROOT / "config" / "indicators.yaml",
        _REPORT_ROOT / "config" / "industries.yaml",
    )
    assert path.read_text(encoding="utf-8") == expected
