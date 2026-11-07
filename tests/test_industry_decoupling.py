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
    """赛道必须登记在**该行业自己的** segments 里，不能靠全局字典凑数。"""
    assert industry.profile_of(skill.indicator_set).segments, (
        f"{skill.skill_id} 的 profile 未声明 segments，会回退全局字典串味"
    )
    own = industry.segment_map_for(skill.indicator_set)
    for name, _code, segment, _aka in skill.companies:
        seg = (segment or "").strip()
        assert seg in own, f"{skill.skill_id} 的 {name} 赛道 {seg!r} 未在本行业登记"


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
    paths = [
        _REPORT_ROOT / "compose.py",
        _REPORT_ROOT / "compose_common.py",
        _REPORT_ROOT / "compose_format.py",
        _REPORT_ROOT / "compose_segment.py",
        _REPORT_ROOT / "compose_gaps.py",
        _REPORT_ROOT / "analyze.py",
        _REPORT_ROOT / "parse.py",
        _REPORT_ROOT / "pdf.py",
        _REPORT_ROOT / "market_finance.py",
    ]
    banned = (
        "semiconductor_industry_report",
        "robotics_industry_report",
        "bank_industry_report",
        "bank_financial_report",
        "_NEXT_PERIOD_NOTE_SEMICONDUCTOR",
        "_NEXT_PERIOD_NOTE_ROBOTICS",
        "_SEMICONDUCTOR_PREFER_MARKET",
    )
    for path in paths:
        src = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in src, f"{path.name} 又出现了行业专属分支 {token}"


@pytest.mark.parametrize("skill", _SEGMENT_SKILLS, ids=lambda s: s.skill_id)
def test_segment_skills_declare_structure(skill):
    """有赛道花名册就必须声明 structure，禁止默认走毛利率叙事。"""
    profile = industry.profile_of(skill.indicator_set)
    assert profile.structure, f"{skill.skill_id} 缺少 structure"
    for name in profile.structure:
        assert name in profile.indicators


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
