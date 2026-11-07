"""行业级赛道语义的架构护栏。

守住一条核心性质：**赛道名不是全局主键**。「设备」在 A 行业可以指
equipment 模板，在 B 行业可以指 cyclical 模板；声明了 segments 的行业
只读自己那份，未登记的赛道落 generic，绝不静默套用别的行业模板。
"""

from __future__ import annotations

import ast
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from ai_agent.report import classification_defaults, classification_snapshot
from ai_agent.report import industry, registry

_REPORT_ROOT = Path(__file__).resolve().parents[1] / "src" / "ai_agent" / "report"
_CONFIG_DIR = _REPORT_ROOT / "config"


def _fake_profiles(monkeypatch, **overrides):
    """在内存里替换 profile 表，不改磁盘配置。"""
    profiles = dict(industry._profiles())
    profiles.update(overrides)
    monkeypatch.setattr(industry, "_profiles", lambda: profiles)
    return profiles


def test_same_segment_name_maps_to_different_templates_per_industry(monkeypatch):
    """两个行业都有「设备」赛道，各自指向不同的指标模板。"""
    generic = industry._profiles()["generic"]
    _fake_profiles(
        monkeypatch,
        alpha=replace(generic, segments=(("设备", "equipment"),)),
        beta=replace(generic, segments=(("设备", "cyclical"),)),
    )
    assert industry.type_for_segment("设备", "alpha") == "equipment"
    assert industry.type_for_segment("设备", "beta") == "cyclical"
    assert industry.segment_map_for("alpha") != industry.segment_map_for("beta")


def test_declared_segments_never_fall_back_to_global_map(monkeypatch):
    """声明了 segments 的行业：全局有同名映射也不借用，避免串味。"""
    assert "设计" in registry.segment_map(), "前提：全局字典有「设计」兜底映射"
    generic = industry._profiles()["generic"]
    _fake_profiles(
        monkeypatch,
        scoped=replace(generic, segments=(("本体", "manufacturing"),)),
    )
    assert industry.type_for_segment("本体", "scoped") == "manufacturing"
    assert industry.type_for_segment("设计", "scoped") == industry.GENERIC


def test_undeclared_industry_still_uses_global_fallback():
    """没声明 segments 的行业保持旧行为（无行业上下文的调用不会被破坏）。"""
    assert industry.segment_map_for("generic") == dict(registry.segment_map())
    assert industry.type_for_segment("设计") == industry.FABLESS


def test_profile_segments_must_point_to_existing_profile():
    """赛道类型写错要 fail-fast，不能等渲染时静默落 generic。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "industries.yaml"
        raw = yaml.safe_load((_CONFIG_DIR / "industries.yaml").read_text(encoding="utf-8"))
        raw["profiles"]["robotics"]["segments"]["本体"] = "not_a_profile"
        path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match="未知公司类型"):
            industry.validate_industries(path)


def test_snapshot_exports_industry_segments():
    """降级快照必须带上行业级赛道语义，否则主配置挂了又退化成全局字典。"""
    expected = classification_snapshot.render_defaults(
        _CONFIG_DIR / "indicators.yaml", _CONFIG_DIR / "industries.yaml"
    )
    assert classification_snapshot.DEFAULTS_PY.read_text(encoding="utf-8") == expected

    raw = yaml.safe_load((_CONFIG_DIR / "industries.yaml").read_text(encoding="utf-8"))
    shipped = {
        key: dict(entry["segments"])
        for key, entry in raw["profiles"].items()
        if entry.get("segments")
    }
    assert classification_defaults.INDUSTRY_SEGMENTS == shipped


def test_render_layer_passes_industry_key_to_segment_lookup():
    """静态护栏：渲染层解析赛道时必须带行业上下文，不许退回全局查询。"""
    source = (_REPORT_ROOT / "compose.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "type_for_segment"
    ]
    assert calls, "compose.py 应当按行业解析赛道类型"
    for node in calls:
        has_key = len(node.args) >= 2 or any(
            kw.arg == "industry_key" for kw in node.keywords
        )
        assert has_key, "type_for_segment 必须传入 industry_key，否则会串到别的行业模板"
