"""章节注册表 conformance：新增章节不改 compose.py，配置不能注入数字或逻辑。

这些用例守住三条边界：
1. 现有 21 个章节标题的分派结果不变（行为逐字不变由 test_report_* 兜底）；
2. SKILL.md 里写 ``section: <id>`` 或往 sections.yaml 加一行别名，
   新增章节就能出正文，不再静默写「这一节没有对应的数据，未写。」；
3. 配置只能选规则，非法结构 / 注入数字 / 拼错 id 一律能被显式拦住。
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import yaml

from ai_agent.report import sections, skills as skills_mod
from ai_agent.report.compose import compose_report
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan, _outline

_SHIPPED_YAML = Path(sections.__file__).resolve().parent / "config" / "sections.yaml"

_LEGACY_TITLES = (
    "阅读说明",
    "本期结论",
    "覆盖范围",
    "行业概览",
    "主要指标与出处",
    "同比与环比",
    "单季节奏",
    "较上年末变动",
    "数据观察",
    "核心主线",
    "经营解读",
    "赛道分组",
    "组内对比",
    "趋势与对比图",
    "下期预告",
    "景气周期",
    "细分赛道",
    "重点公司",
    "趋势",
    "公司间对比",
    "缺失说明",
    "缺失与口径",
    "风险与缺失",
)


def _base_config() -> dict:
    return yaml.safe_load(_SHIPPED_YAML.read_text(encoding="utf-8"))


def _fact(period: str = "2025Q3", value: str = "100,924,954") -> IndicatorFact:
    return IndicatorFact(
        subject="中国能源建设股份有限公司",
        period=period,
        name="营业收入",
        value=value,
        unit="元",
        column="本报告期",
        scope="consolidated",
        evidence=Evidence(
            file_id="file-1",
            filename="中国能源建设2025年第三季度报告.pdf",
            page=12,
            page_end=12,
            section="主要会计数据",
            quote="营业收入 | " + value,
        ),
    )


def _plan(outline: list[OutlineItem]) -> ReportPlan:
    return ReportPlan(
        title="中国能源建设股份有限公司 2025Q3 财报深度分析",
        kind="company",
        deliverable="markdown",
        subjects=["中国能源建设股份有限公司"],
        period="2025Q3",
        indicators=["营业收入", "毛利率"],
        outline=outline,
    )


@pytest.fixture
def config_dir():
    # 项目现有测试在沙箱里用 tmp_path 会撞 mkdir EEXIST，沿用 tempfile 的既有做法。
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


@pytest.fixture
def use_config(config_dir, monkeypatch):
    """把注册表指向临时配置；测试结束先恢复路径再清缓存。"""

    def _use(config: dict) -> Path:
        path = config_dir / "sections.yaml"
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        monkeypatch.setattr(sections, "_SECTIONS_YAML", path)
        sections.reload_sections()
        return path

    yield _use
    monkeypatch.undo()
    sections.reload_sections()


# ——— 现有行为不变 ———


def test_every_legacy_title_resolves_to_a_registered_renderer():
    for title in _LEGACY_TITLES:
        resolved = sections.resolve(title)
        assert resolved is not None, f"{title} 丢了 renderer"
        assert resolved[0] in sections.renderer_ids()


def test_shipped_config_bindings_are_complete():
    sections.validate_bindings()
    config = sections.validate_sections()
    assert config.ids <= sections.renderer_ids()
    assert len(config.by_title) >= len(_LEGACY_TITLES)


def test_unmapped_section_keeps_placeholder():
    document = compose_report(_plan([OutlineItem("全新章节", "")]), [_fact()])
    assert "## 全新章节" in document.markdown
    assert "这一节没有对应的数据，未写。" in document.markdown


# ——— 新增章节不再需要改 compose.py ———


def test_new_skill_section_with_id_renders_without_python_change(config_dir):
    """新增报告类型 = 加一个 SKILL.md 目录；章节自带 id，compose 零改动。"""
    skill_dir = config_dir / "new_energy_report"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: new_energy_report
kind: company
title: 新能源财报深度分析
description: 测试用报告类型
indicator_set: generic
---

## 大纲

### 全新行业章节
section: trend
按 {indicators} 写清楚本期趋势，不推断没有入库的数字。
""",
        encoding="utf-8",
    )
    skill = skills_mod._load_skill(skill_md)
    assert skill.sections == (("全新行业章节", "按 {indicators} 写清楚本期趋势，不推断没有入库的数字。"),)
    assert skill.section_ids == ("trend",)
    built = skill.build_outline(["营业收入"])
    assert built[0][1] == "按 营业收入 写清楚本期趋势，不推断没有入库的数字。"
    assert built[0][2] == "trend"

    plan = _plan([])
    outline = _outline(plan, skill)
    assert outline[0].section_id == "trend"

    document = compose_report(_plan(outline), [_fact(), _fact("2024Q3", "111,000,000")])
    assert "## 全新行业章节" in document.markdown
    assert "这一节没有对应的数据，未写。" not in document.markdown


def test_declared_section_id_overrides_title_alias():
    assert sections.resolve("本期结论", "trend")[0] == "trend"
    assert sections.resolve("本期结论")[0] == "lead"
    # 同一 id 但标题完全陌生，输出应与按别名分派到该 id 时一致。
    by_id = compose_report(
        _plan([OutlineItem("任意新标题", "", section_id="trend")]),
        [_fact(), _fact("2024Q3", "111,000,000")],
    )
    assert "这一节没有对应的数据，未写。" not in by_id.markdown


def test_alias_added_in_yaml_takes_effect(use_config):
    config = copy.deepcopy(_base_config())
    config["aliases"]["lead"].append("新能源景气概览")
    use_config(config)
    assert sections.resolve("新能源景气概览")[0] == "lead"
    document = compose_report(_plan([OutlineItem("新能源景气概览", "")]), [_fact()])
    assert "这一节没有对应的数据，未写。" not in document.markdown


def test_fallback_note_comes_from_config(use_config):
    """占位文案可配置，但只能是纯文案——不能塞数字或模板占位。"""
    config = copy.deepcopy(_base_config())
    config["fallback"]["note"] = "本节暂无可核对数据。"
    use_config(config)
    assert sections.fallback_lines() == ["本节暂无可核对数据。"]
    document = compose_report(_plan([OutlineItem("全新章节", "")]), [_fact()])
    assert "本节暂无可核对数据。" in document.markdown


# ——— 配置不能绕过门禁 ———


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update({"renderer": "lambda x: x"}),
        lambda c: c.update({"version": 2}),
        lambda c: c["aliases"].update({"Bad-Id": ["坏标题"]}),
        lambda c: c["aliases"]["trend"].append("本期结论"),
        lambda c: c["fallback"].update({"note": "缺 3 项数据。"}),
        lambda c: c["fallback"].update({"note": "看 {indicators}"}),
        lambda c: c["fallback"].update({"lines": ["不许注入正文"]}),
        lambda c: c["aliases"].update({"ghost": 123}),
        lambda c: c["fallback"].update({"note": "第一行\n第二行"}),
    ],
    ids=[
        "unknown-top-field",
        "wrong-version",
        "illegal-id",
        "duplicate-title",
        "note-with-digit",
        "note-with-template",
        "unknown-fallback-field",
        "aliases-not-list",
        "multiline-note",
    ],
)
def test_invalid_config_fails_fast(mutate, config_dir):
    config = copy.deepcopy(_base_config())
    mutate(config)
    path = config_dir / "bad.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        sections.validate_sections(path)


def test_unregistered_id_degrades_with_warning(use_config, monkeypatch):
    config = copy.deepcopy(_base_config())
    config["aliases"]["ghost"] = ["幽灵章节"]
    path = use_config(config)
    warnings: list[str] = []
    monkeypatch.setattr(
        sections.logger,
        "warning",
        lambda text, *args: warnings.append(text % args if args else text),
    )
    assert sections.resolve("幽灵章节") is None
    assert any("未注册" in text for text in warnings)
    with pytest.raises(ValueError, match="未注册"):
        sections.validate_bindings(path)


def test_missing_config_degrades_to_builtin_aliases(monkeypatch, config_dir):
    monkeypatch.setattr(sections, "_SECTIONS_YAML", config_dir / "missing.yaml")
    sections.reload_sections()
    assert sections.resolve("趋势")[0] == "trend"
    assert sections.fallback_lines() == ["这一节没有对应的数据，未写。"]
    monkeypatch.undo()
    sections.reload_sections()


def test_duplicate_or_illegal_registration_fails():
    before = sections.renderer_ids()
    with pytest.raises(ValueError, match="重复注册"):
        sections.register("lead")(lambda ctx: [])
    with pytest.raises(ValueError, match="非法 section_id"):
        sections.register("Bad-Id")(lambda ctx: [])
    assert {"lead", "trend", "gaps"} <= before
    assert sections.renderer_ids() == before
