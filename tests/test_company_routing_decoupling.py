"""公司研报解耦的架构护栏。

行业侧的解耦由 ``test_industry_decoupling.py`` / ``test_industry_segment_scoping.py``
守着；本文件守公司侧，验收三条「新增一个公司品类不用改 .py」的路径：

1. ``industry_of`` 靠 industries.yaml 的别名识别品类，不再写死「银行」；
2. 新写一个公司类 SKILL.md 能被自动路由，不需要调用方传 report_type；
3. 「本期结论」口径由 profile.lead 决定，新品类写配置就生效。

最后两条是防回潮的静态断言：渲染层不许再出现 ``skill_id ==`` 分支，
分类层不许再出现写死的行业名判断。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from ai_agent.report import compose, industry, lint, parse, skills
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REPORT_SRC = _PROJECT_ROOT / "src" / "ai_agent" / "report"

_COMPANY_SKILL = """---
name: insurance_financial_report
kind: company
title: 保险财报深度分析
industries: insurance
indicator_set: insurance
description: 保险业公司季报：保费收入、新业务价值、综合成本率等保险口径指标
---

# 保险财报深度分析

管「写什么」：大纲与默认指标集。

## 大纲

### 本期结论
本期结论只提炼主线。
"""

_INSURANCE_PROFILE = {
    "include": ["营业收入", "净利润"],
    "lead": ["净利润", "营业收入"],
    "lead_hint": ["保费收入"],
}


def _fact(name: str, value: str, subject: str = "某某保险股份有限公司") -> IndicatorFact:
    return IndicatorFact(
        subject=subject,
        period="2025Q3",
        name=name,
        value=value,
        unit="元",
        column="本报告期",
        scope="consolidated",
        evidence=Evidence(
            file_id="file-1",
            filename="某某保险2025年第三季度报告.pdf",
            page=3,
            page_end=3,
            section="主要会计数据",
            quote=f"{name} | {value}",
        ),
    )


@pytest.fixture
def insurance_config(monkeypatch):
    """配置层面新增一个「保险」品类：只加 profile + 中文别名，不动 Python。"""
    profiles = dict(industry._profiles())
    profiles["insurance"] = industry.IndicatorProfile(
        indicators=tuple(_INSURANCE_PROFILE["include"]),
        lead=tuple(_INSURANCE_PROFILE["lead"]),
        lead_hint=tuple(_INSURANCE_PROFILE["lead_hint"]),
    )
    aliases = dict(industry._aliases())
    aliases["保险"] = "insurance"

    monkeypatch.setattr(industry, "_profiles", lambda: profiles)
    monkeypatch.setattr(industry, "_aliases", lambda: aliases)
    yield profiles


@pytest.fixture
def insurance_skill(monkeypatch):
    """注册表层面新增一个公司类 skill：只加一个 SKILL.md 目录。

    临时目录里复制全部出厂 skill，保证「新增一个」是在完整注册表上叠加，
    而不是把注册表掏空——否则测不到「新品类不会挤掉既有品类」。
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for source in sorted(skills._SKILLS_ROOT.glob("*/SKILL.md")):
            target = root / source.parent.name
            target.mkdir()
            (target / "SKILL.md").write_text(
                source.read_text(encoding="utf-8"), encoding="utf-8"
            )
        skill_dir = root / "insurance_financial_report"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(_COMPANY_SKILL, encoding="utf-8")
        # 用独立的 monkeypatch context：退出时先还原 _SKILLS_ROOT，再 reload，
        # 否则缓存的临时注册表会跟着测试泄漏到后面（lint 会看到幽灵 skill）。
        with monkeypatch.context() as patch:
            patch.setattr(skills, "_SKILLS_ROOT", root)
            skills.reload_skills()
            yield skills.get_skill("insurance_financial_report")
        skills.reload_skills()


def test_industry_of_matches_configured_alias(insurance_config):
    """行业识别来自别名表：加一行「保险: insurance」就能认出来。"""
    assert industry.industry_of(["某某保险股份有限公司"]) == "insurance"
    assert industry.classify_subject("某某保险股份有限公司") == "insurance"
    # 没配别名的公司仍落 generic，不猜
    assert industry.industry_of(["晶科能源"]) == industry.GENERIC


def test_new_company_skill_routes_without_report_type(insurance_config, insurance_skill):
    """不传 report_type 也能命中新品类：这是原先「公司侧写死二分」的破点。"""
    assert insurance_skill is not None
    assert skills.resolve_skill("company", "保险").skill_id == (
        "insurance_financial_report"
    )
    # 声明英文 profile key 也一样能路由（别名归一后比较）
    assert skills.resolve_skill("company", "insurance").skill_id == (
        "insurance_financial_report"
    )
    # 别的品类不受影响
    assert skills.resolve_skill("company", "bank").skill_id == "bank_financial_report"
    assert skills.resolve_skill("company", "保险").kind == "company"


def test_parse_routes_new_company_category(insurance_config, insurance_skill, monkeypatch):
    """端到端：一句需求 → 新品类 skill，且不再写「没有专属模板」的缺口。"""
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda: [
            {
                "subject": "某某保险股份有限公司",
                "status": "ready",
                "period": "2025Q3",
                "file_id": "f1",
                "filename": "某某保险2025年第三季度报告.pdf",
                "subject_aliases": [],
            }
        ],
    )
    plan = parse.parse_request("生成某某保险股份有限公司2025Q3财报深度分析")
    assert plan.kind == "company"
    assert plan.skill_id == "insurance_financial_report"
    assert plan.company_type == "insurance"
    assert not any("没有专属报告模板" in gap for gap in plan.gaps)


def test_lead_priority_comes_from_profile_config(insurance_config):
    """新增品类的结论口径写在 profile.lead 里，渲染层不用加分支。"""
    assert industry.lead_priority("insurance") == ("净利润", "营业收入")
    assert industry.lead_priority("bank")  # 既有品类仍由档案提供
    assert industry.lead_priority("generic") == ()


def test_conclusion_follows_custom_profile_lead(insurance_config):
    """端到端出句顺序：配置写净利润在前，正文就按净利润在前。"""
    plan = ReportPlan(
        title="某某保险 2025Q3 财报深度分析",
        kind="company",
        deliverable="markdown",
        skill_id="",
        subjects=["某某保险股份有限公司"],
        company_type="insurance",
        period="2025Q3",
        dimensions=["同比"],
        indicators=["营业收入", "净利润"],
        outline=[OutlineItem("本期结论", "")],
    )
    facts = [_fact("营业收入", "100"), _fact("净利润", "200")]
    document = compose.compose_report(plan, facts)
    section = document.markdown.split("## 本期结论", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.strip()]
    assert lines, "本期结论不该为空"
    assert "净利润为 200" in lines[0], f"首句未按档案口径：{lines[0]!r}"


def test_lint_rejects_unroutable_company_skill(monkeypatch):
    """公司档案不可路由时必须变成显式 issue，而不是运行时静默落通用模板。"""
    fake = skills.ReportSkill(
        skill_id="insurance_financial_report",
        kind="company",
        name="保险财报深度分析",
        industries=("保险",),  # 别名表里没有「保险」→ 永远命中不了
        description="测试用",
        indicators=("营业收入",),
        sections=(("本期结论", "只提炼主线"),),
        indicator_set="generic",
    )
    real = skills._load_skills
    monkeypatch.setattr(
        skills, "_load_skills", lambda: (*real(), fake)
    )
    report = lint.lint_report_config()
    assert any(
        issue.startswith("company_skill_routing:") for issue in report.issues
    ), report.issues


@pytest.mark.parametrize(
    "filename,forbidden",
    [
        ("compose.py", "skill_id =="),
        ("compose.py", "lead_bank"),
        ("industry.py", '"银行" in'),
        ("parse.py", "lead_bank"),
    ],
)
def test_no_hardcoded_company_branches(filename: str, forbidden: str):
    """静态护栏：公司侧不许再写回行业专属分支。"""
    source = (_REPORT_SRC / filename).read_text(encoding="utf-8")
    assert forbidden not in source, f"{filename} 又出现了硬编码：{forbidden!r}"


def test_industries_config_has_no_lead_bank_key():
    """结论口径已迁到 profile.lead，indicators.yaml 不该再有第二份名单。"""
    raw = yaml.safe_load(
        (_REPORT_SRC / "config" / "indicators.yaml").read_text(encoding="utf-8")
    )
    assert "lead_bank" not in (raw.get("ordering") or {})
