"""新增行业 scaffold：验证「一条命令、零 .py 改动」确实能落地。

刻意把项目根指向临时目录，避免测试污染真实配置；写入后直接复用
industry / registry / skills 的正式解析入口，确认产物能被现有代码读懂。
"""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
from pathlib import Path

from ai_agent.report import industry, registry, skills

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _PROJECT_ROOT / "src" / "ai_agent" / "report" / "config"
_SCRIPT = _PROJECT_ROOT / "scripts" / "scaffold_industry.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("scaffold_industry_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _temp_root(root: Path) -> Path:
    config = root / "src" / "ai_agent" / "report" / "config"
    config.mkdir(parents=True)
    (root / "src" / "ai_agent" / "report" / "skills").mkdir(parents=True)
    for name in ("industries.yaml", "indicators.yaml"):
        shutil.copy(_CONFIG_DIR / name, config / name)
    return config


def _argv(root: Path) -> list[str]:
    return [
        "--name",
        "光伏",
        "--set",
        "photovoltaics",
        "--companies",
        "隆基绿能|601012|组件, 通威股份|600438|硅料",
        "--segment-types",
        "组件=manufacturing,硅料=cyclical",
        "--root",
        str(root),
    ]


def test_dry_run_writes_nothing():
    mod = _load_script()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _temp_root(root)
        assert mod.main(_argv(root)) == 0
        assert not (
            root / "src/ai_agent/report/skills/photovoltaics_industry_report"
        ).exists()
        before = (root / "src/ai_agent/report/config/industries.yaml").read_text(
            encoding="utf-8"
        )
        assert "photovoltaics" not in before


def test_company_mode_scaffold_produces_routable_category(monkeypatch):
    """新增公司品类：一条命令产出 SKILL.md + 别名 + profile，且能被别名路由到。"""
    mod = _load_script()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = _temp_root(root)
        argv = [
            "--kind",
            "company",
            "--name",
            "保险",
            "--set",
            "insurance",
            "--root",
            str(root),
            "--apply",
        ]
        assert mod.main(argv) == 0

        skill_md = (
            root / "src/ai_agent/report/skills/insurance_financial_report/SKILL.md"
        )
        assert skill_md.exists()
        skill = skills._load_skill(skill_md)
        assert skill.kind == "company"
        assert skill.indicator_set == "insurance"
        assert skill.industries == ("insurance",)
        assert not skill.companies  # 公司品类不带花名册

        text = (config / "industries.yaml").read_text(encoding="utf-8")
        assert "  保险: insurance" in text
        assert "  insurance:" in text

        with monkeypatch.context() as patch:
            patch.setattr(industry, "_INDUSTRIES_YAML", config / "industries.yaml")
            patch.setattr(registry, "_INDUSTRIES_YAML", config / "industries.yaml")
            patch.setattr(registry, "_INDICATORS_YAML", config / "indicators.yaml")
            industry._profiles.cache_clear()
            industry._aliases.cache_clear()
            registry.reload_registry()

            # 中文别名 → profile key，公司路由（industry_of）就能命中
            assert industry.canonical_industry("保险") == "insurance"
            assert industry._normalize_key("保险") == "insurance"
            assert "insurance" in industry.known_company_types()

        industry._profiles.cache_clear()
        industry._aliases.cache_clear()
        registry.reload_registry()


def test_company_mode_rejects_roster_flags():
    """公司品类不该带花名册：混着传说明把两类搞混了，直接报错。"""
    mod = _load_script()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _temp_root(root)
        argv = [
            "--kind",
            "company",
            "--name",
            "保险",
            "--set",
            "insurance",
            "--companies",
            "中国平安|601318",
            "--root",
            str(root),
        ]
        assert mod.main(argv) == 1


def test_apply_produces_config_that_existing_code_can_read(monkeypatch):
    mod = _load_script()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = _temp_root(root)
        assert mod.main(_argv(root) + ["--apply"]) == 0

        skill_md = (
            root
            / "src/ai_agent/report/skills/photovoltaics_industry_report/SKILL.md"
        )
        assert skill_md.exists()

        # 现有解析入口直接读懂新行业，不需要任何新的 Python 分支
        skill = skills._load_skill(skill_md)
        assert skill.kind == "industry"
        assert skill.indicator_set == "photovoltaics"
        assert skill.industries == ("光伏",)
        # 花名册带赛道 → 通用判定会自动走赛道分组路径（全局 SKILLS 未加载临时 skill，
        # 这里直接验证判定所依据的数据本身）
        assert any(segment for _n, _c, segment, _a in skill.companies)
        assert skill.segment_of("隆基绿能") == "组件"

        profiles = industry.validate_industries(config / "industries.yaml")
        assert "photovoltaics" in profiles
        assert registry.validate_registry(config / "indicators.yaml")

        with monkeypatch.context() as patch:
            patch.setattr(industry, "_INDUSTRIES_YAML", config / "industries.yaml")
            patch.setattr(registry, "_INDUSTRIES_YAML", config / "industries.yaml")
            patch.setattr(registry, "_INDICATORS_YAML", config / "indicators.yaml")
            industry._profiles.cache_clear()
            industry._aliases.cache_clear()
            registry.reload_registry()

            assert industry._normalize_key("光伏") == "photovoltaics"
            assert "photovoltaics" in industry.known_company_types()
            # 赛道语义属于这个行业自己的 profile.segments
            own = industry.segment_map_for("photovoltaics")
            assert own == {"组件": "manufacturing", "硅料": "cyclical"}
            assert industry.type_for_segment("组件", "photovoltaics") == "manufacturing"
            assert industry.type_for_segment("硅料", "photovoltaics") == "cyclical"
            # 且**没有污染全局字典**：无行业上下文时仍解析不到新赛道
            assert "组件" not in registry.segment_map()
            assert industry.type_for_segment("组件") == industry.GENERIC
            assert "营业收入" in industry.default_indicators("光伏")

        # 新行业的赛道语义不串到既有行业
        assert industry.type_for_segment("组件", "semiconductor") == industry.GENERIC

        industry._profiles.cache_clear()
        industry._aliases.cache_clear()
        registry.reload_registry()
