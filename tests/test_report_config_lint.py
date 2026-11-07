"""研报配置启动 lint 的契约测试。

核心两条：配置坏时默认**不阻断启动但必须显眼**（issues 非空 / ERROR 日志），
``strict=True`` 则硬 fail-fast；以及降级镜像、内置快照、renderer 绑定、
SKILL.md 声明各自漂移都能被单独查出来。
"""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ai_agent.core.config import settings
from ai_agent.report import lint as lint_module
from ai_agent.report import sections, skills
from ai_agent.report.lint import LintReport, ReportConfigError, lint_report_config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _PROJECT_ROOT / "src" / "ai_agent" / "report" / "config"
_CLI = _PROJECT_ROOT / "scripts" / "lint_report_config.py"

_CHECK_NAMES = {
    "indicators.yaml",
    "indicators_defaults.yaml",
    "snapshot_sync",
    "sections.yaml",
    "renderer_bindings",
    "builtin_aliases_sync",
    "sections_snapshot_sync",
    "classification_builtin_sync",
    "skill_frontmatter",
    "skill_section_ids",
}


@pytest.fixture
def config_dir():
    """复制出厂三件套到临时目录，避免污染真实配置。

    沙箱里 pytest 的 tmp_path 会撞 mkdir EEXIST，沿用项目既有做法用
    tempfile.TemporaryDirectory()。
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("indicators.yaml", "indicators_defaults.yaml", "sections.yaml"):
            shutil.copy(_CONFIG_DIR / name, root / name)
        yield root


def _edit(path: Path, mutate) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False, width=200),
        encoding="utf-8",
    )


def _load_cli():
    spec = importlib.util.spec_from_file_location("lint_report_config_cli", _CLI)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_shipped_config_passes_every_check():
    report = lint_report_config()
    assert report.ok
    assert report.issues == ()
    assert set(report.checked) == _CHECK_NAMES


def test_strict_defaults_to_false_in_settings():
    """生产默认不因一份 YAML 拼错让全站起不来。"""
    assert settings.report_config_strict is False


def test_broken_indicators_reports_issue_without_raising(config_dir):
    _edit(config_dir / "indicators.yaml", lambda raw: raw.update({"unexpected": 1}))
    report = lint_report_config(config_dir=config_dir)
    assert not report.ok
    assert any(issue.startswith("indicators.yaml:") for issue in report.issues)
    # 一项坏不应挡住其余检查
    assert "sections.yaml" in report.checked


def test_strict_mode_fails_fast(config_dir):
    _edit(config_dir / "indicators.yaml", lambda raw: raw.update({"unexpected": 1}))
    with pytest.raises(ReportConfigError, match="研报配置 lint 未通过"):
        lint_report_config(strict=True, config_dir=config_dir)


def test_snapshot_drift_is_detected(config_dir):
    """镜像合法但与主配置不同：只有 snapshot_sync 该报错。"""
    _edit(config_dir / "indicators_defaults.yaml", lambda raw: raw["ordering"].update({"lead_max": 999}))
    report = lint_report_config(config_dir=config_dir)
    names = {issue.split(":", 1)[0] for issue in report.issues}
    assert names == {"snapshot_sync"}


def test_unregistered_renderer_id_is_detected(config_dir):
    _edit(
        config_dir / "sections.yaml",
        lambda raw: raw["aliases"].update({"not_registered": ["全新标题"]}),
    )
    report = lint_report_config(config_dir=config_dir)
    assert any(issue.startswith("renderer_bindings:") for issue in report.issues)


def test_builtin_snapshot_drift_is_detected(monkeypatch):
    monkeypatch.setattr(sections, "_BUILTIN_ALIASES", sections._BUILTIN_ALIASES[:-1])
    report = lint_report_config()
    assert any(issue.startswith("builtin_aliases_sync:") for issue in report.issues)


def test_skill_declared_unknown_renderer_is_detected(monkeypatch):
    fake = SimpleNamespace(skill_id="fake_skill", section_ids=("unknown_renderer",))
    monkeypatch.setattr(skills, "SKILLS", (fake,))
    report = lint_report_config()
    assert any(issue.startswith("skill_section_ids:") for issue in report.issues)


def test_cli_returns_zero_for_shipped_config():
    assert _load_cli().main() == 0


def test_cli_returns_nonzero_for_broken_config(config_dir, monkeypatch):
    _edit(config_dir / "sections.yaml", lambda raw: raw.update({"version": 99}))
    monkeypatch.setattr(lint_module, "_CONFIG_DIR", config_dir)
    assert _load_cli().main() == 1


def test_lifespan_runs_lint_and_does_not_block_on_failure(monkeypatch):
    """坏配置时 lifespan 仍要能进入 yield；lint 以 strict=settings 的值调用。"""
    import ai_agent.api.app as app_module
    import ai_agent.kb.retrieval.rerank as rerank

    calls: list[object] = []
    broken = LintReport(checked=(), issues=("indicators.yaml: 故意损坏",))

    def fake_lint(*, strict: bool = False, **_kwargs):
        calls.append(strict)
        return broken

    monkeypatch.setattr(lint_module, "lint_report_config", fake_lint)
    monkeypatch.setattr(app_module, "run_migrations", lambda: None)
    monkeypatch.setattr(rerank, "probe_reranker", lambda: None)
    monkeypatch.setattr(
        app_module,
        "kb_pipeline",
        SimpleNamespace(recover_stuck=lambda: None, shutdown=lambda: None),
    )
    monkeypatch.setattr(
        app_module,
        "short_term_memory",
        SimpleNamespace(start_cleanup=lambda: None, stop_cleanup=lambda: None),
    )
    monkeypatch.setattr(
        app_module,
        "maintenance_cleaner",
        SimpleNamespace(start=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(app_module, "shutdown_title_jobs", lambda: None)
    monkeypatch.setattr(app_module, "shutdown_turn_finalizer", lambda: None)
    monkeypatch.setattr(app_module, "store", SimpleNamespace(close=lambda: None))

    fake_app = SimpleNamespace(state=SimpleNamespace())

    async def run():
        async with app_module.lifespan(fake_app):
            calls.append("yielded")

    asyncio.run(run())
    assert calls == [settings.report_config_strict, "yielded"]
    # lint 结果挂到 app.state，供 /health 与诊断端点读取
    assert fake_app.state.report_config_lint is broken


def test_sections_snapshot_freshness_is_detected(monkeypatch, tmp_path_fallback):
    """改了 sections.yaml 忘跑同步脚本 / 手工改了生成物，都要被拦住。"""
    from ai_agent.report import sections_snapshot

    monkeypatch.setattr(sections_snapshot, "DEFAULTS_PY", tmp_path_fallback / "stale.py")
    report = lint_report_config()
    assert any(issue.startswith("sections_snapshot_sync:") for issue in report.issues)


def test_classification_builtin_drift_is_detected(monkeypatch):
    from ai_agent.report import registry

    monkeypatch.setattr(registry, "_BUILTIN_NAME_HINTS", registry._BUILTIN_NAME_HINTS[:-1])
    report = lint_report_config()
    assert any(issue.startswith("classification_builtin_sync:") for issue in report.issues)


def test_broken_skill_frontmatter_is_detected(monkeypatch, tmp_path_fallback):
    """SKILL.md 缺必填字段：模块导入期就会炸，lint 要提前把它变成一条 issue。"""
    from ai_agent.report import skills

    bad = tmp_path_fallback / "broken_skill"
    bad.mkdir()
    (bad / "SKILL.md").write_text(
        "---\nname: broken_skill\nkind: company\ntitle: 坏模板\n---\n\n"
        "## 大纲\n\n### 本期结论\nsection: lead\n写点东西\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skills, "_SKILLS_ROOT", tmp_path_fallback)
    report = lint_report_config()
    assert any(issue.startswith("skill_frontmatter:") for issue in report.issues)


def test_sync_script_check_passes_for_shipped_config():
    spec = importlib.util.spec_from_file_location(
        "sync_sections_cli", _PROJECT_ROOT / "scripts" / "sync_sections_snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


def test_sync_script_check_fails_on_snapshot_drift(monkeypatch, tmp_path_fallback):
    spec = importlib.util.spec_from_file_location(
        "sync_sections_cli_check", _PROJECT_ROOT / "scripts" / "sync_sections_snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    stale = tmp_path_fallback / "stale.py"
    stale.write_text("# 手工改过\n", encoding="utf-8")
    # 脚本在 import 时把 DEFAULTS_PY 绑进自己的命名空间
    monkeypatch.setattr(module, "DEFAULTS_PY", stale)
    assert module.main(["--check"]) == 1


def test_render_snapshot_rejects_bad_yaml(tmp_path_fallback):
    from ai_agent.report import sections_snapshot

    bad = tmp_path_fallback / "bad.yaml"
    bad.write_text("version: 2\naliases: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="version 必须为 1"):
        sections_snapshot.render_snapshot(bad)


@pytest.fixture
def tmp_path_fallback():
    """沙箱里 pytest 的 tmp_path 会撞 mkdir EEXIST，沿用项目既有做法。"""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)
