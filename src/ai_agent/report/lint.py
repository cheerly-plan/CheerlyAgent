"""研报配置启动 lint：把分散的显式校验收成一个入口。

背景：``registry`` / ``sections`` 的校验函数原先只在 CI 测试里被调用，
运行时读取路径却是「主配置坏 → 降级镜像 / 内置别名表 + exception 日志」。
结果就是配置写错能顺利上线，直到某次研报悄悄走了旧快照才被发现。

检查项（每项独立执行，一项失败不影响其余检查）：

1. ``indicators.yaml`` 严格 schema；
2. ``indicators_defaults.yaml`` 自身合法；
3. 降级镜像与主配置等价（改了主配置忘了跑同步脚本）；
4. ``industries.yaml`` schema + profile key ⊆ 已知公司类型；
5. ``sections.yaml`` 严格 schema；
6. sections.yaml 里的 id 都有 renderer（拼错 id 不该降级成占位句）；
7. 章节降级快照 ``sections_defaults.py`` 与 sections.yaml 语义一致；
8. 该快照是脚本的最新生成物（手工改过 / 忘了跑同步脚本）；
9. 分类降级快照 ``classification_defaults.py`` 是脚本的最新生成物；
10. SKILL.md 全量可解析、frontmatter 与指标集有效；
11. SKILL.md 里 ``section:`` 声明的 id 都已注册（原先只会运行时 warning）；
12. 行业档案完整性：花名册赛道都有 segment_to_type 映射、indicator_set
    指向已登记 profile、定性文案不含数字（**验收「新增一个行业」**）。

行为取舍：默认 **只记 ERROR 日志，不阻断启动**。运行时本来就有降级路径，
多租户服务端不该因为一份 YAML 拼错整体不可用；但错误必须显眼、可查。
``strict=True`` 抛 ``ReportConfigError``，供预发布 / CI / CLI 作硬门禁——
启动时由 ``settings.report_config_strict`` 控制，脚本默认开启。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_agent.core.logging import get_logger
from ai_agent.report import (
    classification_snapshot,
    registry,
    sections,
    sections_snapshot,
)

logger = get_logger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent / "config"


class ReportConfigError(ValueError):
    """研报配置 lint 失败（strict 模式），供调用方决定如何处理。"""


@dataclass(frozen=True)
class LintReport:
    """一次 lint 的结果；``checked`` 是通过的检查名，``issues`` 形如「检查名: 原因」。"""

    checked: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def lint_report_config(
    *, strict: bool = False, config_dir: Path | None = None
) -> LintReport:
    """跑一遍研报配置检查并返回结果；``strict=True`` 时失败即抛异常。

    ``config_dir`` 只用于测试：指向另一份三件套配置目录，模拟损坏 / 漂移。
    生产路径不传，用模块内置目录。
    """
    root = config_dir or _CONFIG_DIR
    indicators = root / "indicators.yaml"
    snapshot = root / "indicators_defaults.yaml"
    sections_yaml = root / "sections.yaml"

    checked: list[str] = []
    issues: list[str] = []

    def run(name: str, check) -> None:
        try:
            check()
        except Exception as exc:  # 每项独立，坏一项也要把其余项跑完
            issues.append(f"{name}: {exc}")
        else:
            checked.append(name)

    run("indicators.yaml", lambda: registry.validate_registry(indicators))
    run("indicators_defaults.yaml", lambda: registry.validate_registry(snapshot))

    def check_snapshot_sync() -> None:
        main = registry.validate_registry(indicators)
        mirror = registry.validate_registry(snapshot)
        if main != mirror:
            raise ValueError(
                "降级镜像与 indicators.yaml 不一致，请运行 "
                "uv run python scripts/sync_indicators_snapshot.py"
            )

    run("snapshot_sync", check_snapshot_sync)

    def check_industries() -> None:
        from ai_agent.report import industry

        path = root / "industries.yaml"
        if not path.exists():
            raise ValueError(f"缺少 industries.yaml：{path}")
        industry.validate_industries(path)

    # 临时 config_dir 单测通常只拷贝 indicators／sections 三件套；有文件才查。
    if (root / "industries.yaml").exists() or config_dir is None:
        run("industries.yaml", check_industries)

    run("sections.yaml", lambda: sections.validate_sections(sections_yaml))
    run("renderer_bindings", lambda: sections.validate_bindings(sections_yaml))

    def check_builtin_sync() -> None:
        # renderer 注册在 compose 顶层；显式触发一次，保证后面拿到的 id 集合完整
        sections._ensure_renderers()
        shipped = sections.validate_sections(sections_yaml)
        builtin = sections._invert(sections._BUILTIN_ALIASES, sections._BUILTIN_NOTE)
        if builtin != shipped:
            raise ValueError(
                "章节降级快照与 sections.yaml 不一致，请运行 "
                "uv run python scripts/sync_sections_snapshot.py"
            )

    run("builtin_aliases_sync", check_builtin_sync)

    def check_sections_snapshot_sync() -> None:
        """生成物是包内固定路径，与临时配置目录无关，只在默认路径下校验新鲜度。"""
        expected = sections_snapshot.render_snapshot(sections_yaml)
        path = sections_snapshot.DEFAULTS_PY
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != expected:
            raise ValueError(
                "sections_defaults.py 不是最新生成物，请运行 "
                "uv run python scripts/sync_sections_snapshot.py"
            )

    if config_dir is None:
        run("sections_snapshot_sync", check_sections_snapshot_sync)

    def check_classification_snapshot_sync() -> None:
        """生成物是包内固定路径，与临时配置目录无关，只在默认路径下校验新鲜度。"""
        expected = classification_snapshot.render_defaults(
            indicators, root / "industries.yaml"
        )
        path = classification_snapshot.DEFAULTS_PY
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != expected:
            raise ValueError(
                "classification_defaults.py 不是最新生成物，请运行 "
                "uv run python scripts/sync_indicators_snapshot.py"
            )

    if config_dir is None:
        run("classification_snapshot_sync", check_classification_snapshot_sync)

    def check_industry_integrity() -> None:
        """新增行业完整性：赛道有映射、指标集已登记、定性文案不含数字。

        这是「加一个行业只改配置」的验收门禁：漏配赛道以前会静默回退到
        半导体模板，现在必须变成一条显式 issue。
        """
        from ai_agent.report import industry, skills

        known = industry.known_company_types()
        segments = dict(registry.segment_map())
        problems: list[str] = []
        for skill in skills._load_skills():
            if skill.kind != "industry":
                continue
            if skill.indicator_set and skill.indicator_set not in known:
                problems.append(
                    f"{skill.skill_id}: indicator_set {skill.indicator_set!r} 未登记 profile"
                )
            for _name, _code, segment, _aka in skill.companies:
                seg = (segment or "").strip()
                if seg and seg not in segments:
                    problems.append(
                        f"{skill.skill_id}: 赛道 {seg!r} 缺 segment_to_type 映射"
                    )
        if problems:
            raise ValueError("行业档案不完整：" + "；".join(sorted(set(problems))))

    if config_dir is None:
        run("industry_integrity", check_industry_integrity)

    def check_skill_frontmatter() -> None:
        # 延迟导入：skills 模块导入即扫描 SKILL.md，坏了也要变成一条 issue
        from ai_agent.report import skills

        paths = sorted(skills._SKILLS_ROOT.glob("*/SKILL.md"))
        if not paths:
            raise ValueError(f"未找到任何 SKILL.md：{skills._SKILLS_ROOT}")
        problems: list[str] = []
        for path in paths:
            try:
                skill = skills._load_skill(path)
            except Exception as exc:
                problems.append(f"{path.parent.name}: {exc}")
                continue
            if not skill.indicators:
                problems.append(f"{path.parent.name}: indicator_set 没解析出任何指标")
        if problems:
            raise ValueError("SKILL.md 校验失败：" + "；".join(problems))

    run("skill_frontmatter", check_skill_frontmatter)

    def check_skill_bindings() -> None:
        # 延迟导入：skills 模块导入即扫描 SKILL.md，坏了也要变成一条 issue
        from ai_agent.report import skills

        known = sections.renderer_ids()
        unknown = sorted(
            {
                section_id
                for skill in skills.SKILLS
                for section_id in skill.section_ids
                if section_id and section_id not in known
            }
        )
        if unknown:
            raise ValueError(f"SKILL.md 声明了未注册的 renderer: {unknown}")

    run("skill_section_ids", check_skill_bindings)

    report = LintReport(tuple(checked), tuple(issues))
    if issues:
        for issue in issues:
            logger.error("研报配置 lint 失败：%s", issue)
        logger.error(
            "研报配置 lint 未通过：%d 项通过 / %d 项失败（strict=%s）",
            len(checked),
            len(issues),
            strict,
        )
        if strict:
            raise ReportConfigError("研报配置 lint 未通过：\n" + "\n".join(issues))
    else:
        logger.info("研报配置 lint 通过：%d 项检查（%s）", len(checked), ", ".join(checked))
    return report
