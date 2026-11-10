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
12. 行业档案完整性：indicator_set 指向已登记 profile、带赛道花名册的行业
    必须自己声明 segments 且覆盖全部赛道、定性文案不含数字
    （**验收「新增一个行业」**，并拦住同名赛道跨行业串模板）；
13. 公司档案可路由：公司类 skill 的 ``industries`` 必须是已登记的非 generic
    profile、必须有中文别名可被主体名命中、并且确实能被 ``resolve_skill``
    反查回自己（**验收「新增一个公司品类」**）；
14. 结论口径可落地：profile 的 ``lead`` / ``lead_hint`` 只能是已登记指标，
    且 ``lead`` 必须落在该 profile 的 ``include`` 里（否则结论句永远出不来）；
15. 赛道结构口径：带花名册的行业必须声明 ``structure``，且名单落在 include／
    已登记指标内；``structure_gloss`` 的 key 必须是 structure 里的指标。

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
        """新增行业完整性：只认行业自己的 segments，不看全局字典。

        这是「加一个行业只改配置」的验收门禁：带赛道花名册的行业若没声明
        segments，会回退到全局字典，同名赛道可能套上别的行业的指标模板。
        漏配必须变成一条显式 issue，不能静默串味。
        """
        from ai_agent.report import industry, skills

        known = industry.known_company_types()
        problems: list[str] = []
        for skill in skills._load_skills():
            if skill.kind != "industry":
                continue
            key = (skill.indicator_set or "").strip()
            if key and key not in known:
                problems.append(
                    f"{skill.skill_id}: indicator_set {key!r} 未登记 profile"
                )
            roster = {
                (segment or "").strip()
                for _name, _code, segment, _aka in skill.companies
                if (segment or "").strip()
            }
            if not roster:
                continue
            own = dict(industry.profile_of(key).segments)
            if not own:
                problems.append(
                    f"{skill.skill_id}: 花名册含赛道，但 profile {key!r} 未声明 "
                    "segments（会回退全局字典，可能串到别的行业模板）"
                )
                continue
            missing = sorted(roster - set(own))
            if missing:
                problems.append(
                    f"{skill.skill_id}: 赛道 {missing} 缺行业级 segments 映射"
                )
            if not industry.profile_of(key).structure:
                problems.append(
                    f"{skill.skill_id}: 花名册含赛道，但 profile {key!r} 未声明 "
                    "structure（经营解读不会默认毛利率，必须显式配置）"
                )
        if problems:
            raise ValueError("行业档案不完整：" + "；".join(sorted(set(problems))))

    if config_dir is None:
        run("industry_integrity", check_industry_integrity)

    def check_company_skill_routing() -> None:
        """新增一个公司品类的验收：声明了 industries 就必须真的能路由到。

        原先公司侧只有「名字含银行→bank」一条写死规则，新写一个
        ``industries: 保险`` 的公司 skill 永远不会被自动路由。现在路由靠
        industries.yaml 的别名，这里把三个前提都变成显式 issue：
        行业已登记（非 generic）、有中文别名可被主体名命中、反查命中自己。
        """
        from ai_agent.report import industry, skills

        tokens = {key for _token, key in industry._industry_tokens()}
        problems: list[str] = []
        for skill in skills._load_skills():
            if skill.kind != "company" or not skill.industries:
                continue
            for declared in skill.industries:
                key = industry.canonical_industry(declared)
                if not key:
                    problems.append(
                        f"{skill.skill_id}: industries {declared!r} 不是已登记的 profile"
                    )
                    continue
                if key == industry.GENERIC:
                    problems.append(
                        f"{skill.skill_id}: industries 指向 generic（通用模板不该声明行业）"
                    )
                    continue
                if key not in tokens:
                    problems.append(
                        f"{skill.skill_id}: profile {key!r} 缺中文别名，"
                        "主体名永远命中不了它（在 industries.yaml 的 aliases 补一行）"
                    )
                chosen = skills.resolve_skill("company", key)
                if chosen.skill_id != skill.skill_id:
                    problems.append(
                        f"{skill.skill_id}: 被 {chosen.skill_id} 抢先命中，"
                        "公司口径不会生效"
                    )
        if problems:
            raise ValueError("公司档案不可路由：" + "；".join(sorted(set(problems))))

    if config_dir is None:
        run("company_skill_routing", check_company_skill_routing)

    def check_profile_lead_names() -> None:
        """行业结论／结构口径必须能落到真实指标上，否则那句结论永远出不来。"""
        from ai_agent.report import industry

        problems: list[str] = []
        for key, profile in industry._profiles().items():
            for field in ("lead", "lead_hint", "structure"):
                for name in getattr(profile, field):
                    if registry.spec_of(name) is None:
                        problems.append(f"{key}.{field} 的 {name!r} 不是已登记指标")
            for field in ("lead", "structure"):
                missing = [
                    name
                    for name in getattr(profile, field)
                    if name not in profile.indicators
                ]
                if missing:
                    problems.append(f"{key}.{field} 含不在 include 里的指标 {missing}")
            gloss_keys = {name for name, _text in profile.notes.structure_gloss}
            unknown_gloss = sorted(gloss_keys - set(profile.structure))
            if unknown_gloss:
                problems.append(
                    f"{key}.notes.structure_gloss 含不在 structure 里的指标 "
                    f"{unknown_gloss}"
                )
        if problems:
            raise ValueError("结论口径不可落地：" + "；".join(sorted(set(problems))))

    if config_dir is None:
        run("profile_lead_names", check_profile_lead_names)

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
