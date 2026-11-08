"""深度分析层编排（Gate 0-5），独立于 ``report/pipeline.py``。

**Phase 1 边界（重要）**：本包默认不启用、不挂进 ``run_report``、不修改事实层
任何一行代码。调用方显式 ``run_deep_analysis(...)`` 才执行；``enabled=False``
且未注入补全函数时只产出零 token 的骨架（台账 + 候选主线 + 缺口 + checklist）。
这样 885 个既有用例、防幻觉门禁、PDF 定稿流程完全不受影响。

流程对应 investment-deep-analysis v2.0 的 Gate：

    Gate 0 框架加载 → Gate 1 场景识别 → 第1章 候选主线（假设，规则化）
    → 第2-8章 逐章验证 + 专家评审 → 第9章 收敛定稿 → 第10章 checklist
    → Gate 4 执行报告 → Gate 5 跨章一致性 + 候选vs终稿差异 → 跟踪与证伪
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.report.deep.analyzer import (
    BudgetExceeded,
    analyze_chapter,
    assess_convergence,
    converge,
    default_complete,
    review_chapter,
)
from ai_agent.report.deep.chapters import (
    CONTROL_GROUP_NOTE,
    DEFAULT_EXPERTS,
    EXPERT_NAMES,
    MAINLINE_RULES,
    chapters_for,
    spec_of,
)
from ai_agent.report.deep.ledger import SourceLedger
from ai_agent.report.deep.schema import (
    ChapterResult,
    ChecklistItem,
    Convergence,
    DataGap,
    DeepAnalysisResult,
    ExecutionReport,
    Falsifier,
    Mainline,
    QUALITY_LABELS,
    Scenario,
    TrackingItem,
)

logger = get_logger(__name__)


@dataclass
class DeepConfig:
    """深度分析运行参数。所有成本相关项都有默认值，可按需调。"""

    enabled: bool = False
    chapters: tuple[int, ...] | None = None
    # 全量 12 专家约 4 倍成本；默认 3 个高性价比视角
    full_panel: bool = False
    experts: tuple[str, ...] = DEFAULT_EXPERTS
    # 评审 = N 轮 + 1 次一致性/终审合并轮。技能原协议是 5 轮，这里默认 2+1
    review_rounds: int = 2
    max_claims_per_chapter: int = 6
    max_review_opinions: int = 4
    max_tokens: int = 1600
    temperature: float = 0.1
    timeout: float = 90.0
    # 章节分析与专家评审的业务模型名；空 = 跟随 SMALL_MODEL + 本地网关。
    model: str = ""
    # 决策 ④：第 9 章收敛与其终审走主模型。空 = settings.model（主网关）。
    # 想退回全流程小模型，显式填小模型名；填别的值可做单章 A/B。
    convergence_model: str = ""
    # 硬预算闸：超了就把剩余章节标 skipped，不静默缩水
    max_llm_calls: int = 60

    def active_experts(self) -> tuple[str, ...]:
        return EXPERT_NAMES if self.full_panel else self.experts


@dataclass(frozen=True)
class ModelPlan:
    """两条补全通道实际使用的模型。纯数据，不发起任何调用。"""

    chapter_model: str = ""  # 传给 lightweight_llm 的业务模型名；"" = 跟随 SMALL_MODEL
    convergence_model: str = ""  # 主模型名 → lightweight_llm 走主网关
    chapter_label: str = ""  # 审计展示用（解析后的真实模型名）
    convergence_label: str = ""
    route: str = "none"
    warnings: tuple[str, ...] = ()


def resolve_model_plan(
    cfg: DeepConfig,
    *,
    small_model: str = "",
    primary_model: str = "",
    injected: bool = False,
) -> ModelPlan:
    """解析模型与网关路由。纯函数，便于零成本单测；不碰 settings、不调 LLM。

    已核对 ``llm/client.py:55-68`` 的既有语义：``lightweight_llm("")`` 在
    SMALL_MODEL 已配置时走小模型 + 本地网关，**未配置时静默退回主模型 + 主网关**；
    传入与 SMALL_MODEL 不同的显式模型名则走该模型 + 主网关。因此「第 9 章用主模型」
    = 把 ``settings.model`` 作为业务模型名传入；而 SMALL_MODEL 缺失必须显式告警，
    不能让决策 ④ 悄悄变成全流程主模型（成本差一个量级）。
    """
    if injected:
        return ModelPlan(route="injected", chapter_label="外部注入", convergence_label="外部注入")

    small = (small_model or "").strip()
    primary = (primary_model or "").strip()
    chapter_override = (cfg.model or "").strip()
    convergence_override = (cfg.convergence_model or "").strip()
    warnings: list[str] = []

    if chapter_override:
        chapter_label, chapter_model = chapter_override, chapter_override
    elif small:
        chapter_label, chapter_model = small, ""  # 交给 lightweight_llm 走本地网关
    else:
        chapter_label, chapter_model = (primary or "未配置"), ""
        warnings.append(
            "SMALL_MODEL 未配置：章节与评审按 lightweight_llm 既有语义退回主模型 + 主网关，"
            "成本高于决策 ④ 的预期，请检查 .env"
        )

    if convergence_override:
        convergence_label, convergence_model = convergence_override, convergence_override
    else:
        convergence_label, convergence_model = (primary or "未配置"), primary
        if not primary:
            warnings.append("MODEL 未配置：第 9 章收敛无法显式指定主模型")

    # route 描述的是**实际生效**的两条通道，而不是配置写法：
    # 显式把收敛模型设回小模型名时，应识别为 all-small 而非笼统的 override。
    if small and convergence_label == small and chapter_label == small:
        route = "all-small"
    elif chapter_override or convergence_override:
        route = "override"
    elif not small:
        route = "all-primary"
    else:
        route = "small+primary"
    return ModelPlan(
        chapter_model=chapter_model,
        convergence_model=convergence_model,
        chapter_label=chapter_label,
        convergence_label=convergence_label,
        route=route,
        warnings=tuple(warnings),
    )


class _Budget:
    """调用计数与 token 粗估。项目当前未接 usage 统计，这里按字符 /1.5 估算。"""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0
        self.input_chars = 0
        self.output_chars = 0

    def wrap(self, fn):
        def call(system: str, user: str) -> str:
            if self.calls >= self.limit:
                raise BudgetExceeded(f"LLM 调用达到上限 {self.limit}")
            self.calls += 1
            self.input_chars += len(system) + len(user)
            raw = fn(system, user)
            self.output_chars += len(raw or "")
            return raw or ""
        return call

    @property
    def input_tokens(self) -> int:
        return int(self.input_chars / 1.5)

    @property
    def output_tokens(self) -> int:
        return int(self.output_chars / 1.5)


def generate_mainlines(ledger: SourceLedger, mode: str) -> tuple[list[Mainline], str]:
    """第 1 章：五规则扫描，确定性生成候选主线（假设），不调用 LLM。"""
    hits: dict[str, Any] = {}
    for rule_name, keywords, description in MAINLINE_RULES:
        for record in ledger.records:
            if any(keyword in record.item for keyword in keywords):
                hits[rule_name] = (description, record)
                break
    mainlines = [
        Mainline(
            business=record.subject or record.item,
            rule=rule_name,
            chain=(
                f"{record.item}={record.value}{record.unit} ← {description}；"
                "更深层链条待第 9 章收敛"
            ),
            tier=record.tier,
        )
        for rule_name, (description, record) in hits.items()
    ]
    if mode == "industry":
        company_type = f"行业横比（多主体对照组）；{CONTROL_GROUP_NOTE}"
    elif len({record.subject for record in ledger.records}) > 1:
        company_type = "多业务平台型"
    else:
        company_type = "单一主线型（待验证）"
    return mainlines, company_type


def _prior_text(chapters: list[ChapterResult]) -> str:
    parts = []
    for chapter in chapters:
        if chapter.status in {"skipped", "fallback"}:
            continue
        line = f"第{chapter.number}章 {chapter.title}：{chapter.verdict or '未形成结论'}"
        known = [claim.text for claim in chapter.claims if claim.claim_type == "known"]
        if known:
            line += "；已知：" + "；".join(known[:2])
        parts.append(line)
    return "\n".join(parts) if parts else "（无前序章节）"


def _checklist(result: DeepAnalysisResult, *, use_llm: bool) -> list[ChecklistItem]:
    """第 10 章：确定性规则，不给买卖建议，只做研究质量自检。"""
    chapters = {chapter.number: chapter for chapter in result.chapters}
    reviewed = [c for c in result.chapters if c.number not in {1, 10} and c.status == "reviewed"]
    items = [
        ChecklistItem(
            "财报可信度已完成评级",
            "✅" if 2 in chapters and chapters[2].verdict else "⚠️",
            chapters[2].verdict if 2 in chapters else "本模式未启用第 2 章",
        ),
        ChecklistItem(
            "每章均有数据来源表（可溯源）",
            "✅" if all(c.sources for c in result.chapters if c.status != "skipped") else "❌",
            "Tier 4 及以下不得进来源表",
        ),
        ChecklistItem(
            "数据缺口已显式声明",
            "✅" if result.all_gaps else "⚠️",
            f"共 {len(result.all_gaps)} 项；缺口不得臆测填补",
        ),
        ChecklistItem(
            "关键判断配了可证伪信号",
            "✅" if result.falsifiers else "❌",
            "没有跟踪指标的判断 = 不可验证的断言",
        ),
        ChecklistItem(
            "候选主线被诚实修正/推翻检查",
            "✅" if result.convergence and result.convergence.delta else "⚠️",
            "防止候选变预设立场",
        ),
        ChecklistItem(
            "第 9 章收敛未停在断言层",
            "✅" if result.execution.convergence_quality == "solid"
            else ("⚠️" if result.execution.convergence_quality else "—"),
            QUALITY_LABELS.get(result.execution.convergence_quality, "未执行收敛"),
        ),
        ChecklistItem(
            "分析结论与事实分层",
            "✅",
            "已知/推断/不确定三分 + Tier + evidence_ids",
        ),
        ChecklistItem(
            "章节评审完整",
            "✅" if reviewed and all(c.review for c in reviewed) else "⚠️",
            f"{len(reviewed)} 章完成评审" if use_llm else "未启用 LLM，本项为骨架自检",
        ),
        ChecklistItem(
            "估值与买卖建议",
            "✅",
            "按用户偏好不做估值，只给投资逻辑与需跟踪的经营变量",
        ),
    ]
    return items


def _cross_checks(result: DeepAnalysisResult) -> list[str]:
    """Gate 5：跨章交叉验证 + 候选主线 vs 最终链条差异审查。"""
    notes: list[str] = []
    by_number = {chapter.number: chapter for chapter in result.chapters}
    missing_sources = [
        chapter.number for chapter in result.chapters
        if chapter.status not in {"skipped", "fallback"} and not chapter.sources
    ]
    notes.append(
        f"数据来源表覆盖：{'全章齐全' if not missing_sources else '第 ' + ','.join(map(str, missing_sources)) + ' 章缺来源表'}"
    )
    roe = by_number.get(4)
    moat = result.convergence.moat_rating if result.convergence else ""
    if roe and roe.verdict and "高" in roe.verdict:
        notes.append(
            "ROE 高位与护城河交叉印证：" + ("已给出护城河评级 " + moat if moat else "缺少对应护城河来源，需回查第 4/9 章")
        )
    else:
        notes.append("ROE 与护城河交叉印证：本期未触发高 ROE 条件")
    notes.append(
        "候选主线 vs 最终链条：" + (
            result.convergence.delta if result.convergence and result.convergence.delta
            else "未记录差异，视为未收敛"
        )
    )
    notes.append(
        f"数据缺口在结论中声明：{len(result.all_gaps)} 项"
        + ("；含缺口的推断须标低置信度" if result.all_gaps else "")
    )
    notes.append(
        "跟踪指标覆盖：" + (
            f"{len(result.tracking)} 条关键判断已配先行指标"
            if result.tracking else "无跟踪指标，相关判断不可验证，风险更高"
        )
    )
    inconsistent = [chapter.number for chapter in result.chapters if chapter.status == "fallback"]
    if inconsistent:
        notes.append(f"第 {','.join(map(str, inconsistent))} 章降级，结论链不完整")
    quality = result.execution.convergence_quality
    if quality:
        notes.append(
            "第 9 章收敛质量门禁：" + QUALITY_LABELS.get(quality, quality)
            + "（" + "；".join(result.execution.convergence_checks) + "）"
        )
    else:
        notes.append("第 9 章收敛质量门禁：未执行（零 token 骨架或收敛未产出）")
    return notes


def run_deep_analysis(
    target: str,
    period: str,
    facts: list,
    *,
    mode: str = "company",
    config: DeepConfig | None = None,
    complete=None,
    convergence_complete=None,
    external_sources: list[dict] | None = None,
) -> DeepAnalysisResult:
    """跑一次深度分析，产出结构化结论对象。

    ``facts`` 直接接 ``ReportResult.facts``（list[IndicatorFact]）；
    ``complete`` / ``convergence_complete`` 可注入 ``(system, user) -> str``，
    分别对应章节通道与第 9 章收敛通道，便于零成本单测与单章 A/B。
    ``external_sources`` 每项传给 ``SourceLedger.add_external``，用于接 L2-L3 数据源。

    决策 ④：章节分析与专家评审走 SMALL_MODEL，第 9 章收敛**及其终审**走主模型；
    实际模型名与路由写进 ExecutionReport。默认（``enabled=False`` 且未注入 complete）
    只产出零 token 骨架，且完全不读取模型配置。
    """
    cfg = config or DeepConfig()
    if mode not in {"company", "industry"}:
        raise ValueError("mode 只能是 company 或 industry")
    ledger = SourceLedger.from_facts(list(facts or []))
    for item in external_sources or []:
        ledger.add_external(**item)
    if mode == "industry":
        ledger.gaps.append(
            DataGap(
                item="行业产能/份额/价格战等格局数据（L3 原始统计）",
                chapters=[6, 9],
                impact="第 6 章供给结构与对照组验证无法量化",
                handling="接入行业协会/统计局原始发布，L4 二手数据须回溯",
            )
        )

    chapter_numbers = list(cfg.chapters or chapters_for(mode))
    enabled_set = set(chapter_numbers)
    skipped = [n for n in range(1, 11) if n not in enabled_set]
    mainlines, company_type = generate_mainlines(ledger, mode)
    result = DeepAnalysisResult(
        target=target, period=period, mode=mode,
        mainlines=mainlines, company_type=company_type,
        ledger_gaps=list(ledger.gaps),
    )

    uses_stub = complete is not None or convergence_complete is not None
    use_llm = cfg.enabled or uses_stub
    budget = _Budget(cfg.max_llm_calls)
    if not use_llm:
        plan = ModelPlan()  # route="none"：零 token 骨架，不读取任何模型配置
        call = converge_call = None
    elif uses_stub:
        plan = ModelPlan(route="injected", chapter_label="外部注入", convergence_label="外部注入")
        primary_stub = complete if complete is not None else convergence_complete
        call = budget.wrap(primary_stub)
        converge_call = budget.wrap(
            convergence_complete if convergence_complete is not None else primary_stub
        )
    else:
        plan = resolve_model_plan(
            cfg, small_model=settings.small_model, primary_model=settings.model
        )
        call = budget.wrap(default_complete(
            cfg.max_tokens, cfg.temperature, cfg.timeout, plan.chapter_model
        ))
        converge_call = budget.wrap(default_complete(
            cfg.max_tokens, cfg.temperature, cfg.timeout, plan.convergence_model
        ))
    for warning in plan.warnings:
        logger.warning("深度分析模型路由告警：%s", warning)

    notes = [
        "第 1 章候选主线由五规则确定性生成（零 LLM 调用），性质为假设",
        "第 10 章 checklist 由确定性规则生成（零 LLM 调用）",
        f"评审协议：{len(cfg.active_experts())} 视角 × {cfg.review_rounds} 轮 + 1 次一致性/终审合并轮"
        + ("（全量 12 视角）" if cfg.full_panel else "（非 12专家×5轮全量，成本折中）"),
        "token 为字符 /1.5 粗估，项目当前未接 LLM usage 统计",
    ]
    if use_llm:
        notes.append(
            f"模型路由：{plan.route}；章节/评审={plan.chapter_label or '—'}，"
            f"第 9 章收敛与终审={plan.convergence_label or '—'}"
            "（决策 ④：仅收敛用主模型）"
        )
        notes.extend(plan.warnings)

    chapters: list[ChapterResult] = []
    budget_hit = False
    # 第 2-8 章：逐章验证 + 评审（第 1/9/10 章单独处理）
    for number in [n for n in chapter_numbers if 2 <= n <= 8]:
        spec = spec_of(number)
        if call is None or budget_hit:
            chapters.append(
                ChapterResult(number=number, title=spec.title, question=spec.question,
                              method="、".join(spec.methods), status="skipped")
            )
            continue
        try:
            chapter = analyze_chapter(
                spec, ledger, target=target, period=period, mode=mode,
                prior=_prior_text(chapters), complete=call,
                max_claims=cfg.max_claims_per_chapter,
            )
            chapter.review = review_chapter(
                chapter, ledger, experts=cfg.active_experts(), rounds=cfg.review_rounds,
                prior=_prior_text(chapters), complete=call,
                max_opinions=cfg.max_review_opinions,
            )
            if chapter.review and chapter.status != "fallback":
                chapter.status = "reviewed"
            chapters.append(chapter)
        except BudgetExceeded as exc:
            budget_hit = True
            notes.append(f"预算闸触发：{exc}；后续章节标 skipped")
            chapters.append(
                ChapterResult(number=number, title=spec.title, question=spec.question,
                              method="、".join(spec.methods), status="skipped")
            )

    # 第 9 章：收敛（候选主线裁决 + 护城河 + 证伪 + 三情景）
    convergence: Convergence | None = None
    tracking: list[TrackingItem] = []
    falsifiers: list[Falsifier] = []
    scenarios: list[Scenario] = []
    if 9 in enabled_set and converge_call is not None and not budget_hit:
        spec = spec_of(9)
        mainlines_text = "\n".join(
            f"- {item.business}（{item.rule}）：{item.chain}｜{item.tier}｜{item.status}"
            for item in mainlines
        ) or "（无候选主线）"
        try:
            convergence, tracking, scenario_rows, falsifier_rows = converge(
                target=target, period=period, mainlines_text=mainlines_text,
                prior=_prior_text(chapters), ledger=ledger, complete=converge_call,
            )
            for row in scenario_rows:
                scenarios.append(Scenario(**row))
            for row in falsifier_rows:
                falsifiers.append(Falsifier(**row))
            if convergence is not None:
                ch9 = ChapterResult(
                    number=9, title=spec.title, question=spec.question,
                    method="、".join(spec.methods), verdict=convergence.final_chain,
                    sources=ledger.source_rows()[:5],
                    gaps=[DataGap(item="不可复制性实证", chapters=[9],
                                  impact="护城河属性存疑",
                                  handling="需 L2-L3 多源验证，未找到即标注存疑")],
                    status="draft",
                )
                ch9.review = review_chapter(
                    ch9, ledger, experts=cfg.active_experts(), rounds=1,
                    prior=_prior_text(chapters), complete=converge_call,
                    max_opinions=cfg.max_review_opinions,
                )
                ch9.status = "reviewed"
                chapters.append(ch9)
        except BudgetExceeded as exc:
            budget_hit = True
            notes.append(f"第 9 章收敛时预算闸触发：{exc}")
    # 章节级 tracking 也汇总进来（模型在章节 JSON 里给的建议）
    for chapter in chapters:
        for claim in chapter.claims:
            if claim.claim_type == "inferred" and not any(
                item.judgment == claim.text for item in tracking
            ):
                tracking.append(
                    TrackingItem(
                        judgment=claim.text[:200],
                        indicator="待补：可观测先行指标与阈值",
                        direction="未定义证伪阈值，属不可验证断言",
                        source="待补 L1-L3 数据源",
                    )
                )
                break

    result.chapters = chapters
    result.convergence = convergence
    result.tracking = tracking[:12]
    result.falsifiers = falsifiers
    result.scenarios = scenarios

    # 收敛质量门禁在 execution 之后被 cross_checks / checklist 读取，故先落 execution。
    quality, quality_checks = assess_convergence(
        convergence, mainlines, result.scenarios, result.falsifiers, result.tracking
    )
    if not use_llm:
        notes.append("未启用 LLM：本次仅产出结构化骨架，所有验证章节为 skipped（零 token）")
    result.execution = ExecutionReport(
        target=target, period=period, mode=mode,
        enabled_chapters=chapter_numbers,
        skipped_chapters=skipped,
        llm_calls=budget.calls,
        estimated_input_tokens=budget.input_tokens,
        estimated_output_tokens=budget.output_tokens,
        review_rounds=cfg.review_rounds + 1 if use_llm else 0,
        chapter_model=plan.chapter_label if use_llm else "",
        convergence_model=plan.convergence_label if use_llm else "",
        model_route=plan.route,
        convergence_quality=quality,
        convergence_checks=quality_checks,
        notes=notes,
    )
    result.cross_checks = _cross_checks(result)
    result.checklist = _checklist(result, use_llm=use_llm)
    return result
