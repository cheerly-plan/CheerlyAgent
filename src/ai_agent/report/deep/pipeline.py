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

import threading
from concurrent.futures import ThreadPoolExecutor
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
    guard_conclusion,
    guard_numbers,
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

# 对外入口（HTTP / 聊天工具）的默认章节并行度：库级 DeepConfig 仍是 1（既有
# 行为逐字不变），入口显式提速。理由：章节之间本来是独立验证，串行只为了把
# 前序结论带进 prompt；并行改用基线快照可把整份报告耗时压到接近单章。
ENTRY_CHAPTER_CONCURRENCY = 3


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
    # 第 2-8 章并行度。1（默认）= 原有逐章串行、保留前序章节结论链；>1 时
    # 前序上下文改用基线快照（候选主线 + 台账要点），结果仍按章节号确定性汇总。
    chapter_concurrency: int = 1

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
        # 章节并行时多个线程会同时经过 wrap，计数与字符累加必须原子化。
        self._lock = threading.Lock()

    def wrap(self, fn):
        def call(system: str, user: str) -> str:
            with self._lock:
                if self.calls >= self.limit:
                    raise BudgetExceeded(f"LLM 调用达到上限 {self.limit}")
                self.calls += 1
                self.input_chars += len(system) + len(user)
            raw = fn(system, user)
            with self._lock:
                self.output_chars += len(raw or "")
            return raw or ""
        return call

    @property
    def input_tokens(self) -> int:
        return int(self.input_chars / 1.5)

    @property
    def output_tokens(self) -> int:
        return int(self.output_chars / 1.5)


def pick_industry_sample(ledger: SourceLedger) -> tuple[str, str]:
    """行业深度先选定样本公司，避免「行业深度」实际只默写一家却不声明。

    选取规则：本期台账里事实条数最多的主体；并列时按名称稳定排序。
    返回 (样本公司, 选取说明)；台账无主体时样本为空串。
    """
    counts: dict[str, int] = {}
    for record in ledger.records:
        subject = (record.subject or "").strip()
        if subject:
            counts[subject] = counts.get(subject, 0) + 1
    if not counts:
        return "", "行业深度暂无可用样本公司（台账无主体字段）；不得假装覆盖全行业。"
    sample, n = max(counts.items(), key=lambda item: (item[1], item[0]))
    others = sorted(name for name in counts if name != sample)
    note = (
        f"样本公司：{sample}（本期台账事实 {n} 条，按条数最多选取）；"
        "第 2-9 章以该公司为推断对象，其余代表公司仅作对照背景，不逐家展开 10 章。"
        "行业格局（产能/份额/价格）仍依赖 L3 外部数据，未接入前第 6 章应诚实从略。"
    )
    if others:
        shown = "、".join(others[:8])
        note += f" 对照池：{shown}" + (" 等" if len(others) > 8 else "") + "。"
    return sample, note


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


def _baseline_prior(result: DeepAnalysisResult, ledger: SourceLedger) -> str:
    """章节并行时的静态上下文：候选主线 + 台账要点，不含其它章节的结论。

    串行模式靠 ``_prior_text`` 把前序章节结论带进 prompt；并行时章节同时开跑，
    不能引用尚未产出的结论，否则同一份报告两次运行会得到不同输入。这里换成
    对每章都相同的确定性快照，保证并发结果与章节执行顺序无关。
    """
    lines = [
        f"候选主线：{item.business}（{item.rule}）{item.chain}"
        for item in result.mainlines[:3]
    ]
    lines.extend(
        f"台账要点：{record.item}={record.value}{record.unit}（{record.tier}）"
        for record in ledger.records[:8]
    )
    return "\n".join(lines) if lines else "（无前序章节）"


def _skipped_chapter(number: int) -> ChapterResult:
    spec = spec_of(number)
    return ChapterResult(
        number=number, title=spec.title, question=spec.question,
        method="、".join(spec.methods), status="skipped",
    )


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


def _guard_result_numbers(result: DeepAnalysisResult, ledger: SourceLedger) -> None:
    """推断层数字台账兜底门禁：所有模型给出的自由文本都不许带台账外数字。

    章节 verdict / 收敛 final_chain 已有 ``guard_conclusion`` 第一道门禁；这里覆盖
    评审意见、护城河分项、跟踪与证伪、三情景说明等同样会被印进 PDF 的模型文本。
    遮蔽而非删除，保留结论骨架，并在章节 gaps 里留可追溯的缺口记录。
    事实层的 sources 表、执行报告里的工程计量与诊断数字不属于模型结论，不在此列。
    """
    allowed = ledger.allowed_numbers

    def safe(text: str) -> str:
        return guard_numbers(text or "", allowed)[0]

    for chapter in result.chapters:
        chapter.verdict = safe(chapter.verdict)
        chapter.fallback_reason = safe(chapter.fallback_reason)
        for claim in chapter.claims:
            claim.text = safe(claim.text)
        for gap in chapter.gaps:
            gap.item = safe(gap.item)
            gap.impact = safe(gap.impact)
            gap.handling = safe(gap.handling)
        for opinion in chapter.review:
            opinion.issue = safe(opinion.issue)
            opinion.reason = safe(opinion.reason)
            opinion.revision = safe(opinion.revision)
    if result.convergence is not None:
        conv = result.convergence
        conv.final_chain = safe(conv.final_chain)
        conv.delta = safe(conv.delta)
        conv.falsifiers = [safe(item) for item in conv.falsifiers]
        conv.moat_sources = {key: safe(value) for key, value in conv.moat_sources.items()}
        conv.decisions = [
            {
                **row,
                "mainline": safe(str(row.get("mainline", ""))),
                "reason": safe(str(row.get("reason", ""))),
            }
            for row in conv.decisions
        ]
    for item in result.tracking:
        item.judgment = safe(item.judgment)
        item.indicator = safe(item.indicator)
        item.direction = safe(item.direction)
        item.source = safe(item.source)
    for item in result.falsifiers:
        item.judgment = safe(item.judgment)
        item.signal = safe(item.signal)
    for item in result.scenarios:
        item.assumption = safe(item.assumption)
        item.trigger = safe(item.trigger)
        item.impact = safe(item.impact)


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
    sample, sample_note = ("", "")
    display_target = target
    if mode == "industry":
        sample, sample_note = pick_industry_sample(ledger)
        if sample:
            display_target = f"{target}（样本：{sample}）"
    mainlines, company_type = generate_mainlines(ledger, mode)
    if mode == "industry" and sample:
        company_type = (
            f"行业横比（样本公司={sample}；其余主体仅作对照）；{CONTROL_GROUP_NOTE}"
        )
    result = DeepAnalysisResult(
        target=display_target, period=period, mode=mode,
        mainlines=mainlines, company_type=company_type,
        ledger_gaps=list(ledger.gaps),
    )

    uses_stub = complete is not None or convergence_complete is not None
    use_llm = cfg.enabled or uses_stub
    budget = _Budget(cfg.max_llm_calls)
    model_events: list[str] = []
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
        # 思考型小模型会把 token 全耗在 reasoning 上返回空 content；此时用收敛通道
        # 的主模型重试一次，事件写进 ExecutionReport 与 PDF 附录 A，便于成本归因。
        call = budget.wrap(default_complete(
            cfg.max_tokens, cfg.temperature, cfg.timeout, plan.chapter_model,
            escalation_model=plan.convergence_model, events=model_events,
        ))
        converge_call = budget.wrap(default_complete(
            cfg.max_tokens, cfg.temperature, cfg.timeout, plan.convergence_model,
            events=model_events,
        ))
    for warning in plan.warnings:
        logger.warning("深度分析模型路由告警：%s", warning)

    notes = [
        "第 1 章候选主线由五规则确定性生成（零 LLM 调用），性质为假设",
        "第 10 章 checklist 由确定性规则生成（零 LLM 调用）",
        f"评审协议：{len(cfg.active_experts())} 视角 × {cfg.review_rounds} 轮 + 1 次一致性/终审合并轮"
        + ("（全量 12 视角）" if cfg.full_panel else "（非 12专家×5轮全量，成本折中）"),
        "token 为按字符数粗估，项目当前未接 LLM usage 统计",
    ]
    if sample_note:
        notes.insert(0, sample_note)
    if ledger.unit_conversions:
        notes.append(
            f"单位换算白名单：{len(ledger.unit_conversions)} 条台账金额已放行万元/亿元"
            "等价写法（按固定倍率从 L1 原文确定性换算，非模型心算；"
            "同比、差值、占比等计算派生值仍一律遮蔽）"
        )
    if use_llm:
        notes.append(
            f"模型路由：{plan.route}；章节/评审={plan.chapter_label or '—'}，"
            f"第 9 章收敛与终审={plan.convergence_label or '—'}"
            "（决策 ④：小模型优先；空 content 自动升级主模型重试一次）"
        )
        notes.extend(plan.warnings)

    chapters: list[ChapterResult] = []
    budget_hit = False
    # 第 2-8 章：逐章验证 + 评审（第 1/9/10 章单独处理）
    chapter_2_8 = [n for n in chapter_numbers if 2 <= n <= 8]
    workers = max(1, min(int(cfg.chapter_concurrency or 1), len(chapter_2_8) or 1))
    parallel = workers > 1 and call is not None
    if parallel:
        # 章节之间本来是独立验证；并行时不能引用尚未产出的前序结论，否则
        # prompt 会随线程完成顺序变化。统一换成确定性基线快照。
        baseline = _baseline_prior(result, ledger)
        notes.append(
            f"章节并行：第 2-8 章 {workers} 并发，前序上下文改用基线快照"
            "（候选主线+台账要点），结果按章节号确定性汇总；模型事件按完成顺序记录"
        )

        def run_one(number: int) -> tuple[ChapterResult, str]:
            spec = spec_of(number)
            try:
                chapter = analyze_chapter(
                    spec, ledger, target=display_target, period=period, mode=mode,
                    prior=baseline, complete=call,
                    max_claims=cfg.max_claims_per_chapter,
                )
                chapter.review = review_chapter(
                    chapter, ledger, experts=cfg.active_experts(), rounds=cfg.review_rounds,
                    prior=baseline, complete=call,
                    max_opinions=cfg.max_review_opinions,
                )
                if chapter.review and chapter.status != "fallback":
                    chapter.status = "reviewed"
                return chapter, ""
            except BudgetExceeded as exc:
                return _skipped_chapter(number), str(exc)

        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="deep-chapter"
        ) as pool:
            futures = {n: pool.submit(run_one, n) for n in chapter_2_8}
            for number in chapter_2_8:
                chapter, reason = futures[number].result()
                if reason and not budget_hit:
                    budget_hit = True
                    notes.append(f"预算闸触发：{reason}；后续章节标 skipped")
                chapters.append(_skipped_chapter(number) if budget_hit else chapter)
    else:
        for number in chapter_2_8:
            spec = spec_of(number)
            if call is None or budget_hit:
                chapters.append(_skipped_chapter(number))
                continue
            try:
                chapter = analyze_chapter(
                    spec, ledger, target=display_target, period=period, mode=mode,
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
                chapters.append(_skipped_chapter(number))

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
                target=display_target, period=period, mainlines_text=mainlines_text,
                prior=_prior_text(chapters), ledger=ledger, complete=converge_call,
            )
            for row in scenario_rows:
                scenarios.append(Scenario(**row))
            for row in falsifier_rows:
                falsifiers.append(Falsifier(**row))
            if convergence is not None:
                # 收敛结论与章节 verdict 同样过数字台账门禁：此前 final_chain 直接
                # 赋值绕过 numbers_allowed，导致推断层 PDF 出现台账外数字。
                safe_chain, chain_gap = guard_conclusion(convergence.final_chain, ledger, 9)
                safe_delta, delta_gap = guard_conclusion(convergence.delta, ledger, 9)
                convergence.final_chain = safe_chain
                convergence.delta = safe_delta
                ch9_gaps = [DataGap(item="不可复制性实证", chapters=[9],
                                    impact="护城河属性存疑",
                                    handling="需 L2-L3 多源验证，未找到即标注存疑")]
                ch9_gaps.extend(gap for gap in (chain_gap, delta_gap) if gap)
                ch9 = ChapterResult(
                    number=9, title=spec.title, question=spec.question,
                    method="、".join(spec.methods), verdict=safe_chain,
                    sources=ledger.source_rows()[:5],
                    gaps=ch9_gaps,
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
    _guard_result_numbers(result, ledger)

    # 收敛质量门禁在 execution 之后被 cross_checks / checklist 读取，故先落 execution。
    quality, quality_checks = assess_convergence(
        convergence, mainlines, result.scenarios, result.falsifiers, result.tracking
    )
    if not use_llm:
        notes.append("未启用 LLM：本次仅产出结构化骨架，所有验证章节为 skipped（零 token）")
    if model_events:
        notes.extend(f"模型事件 {i}：{event}" for i, event in enumerate(model_events, 1))
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
        model_events=model_events,
        notes=notes,
    )
    result.cross_checks = _cross_checks(result)
    result.checklist = _checklist(result, use_llm=use_llm)
    return result
