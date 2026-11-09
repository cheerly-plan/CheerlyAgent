"""深度分析层 ↔ 主研报流水线的胶水层（Phase 2）。

**挂接原则**：只做「主产物落定之后」的旁路，不参与 parse/extract/compare/
compose/validate 任何一步。事实层的 Markdown 正文始终是数字事实的唯一权威来源，
本层结论单独落到 ``<stem>.deep.json`` / ``<stem>.deep.md``，
**绝不合并进 Markdown 正文**，也不进 ``ReportArtifacts``（PDF 附录例外，见下）。

PDF 有两种模式（``DeepOptions.pdf_mode``）：

* ``separate``（默认）：**两份 PDF**。事实层底稿保持纯数字事实、可自动定稿；
  推断层单独渲染成 ``<stem>.deep.pdf``，由 ``<report_id>-deep`` 这条独立
  审核记录强制人工批准，底稿外发不受推断拖住；
* ``merge``（兼容档）：在事实层 PDF **末尾追加显式标注的推断层附录**，
  正文页面、Markdown、溯源、校验门禁一字不改；合并成功后 pipeline 会往
  ``validation.review_queue`` 追加一条待审项，整份报告禁止自动定稿。

两条审核记录都走 ``review.py`` 同一套状态机与指纹门禁，不为推断层另开弱门禁。

``run_report(deep=None)`` 时本模块完全不会被 import，零开销、零行为变化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from ai_agent.report.deep.pipeline import DeepConfig, run_deep_analysis
from ai_agent.report.deep.schema import DeepAnalysisResult
from ai_agent.report.deep.sources import (
    DEFAULT_YEARS,
    SourceBundle,
    SourceProvider,
    collect_sources,
)

if TYPE_CHECKING:  # 避免 pipeline <-> bridge 循环导入
    from ai_agent.report.pipeline import ReportResult

Complete = Callable[[str, str], str]
ListFiles = Callable[[], Sequence[dict]]
StatFetcher = Callable[[str, str, int], Sequence[dict]]


@dataclass
class DeepOptions:
    """传给 ``run_report`` 的可选旁路参数。不传 = 不跑深度分析。

    ``enabled=False`` 且未注入 ``complete`` 时是**零 token 骨架**：只产出
    台账、候选主线、缺口与 checklist，适合先验证数据覆盖度再决定是否烧钱。
    """

    enabled: bool = False
    mode: str = ""  # 空 = 按 plan.kind 自动推断 company / industry
    target: str = ""  # 空 = 按 plan.subjects / plan.industry 推断
    years: int = DEFAULT_YEARS
    write_artifacts: bool = True
    # separate（默认）= 两份 PDF，深度 PDF 单独审核；merge = 兼容档，附录进同一份 PDF
    pdf_mode: str = "separate"
    # False 回退到上一轮的「只落 .deep.json / .deep.md，不渲染独立深度 PDF」。
    # API 侧等价回退是 deliverable="markdown"（事实层无 PDF 时深度 PDF 一并跳过）。
    write_deep_pdf: bool = True
    collect_external: bool = True
    facts_dir: Path | str | None = None
    config: DeepConfig | None = None
    # 第 2-8 章并行度，只在未显式传 config 时生效。1（库级默认）= 原有逐章
    # 串行与前序结论链；对外入口会显式设为 ENTRY_CHAPTER_CONCURRENCY 提速。
    chapter_concurrency: int = 1
    complete: Complete | None = None  # 注入 stub 便于零成本跑通 / A/B
    list_files: ListFiles | None = None  # 注入知识库行，避免测试连库
    stat_fetcher: StatFetcher | None = None  # 行业统计注入点
    providers: Sequence[SourceProvider] | None = None
    notes: list[str] = field(default_factory=list)


def resolve_target_mode(plan, options: DeepOptions) -> tuple[str, str]:
    """从 ReportPlan 推断标的与分析模式；显式 options 优先。"""
    kind = str(getattr(plan, "kind", "") or "")
    mode = options.mode or ("industry" if kind == "industry" else "company")
    if options.target:
        return options.target, mode
    if mode == "industry":
        return str(getattr(plan, "industry", "") or getattr(plan, "title", "") or "行业"), mode
    subjects = list(getattr(plan, "subjects", None) or [])
    return (
        str(subjects[0]) if subjects else str(getattr(plan, "title", "") or "未识别标的"),
        mode,
    )


def _merge_gaps(deep: DeepAnalysisResult, gaps) -> None:
    seen = {
        f"{gap.item}|{','.join(str(n) for n in gap.chapters)}"
        for gap in deep.ledger_gaps
    }
    for gap in gaps:
        key = f"{gap.item}|{','.join(str(n) for n in gap.chapters)}"
        if key not in seen:
            seen.add(key)
            deep.ledger_gaps.append(gap)


def attach_deep(result: "ReportResult", options: DeepOptions) -> DeepAnalysisResult:
    """在主报告结果上跑一次深度分析，返回结构化结论对象。

    外部数据源的缺口与盘点说明会并入 ``ledger_gaps`` / ``execution.notes``，
    这样深度报告里能看到「第 2 章缺近 5 年审计报告」这种可执行清单。
    """
    plan = result.plan
    target, mode = resolve_target_mode(plan, options)
    period = str(getattr(plan, "period", "") or "未标注报告期")

    bundle = SourceBundle()
    if options.collect_external:
        bundle = collect_sources(
            target,
            period,
            years=options.years,
            mode=mode,
            facts_dir=options.facts_dir,
            list_files=options.list_files,
            stat_fetcher=options.stat_fetcher,
            providers=options.providers,
        )

    cfg = options.config or DeepConfig(
        enabled=options.enabled,
        chapter_concurrency=options.chapter_concurrency,
    )
    deep = run_deep_analysis(
        target,
        period,
        list(result.facts or []),
        mode=mode,
        config=cfg,
        complete=options.complete,
        external_sources=bundle.ledger_kwargs(),
    )
    _merge_gaps(deep, bundle.gaps)
    notes = [
        f"外部数据源：{len(bundle.facts)} 条结构化事实候选、"
        f"{len(bundle.documents)} 份源文件、{len(bundle.gaps)} 项缺口",
        f"近 {options.years} 年窗口（{period} 为基准）",
        *bundle.notes,
        *options.notes,
    ]
    deep.execution.notes.extend(notes)
    return deep
