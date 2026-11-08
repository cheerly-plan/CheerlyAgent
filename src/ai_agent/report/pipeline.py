"""研报流水线编排：把确定性步骤串成一次可调用的任务。

设计约束：

* **数字路径零 LLM**：需求解析、指标抽取、增减计算、结论/表格/同比撰写、
  数字校验、PDF 渲染都是规则/代码——零「模型口算」。
* **分析节可选受约束 LLM**：若大纲含「经营解读」且 ``report_analysis_enabled``，
  用已抽取事实 + 季报原文证据写解读；数字与引用仍过 validate 门禁
  （正文里任何不在 allowed 集合里的数字都会被判 error）；失败则从略。
* **缺原文可自动巨潮补齐**（``enable_cninfo_fetch``）：解析需求后若缺本期/
  同比基期 PDF，按股票代码拉取并等到 ready，再抽数；不经对话「补文件」。
* 数字优先从知识库表格/正文抽取；半导体等行业 skill 对仍缺的比率类指标
  可用 akshare 结构化摘要补齐（带出处 quote），绝不凭空编造。
* PDF 渲染失败（缺中文字体 / WeasyPrint 缺系统库）时**降级为 Markdown 交付**，
  正文、溯源附件、人工审核队列照常产出，不因渲染环境整体失败。
* M2 定稿门禁：校验有 error 时**不渲染 PDF**；生成完成后登记审核记录
  （report/review.py），PDF 下载只放行已定稿的报告。

输出物统一落在 report_output_dir：正文 .md、溯源附件 .provenance.md、
人工审核队列 .review.json、可选 .pdf。文件名带 report_id 防覆盖。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.report.analyze import analyze_operations
from ai_agent.report.compare import SeriesCompare, compare_measures
from ai_agent.report.compose import ReportDocument, compose_report
from ai_agent.report.ensure_filings import ensure_filings_for_report
from ai_agent.report.extract import IndicatorFact, ensure_amount_units, extract_indicators
from ai_agent.report.parse import ReportPlan, parse_request
from ai_agent.report.market_finance import supplement_facts_from_market
from ai_agent.report.pdf import (
    ChartSpec,
    PdfFile,
    charts_for_report,
    render_html_pdf,
    render_report_pdf,
)
from ai_agent.report.review import create_deep_record, create_record
from ai_agent.report.supplemental import SupplementalResult, extract_supplemental
from ai_agent.report.validate import ReviewItem, ValidationResult, validate_report

if TYPE_CHECKING:  # 深度分析旁路默认关闭，运行时才局部 import，避免任何导入开销
    from ai_agent.report.deep.bridge import DeepOptions
    from ai_agent.report.deep.schema import DeepAnalysisResult

logger = get_logger(__name__)

_ILLEGAL = re.compile(r'[\\/:*?"<>|\s]+')
_NO_REQUEST = "没有看到研报或财报分析需求"


@dataclass
class ReportArtifacts:
    """落盘文件名（只给 basename，不给绝对路径，避免把服务端目录结构泄给前端）。"""

    markdown: str = ""
    provenance: str = ""
    review: str = ""
    pdf: str = ""


@dataclass
class ReportResult:
    request: str
    plan: ReportPlan
    document: ReportDocument
    validation: ValidationResult
    facts: list[IndicatorFact] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    series: list[SeriesCompare] = field(default_factory=list)
    views: SupplementalResult = field(default_factory=SupplementalResult)
    pdf: PdfFile | None = None
    report_id: str = ""
    artifacts: ReportArtifacts = field(default_factory=ReportArtifacts)
    output_dir: Path | None = None
    # ok / no_request / no_data
    status: str = "ok"
    # generated / not_requested / failed / blocked（校验未通过，禁止渲染）
    pdf_status: str = "not_requested"
    pdf_error: str = ""
    # pending / approved / rejected / finalized（由 review.create_record 回填）
    review_status: str = "pending"
    # 深度分析旁路（默认 None）：结论不进 Markdown 正文、不进 ReportArtifacts。
    # pdf_mode=merge 时同一份 PDF 末尾多出推断层附录（PDF 字节变化，
    # create_record 在落盘之后按最终字节算指纹，不会失配）；
    # separate 时另有 <stem>.deep.pdf，由独立审核记录覆盖，不参与主记录指纹。
    deep_analysis: "DeepAnalysisResult | None" = None
    deep_artifacts: dict[str, str] = field(default_factory=dict)
    deep_error: str = ""
    deep_write_artifacts: bool = True
    deep_pdf_merged: bool = False
    deep_pdf_error: str = ""
    # 双 PDF 档的推断层 PDF：单独落盘、单独审核，不影响底稿自动定稿。
    deep_pdf: PdfFile | None = None
    deep_pdf_review_id: str = ""
    deep_pdf_mode: str = ""  # 本次实际生效的 pdf_mode，供响应精确区分错误口径

    @property
    def deliverable(self) -> str:
        """实际交付形态：PDF 成功才算 pdf，否则是 markdown。"""
        return "pdf" if self.pdf is not None else "markdown"

    @property
    def gaps(self) -> list[str]:
        """计划缺口 + 撰写缺口 + 抽取缺口，去重保序，统一给调用方看。"""
        merged: list[str] = []
        for item in (*self.plan.gaps, *self.document.gaps, *self.missing):
            if item and item not in merged:
                merged.append(item)
        return merged

    @property
    def validation_summary(self) -> str:
        return self.validation.summary


def run_report(
    request: str,
    *,
    skill_id: str = "",
    deliverable: str | None = None,
    output_dir: Path | str | None = None,
    write_pdf: bool = True,
    deep: "DeepOptions | None" = None,
) -> ReportResult:
    """跑一次研报生成。同步、确定性、可单测；不触达 /v1/chat 的 Agent 循环。

    skill_id 是报告类型（report/skills.py 注册表）：显式指定时覆盖自动路由，
    大纲与默认指标集换用该 skill 的版本；抽数/计算/校验/渲染这四块积木
    对所有 skill 一视同仁，防幻觉门禁不因报告类型放宽。

    deliverable 显式传 "pdf" / "markdown" 时覆盖需求文本的推断；
    write_pdf=False 用于只要 Markdown 的场景（省掉渲染开销）。

    deep 是**默认关闭的深度分析旁路**：传 DeepOptions 才跑，且只在主报告
    的正文/校验/PDF 全部定稿后执行；深度层异常被隔离在 deep_error，
    主报告照常交付。深度结论永远单独落 .deep.json / .deep.md，不进正文。

    ``deep.pdf_mode="separate"``（默认）生成**两份 PDF**：事实层底稿保持纯
    数字事实、沿用原门禁可自动定稿；推断层单独出 ``<stem>.deep.pdf``，由
    ``<report_id>-deep`` 这条独立审核记录强制人工批准后才可下载。
    ``"merge"`` 是兼容档，仍把带标注的附录追加进同一份 PDF 并让整份报告
    进入人工审核。两种模式都不改 Markdown 正文、溯源附件与校验门禁。
    """
    from ai_agent.tools.progress import emit_progress

    # skill_id 为空时保持旧调用形态（只传 request）：外部对 parse_request
    # 的替身多为单参数 lambda，统一带 kwarg 会让它们全部 TypeError。
    # 用「理解需求」避免和 PDF 入库「解析中」混淆。
    emit_progress("正在理解研报需求…")
    plan = parse_request(request, skill_id=skill_id) if skill_id else parse_request(request)
    if deliverable in {"pdf", "markdown"}:
        plan.deliverable = deliverable
    want_pdf = plan.deliverable == "pdf" and write_pdf

    # 缺本期/基期原文时先巨潮补齐，再重新绑定 plan.files，避免空库硬抽。
    ensure_notes: list[str] = []
    try:
        subject_n = len(plan.subjects) or 1
        emit_progress(
            f"正在核对／补齐财报原文（{subject_n} 家公司）…"
            if subject_n > 1
            else "正在核对／补齐财报原文…"
        )
        ensured = ensure_filings_for_report(plan, request_text=request)
        ensure_notes = list(ensured.notes)
        if ensured.fetched_any:
            refreshed = (
                parse_request(request, skill_id=skill_id)
                if skill_id
                else parse_request(request)
            )
            if deliverable in {"pdf", "markdown"}:
                refreshed.deliverable = deliverable
            plan = refreshed
    except Exception:
        logger.exception("研报自动补齐原文失败，继续用现有知识库抽数")
        ensure_notes.append("自动补齐原文失败，已跳过")

    emit_progress("正在抽取财务指标…")
    facts, missing, views = _extract_all(plan)
    if ensure_notes:
        missing = [*ensure_notes, *missing]
    facts = ensure_amount_units(facts)
    measures = [f.to_measure() for f in facts]
    emit_progress("正在计算同比／环比与撰写正文…")
    series = compare_measures(measures)
    analysis = analyze_operations(plan, facts, series=series, views=views)
    if analysis.derived_facts:
        facts = [*facts, *analysis.derived_facts]
    document = compose_report(plan, facts, views=views, analysis=analysis.lines)
    if analysis.gaps:
        # 分析节缺口并入 document.gaps，便于审核队列与 gaps 属性统一可见。
        for gap in analysis.gaps:
            if gap and gap not in document.gaps:
                document.gaps.append(gap)
    # series 显式传进去：和撰写用的是同一批 Measure，校验允许的数字集合
    # 才和正文一致，不会出现「正文写了、校验说没算过」的假错。
    # views 也要传：单季列与较上年末变动的数字必须过原文核对。
    emit_progress("正在校验数字与出处…")
    validation = validate_report(
        document, facts, plan=plan, series=series, views=views
    )

    # M2 第一道门禁：正文数字/引用有 error，连 PDF 都不出，避免错误稿被外发。
    # 审核记录里仍能看到 error 明细，修数据后重新生成即可。
    blocked = want_pdf and not validation.passed
    # 合并附录时要复用同一批图表，避免第二次渲染重算导致图与数字不一致。
    charts: list[ChartSpec] = []
    result = ReportResult(
        request=request,
        plan=plan,
        document=document,
        validation=validation,
        facts=facts,
        missing=missing,
        series=series,
        views=views,
        report_id=uuid.uuid4().hex[:10],
        output_dir=Path(output_dir) if output_dir else settings.report_output_dir,
        status=_status(plan, facts),
        pdf_status=("not_requested" if not want_pdf else "blocked" if blocked else "failed"),
    )

    if blocked:
        result.pdf_error = "校验未通过：禁止渲染 PDF，请在审核队列处置错误后重新生成"
        document.deliverable = "markdown"
    elif want_pdf:
        emit_progress("正在渲染 PDF…")
        try:
            charts = charts_for_report(plan, facts, series)
            result.pdf = render_report_pdf(document, charts=charts)
            result.pdf_status = "generated"
        except Exception as exc:
            # 渲染是交付形态，不是数据正确性。缺字体/缺 Pango 时降级，
            # 但把原因如实写回结果，不能假装 PDF 生成过。
            result.pdf_error = f"{type(exc).__name__}: {exc}"
            document.deliverable = "markdown"
            logger.warning("研报 PDF 渲染失败，降级为 Markdown：%s", result.pdf_error)

    # 深度分析旁路：主报告的正文/校验/PDF 已全部定稿，这里只做加法。
    # 任何失败都不得影响事实层交付，故整段隔离并如实记入 deep_error。
    if deep is not None:
        result.deep_pdf_mode = deep.pdf_mode
        try:
            from ai_agent.report.deep.bridge import attach_deep

            emit_progress("正在生成深度分析旁路产物…")
            result.deep_analysis = attach_deep(result, deep)
            result.deep_write_artifacts = deep.write_artifacts
        except Exception as exc:
            result.deep_error = f"{type(exc).__name__}: {exc}"
            logger.exception("深度分析旁路失败（不影响主报告） report_id=%s", result.report_id)

        # pdf_mode=merge：把推断层附录追加进同一份 PDF。没有 PDF（markdown 交付、
        # 校验被挡住、渲染降级）时自然退化为 separate，不影响事实层交付。
        if (
            result.deep_analysis is not None
            and result.pdf is not None
            and deep.pdf_mode == "merge"
        ):
            emit_progress("正在合并深度分析附录…")
            _merge_deep_appendix(result, document, charts)
        elif (
            result.deep_analysis is not None
            and deep.pdf_mode == "separate"
            and deep.write_deep_pdf
        ):
            # 双 PDF 档：推断层单独成文件。事实层 PDF 缺失时函数内部会跳过，
            # 避免出现「没有底稿可对照的推断文件」单独外流。
            emit_progress("正在生成独立深度分析 PDF…")
            _render_separate_deep_pdf(result)

    emit_progress("正在落盘产物…")
    _write_artifacts(result)
    # 产物全部落盘后再登记审核：指纹才覆盖最终字节。登记失败宁可整体抛错，
    # 也不能出现「有 PDF、没有审核记录」的漏网之鱼。
    result.review_status = create_record(result, result.output_dir).status
    # 深度 PDF 的第二条审核记录也必须在落盘之后建，指纹才覆盖 .deep.* 三份文件。
    # 它失败不影响事实层交付：没有审核记录时下载端点一律拒绝（fail-closed）。
    if result.deep_artifacts.get("pdf"):
        try:
            deep_record = create_deep_record(result, result.output_dir)
            result.deep_pdf_review_id = deep_record.report_id if deep_record else ""
        except Exception as exc:
            result.deep_pdf_error = f"深度审核记录登记失败：{type(exc).__name__}: {exc}"
            logger.exception("独立深度 PDF 审核记录登记失败 report_id=%s", result.report_id)
    # 同标题旧版当场裁掉，避免「中行 2026Q3」连生成五份堆在目录里。
    try:
        from ai_agent.report.cleanup import prune_older_editions

        pruned = prune_older_editions(
            result.document.title,
            keep_report_id=result.report_id,
            output_dir=result.output_dir,
        )
        if pruned.deleted_report_ids:
            logger.info(
                "研报旧版已清理 title=%s kept=%s removed=%s",
                result.document.title,
                result.report_id,
                ",".join(pruned.deleted_report_ids),
            )
    except Exception:
        logger.exception("研报旧版清理失败（不影响本次生成） report_id=%s", result.report_id)
    return result


def _extract_all(
    plan: ReportPlan,
) -> tuple[list[IndicatorFact], list[str], SupplementalResult]:
    """按计划里出现的报告期逐一抽取（本期 / 同比基期 / 环比基期）。

    plan.files 为空不代表不用抽：库里没登记对应期时也要走一次，
    好让 extract 明确写出「没有找到 X 的季报」，而不是静默略过。
    """
    periods: list[str] = []
    for source in plan.files:
        if source.period and source.period not in periods:
            periods.append(source.period)
    if not periods and plan.period:
        periods.append(plan.period)
    if not plan.subjects or not plan.indicators:
        return [], ["需求里没有可用的公司或指标，跳过抽取"], SupplementalResult()

    facts: list[IndicatorFact] = []
    missing: list[str] = []
    for period in periods:
        extracted = extract_indicators(plan.subjects, period, plan.indicators)
        facts.extend(extracted.facts)
        missing.extend(f"[{period}] {item}" for item in extracted.missing)
    # PDF 抽不到的毛利率等比率：用结构化摘要补本期缺口（行业名单或巨潮代码）。
    facts, missing = supplement_facts_from_market(plan, facts, missing)
    # 单季/较上年末视图只查本期文件，不跨期拼数；缺口如实回传。
    views = extract_supplemental(plan.subjects, plan.period, plan.indicators)
    missing.extend(f"[视图] {item}" for item in views.gaps)
    return facts, missing, views


def _status(plan: ReportPlan, facts: list[IndicatorFact]) -> str:
    if plan.kind == "unknown" or _NO_REQUEST in plan.gaps:
        return "no_request"
    if not facts:
        return "no_data"
    return "ok"


def _stem(title: str, report_id: str) -> str:
    cleaned = _ILLEGAL.sub(" ", title or "").strip()[:60]
    return f"{cleaned or '研报'}_{report_id}"


_SAMPLE_IN_TARGET = re.compile(r"（样本：[^）]*）")


def deep_report_title(result: ReportResult) -> str:
    """深度层对外标题／文件名前缀：``{标的} {期间} 财报深度分析报告``。

    优先从事实层标题改写（行业底稿 → 「机器人行业 2026Q3 财报深度分析报告」），
    不把「样本：某某」写进文件名；回退时再拼 ``deep.target`` / ``deep.period``。
    """
    base = (result.document.title or "").strip()
    if base:
        title = (
            base.replace("财报对比底稿（赛道分组版）", "财报深度分析报告")
            .replace("对比底稿（赛道分组版）", "深度分析报告")
            .replace("财报对比底稿", "财报深度分析报告")
            .replace("对比底稿", "深度分析报告")
        )
        if "底稿" in title:
            title = title.replace("底稿", "深度分析报告")
        if title.endswith("财报深度分析") or (
            title.endswith("深度分析") and "深度分析报告" not in title
        ):
            title = f"{title}报告"
        if "深度分析" not in title:
            title = f"{title} 财报深度分析报告"
        return title

    deep = result.deep_analysis
    if deep is None:
        return "财报深度分析报告"
    target = _SAMPLE_IN_TARGET.sub("", deep.target or "").strip()
    if deep.mode == "industry" and target and not target.endswith("行业"):
        target = f"{target}行业"
    period = (deep.period or "").strip()
    if target and period:
        return f"{target} {period} 财报深度分析报告"
    return f"{target or period} 财报深度分析报告".strip()


def _deep_stem(result: ReportResult) -> str:
    """深度产物用独立文件名前缀，避免沿用事实层「底稿」标题看起来像同一份。"""
    return _stem(deep_report_title(result), result.report_id)


def _merge_deep_appendix(
    result: ReportResult, document: ReportDocument, charts: list[ChartSpec]
) -> None:
    """把推断层附录追加进事实层 PDF；失败保留原 PDF 并回退独立产物。

    合并成功后往 ``validation.review_queue`` 追加一条待审项，``create_record``
    会据此生成 pending 审核项：``gate()`` 因此拒绝自动定稿，PDF 下载端点也会
    拦到人工批准 + 定稿之后。**这是合并档必须重定义的门禁**，原来的
    「校验通过且无异常即自动定稿」不能替推断内容盖章，review.py 无需改动。
    """
    deep = result.deep_analysis
    if deep is None:
        return
    from ai_agent.report.deep.pdf_appendix import (
        APPENDIX_REVIEW_REASON,
        render_deep_appendix_html,
    )

    try:
        appendix = render_deep_appendix_html(deep)
        merged = render_report_pdf(document, charts=charts, appendix_html=appendix)
    except Exception as exc:
        # 重渲染失败 ≠ 深度分析失败：保留纯事实层 PDF，只如实记合并错误。
        result.deep_pdf_error = f"{type(exc).__name__}: {exc}"
        result.deep_pdf_merged = False
        logger.warning(
            "深度分析附录合并失败，回退独立产物 report_id=%s error=%s",
            result.report_id,
            result.deep_pdf_error,
        )
        return
    result.pdf = merged
    result.deep_pdf_merged = True
    result.validation.review_queue.append(
        ReviewItem(
            reason=APPENDIX_REVIEW_REASON,
            value="deep_pdf_merge",
            subject=deep.target,
            period=deep.period,
        )
    )


def _render_separate_deep_pdf(result: ReportResult) -> None:
    """双 PDF 档：把推断层单独渲染成 ``<stem>.deep.pdf``。

    前置是事实层 PDF 已成功生成——校验被挡、markdown 交付或渲染降级时，
    宁可只留 ``.deep.json`` / ``.deep.md``，也不让一份没有底稿可对照的推断
    PDF 单独流出去。渲染失败只记 ``deep_pdf_error``，事实层照常交付。

    注意这里**不往主审核队列追加任何条目**：底稿因此保持 clean、可自动定稿，
    推断层的强制人工审由 ``create_deep_record`` 的独立记录承担。
    """
    deep = result.deep_analysis
    if deep is None:
        return
    if result.pdf is None:
        result.deep_pdf_error = "事实层 PDF 未生成，独立深度 PDF 一并跳过"
        return
    if not result.deep_write_artifacts:
        result.deep_pdf_error = "深度层产物落盘被关闭，独立深度 PDF 一并跳过"
        return
    from ai_agent.report.deep.pdf_appendix import deep_standalone_document

    try:
        result.deep_pdf = render_html_pdf(
            deep_standalone_document(deep),
            title=f"{deep_report_title(result)}（推断层）",
            deep_css=True,
        )
    except Exception as exc:
        result.deep_pdf = None
        result.deep_pdf_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "独立深度 PDF 渲染失败，仅保留结构化产物 report_id=%s error=%s",
            result.report_id,
            result.deep_pdf_error,
        )


def _write_artifacts(result: ReportResult) -> None:
    """正文、溯源总是落盘；PDF 只在渲染成功时落盘。

    .review.json 由 review.create_record 在最后统一写完整审核记录（M2），
    这里只登记文件名，避免同一份文件写两遍、或留下 M1 那种只有数组的快照。
    """
    directory = result.output_dir or settings.report_output_dir
    directory.mkdir(parents=True, exist_ok=True)
    stem = _stem(result.document.title, result.report_id)

    markdown_name = f"{stem}.md"
    provenance_name = f"{stem}.provenance.md"
    review_name = f"{stem}.review.json"
    (directory / markdown_name).write_text(result.document.markdown, encoding="utf-8")
    (directory / provenance_name).write_text(
        result.validation.provenance.markdown, encoding="utf-8"
    )
    result.artifacts.markdown = markdown_name
    result.artifacts.provenance = provenance_name
    result.artifacts.review = review_name

    if result.pdf is not None:
        pdf_name = f"{stem}.pdf"
        (directory / pdf_name).write_bytes(result.pdf.content)
        result.artifacts.pdf = pdf_name

    # 深度分析产物与事实层物理隔离：独立文件名前缀 + 独立后缀，
    # 不写进 document.markdown，不进 result.artifacts。
    if result.deep_analysis is not None and result.deep_write_artifacts:
        deep_stem = _deep_stem(result)
        deep_json_name = f"{deep_stem}.deep.json"
        deep_md_name = f"{deep_stem}.deep.md"
        (directory / deep_json_name).write_text(
            result.deep_analysis.to_json(), encoding="utf-8"
        )
        (directory / deep_md_name).write_text(
            result.deep_analysis.to_markdown(), encoding="utf-8"
        )
        result.deep_artifacts = {"json": deep_json_name, "markdown": deep_md_name}
        if result.deep_pdf is not None:
            deep_pdf_name = f"{deep_stem}.deep.pdf"
            (directory / deep_pdf_name).write_bytes(result.deep_pdf.content)
            result.deep_artifacts["pdf"] = deep_pdf_name
