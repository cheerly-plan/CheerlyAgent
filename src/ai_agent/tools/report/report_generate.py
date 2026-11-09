"""聊天侧的研报生成入口：模型只给一句需求，落盘位置由服务端固定。

分层（与 report/skills/ 注册表配合）：

* 本工具是「通用报告入口」——报告类型默认自动路由，用户点名时才传
  ``report_type``。skill 决定大纲与指标集；抽数、计算、校验、渲染仍是
  流水线里的确定性积木，不因报告类型放宽防幻觉门禁。
* 抽数/算增减/校验三块积木另以 ``report_extract_indicators`` /
  ``report_compare`` / ``report_validate`` 单独注册，供模型做轻量查询，
  它们无路径参数、不写盘。
* ``report_pdf`` 仍不注册：渲染含落盘与 M2 审核门禁（校验不过不出 PDF、
  下载须过定稿门禁），只能由服务端流水线触发，不能交给模型自主调用。

默认交付**两份报告**：事实层底稿 PDF + 独立深度解析 PDF（``.deep.pdf``）。
深度结论永不并入 Markdown 正文；深度 PDF 走独立审核记录。
"""

from __future__ import annotations

from urllib.parse import quote

from pydantic import BaseModel, Field

from ai_agent.prompts.tools import render_report_generate_desc, render_report_type_desc
from ai_agent.report.pipeline import run_report
from ai_agent.report.review import FINALIZED, PENDING
from ai_agent.report.skills import skill_catalog
from ai_agent.tools.base import Tool

# execute_tool 会把工具返回截到 3000 字符，缺口只挑前几条如实转述。
_MAX_GAPS = 5
_GAP_CHARS = 110

# 工具描述在组装层渲染：catalog 来自 skills 注册表（tools 层允许依赖业务域，
# prompts 层只留模板）。注册新 skill 即自动出现在描述里，模型才知道能点名。
_REPORT_CATALOG = skill_catalog()
REPORT_GENERATE_DESC = render_report_generate_desc(_REPORT_CATALOG)
REPORT_TYPE_DESC = render_report_type_desc(_REPORT_CATALOG)


class GenerateReportArgs(BaseModel):
    request: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "用户的研报需求原话，原样传入，不要改写、不要自己拆公司名或指标。"
            "例如「生成中国银行2026Q3财报深度分析PDF研报」"
        ),
    )
    report_type: str = Field(
        default="",
        max_length=40,
        # 可选值从 skills 注册表渲染（skill_catalog()），与工具描述同源，不手抄
        description=REPORT_TYPE_DESC,
    )
    write_pdf: bool = Field(
        default=True,
        description=(
            "是否渲染 PDF，默认 True。"
            "仅当用户明确只要 Markdown / 不要 PDF 时才传 False；"
            "渲染环境不可用时自动降级为 Markdown，不会整体失败"
        ),
    )


def _generate_report(request: str, report_type: str = "", write_pdf: bool = True) -> str:
    from ai_agent.report.deep import (
        ENTRY_CHAPTER_CHANNEL,
        ENTRY_CHAPTER_CONCURRENCY,
        default_facts_dir,
    )
    from ai_agent.report.deep.bridge import DeepOptions
    from ai_agent.report.progress import msg_entry_with_deep, msg_report_done
    from ai_agent.tools.progress import emit_progress

    emit_progress(msg_entry_with_deep())
    # 聊天入口默认双 PDF：底稿可自动定稿；深度 .deep.pdf 单独强制人工审。
    result = run_report(
        request,
        skill_id=report_type or "",
        write_pdf=write_pdf,
        deep=DeepOptions(
            enabled=True, pdf_mode="separate",
            chapter_concurrency=ENTRY_CHAPTER_CONCURRENCY,
            # 对外入口章节/评审直接主模型（决策 ④-B）：小模型空响应已实测 100%，
            # 先跑一轮只会白付延迟；库级默认仍是 small。
            chapter_channel=ENTRY_CHAPTER_CHANNEL,
            # 对外入口显式挂载结构化事实清单；库级 DeepOptions 默认不启用，
            # 目录不存在时由 LocalFactsProvider 如实记缺口，不假装有数据。
            facts_dir=default_facts_dir(),
        ),
    )
    gaps = len(getattr(result, "gaps", []) or [])
    emit_progress(msg_report_done(gaps=gaps))
    artifacts = result.artifacts
    lines = [
        "研报已由服务端确定性流水线生成（正文数字只来自知识库表格，已过校验门禁）。",
        "默认同时交付：事实层底稿 + 独立深度解析（推断层，不并入底稿正文）。",
        f"标题：{result.document.title or '（未识别到标题）'}",
        f"状态：{result.status} / 实际交付：{result.deliverable} / PDF：{result.pdf_status}",
        (
            f"校验：{result.validation.summary}"
            f"（错误 {sum(1 for item in result.validation.findings if item.severity == 'error')} 条，"
            f"待人工审核 {len(result.validation.review_queue)} 条）"
        ),
        "把下面的 Markdown 下载链接**逐字原样**发给用户（方括号标题 + 圆括号路径一个字符都不要改），"
        "底稿与深度两套都要发；并简要说明校验结论与缺口；"
        "禁止改成 http/https、example.com、其它域名或绝对 URL，必须保持 /v1/report/files/ 开头；"
        "不要自己撰写研报正文、不要复述或改写报告里的数字。"
        "若缺口里出现「入库等待超时」或「尚未 ready」，请明确告诉用户是原文还在解析、"
        "不是聊天超时；可建议稍后重试或到知识库查看文件状态。",
    ]
    # M2：审核状态决定 PDF 链接是否可点，必须让模型如实转述，别让用户白点。
    review_status = getattr(result, "review_status", PENDING)
    report_id = getattr(result, "report_id", "")
    if review_status == FINALIZED:
        lines.append("审核：事实层底稿已定稿，底稿 PDF 可直接下载。")
    else:
        lines.append(
            f"审核：事实层尚未定稿（{review_status}），底稿 PDF 链接需要管理员在"
            f" /v1/report/reviews/{report_id} 处置审核项并定稿后才能下载；"
            "Markdown 正文与溯源附件可先预览。请如实告知用户。"
        )
    downloads = [
        ("底稿 PDF", artifacts.pdf),
        ("底稿 Markdown", artifacts.markdown),
        ("溯源附件", artifacts.provenance),
        ("人工审核队列", artifacts.review),
    ]
    for label, name in downloads:
        if name:
            # 前端认 Markdown 链与裸路径；中文文件名做百分号编码，避免空格截断
            lines.append(f"- [{label}](/v1/report/files/{quote(name, safe='')})")
    if result.pdf_error:
        lines.append(f"底稿 PDF 未生成的原因：{result.pdf_error[:150]}")

    deep_artifacts = getattr(result, "deep_artifacts", None) or {}
    deep_error = getattr(result, "deep_error", "") or ""
    lines.append(
        "深度解析为推断层，与事实层物理隔离；深度 PDF 须单独人工批准后才能下载。"
    )
    deep_downloads = [
        ("深度解析 PDF", deep_artifacts.get("pdf")),
        ("深度解析 Markdown", deep_artifacts.get("markdown")),
        ("深度解析 JSON", deep_artifacts.get("json")),
    ]
    for label, name in deep_downloads:
        if name:
            lines.append(f"- [{label}](/v1/report/files/{quote(name, safe='')})")
    deep_review_id = getattr(result, "deep_pdf_review_id", "") or ""
    if deep_review_id:
        lines.append(
            f"深度 PDF 审核队列：/v1/report/reviews/{deep_review_id}"
            "（与事实层底稿分开审核，互不影响）。"
        )
    if deep_error:
        lines.append(f"深度解析失败（不影响底稿）：{deep_error[:180]}")
    pdf_err = getattr(result, "deep_pdf_error", "") or ""
    if pdf_err and not deep_artifacts.get("pdf"):
        lines.append(f"深度 PDF 未生成的原因：{pdf_err[:150]}")

    gaps = result.gaps[:_MAX_GAPS]
    if gaps:
        lines.append("缺口（如实转述，不要补数）：")
        lines.extend(f"- {gap[:_GAP_CHARS]}" for gap in gaps)
        if len(result.gaps) > _MAX_GAPS:
            lines.append(f"（另有 {len(result.gaps) - _MAX_GAPS} 条缺口未列出）")
    if result.status != "ok":
        lines.append("状态不是 ok：请如实说明没有生成有效研报及原因，不要补写数字。")
    return "\n".join(lines)


generate_report_tool = Tool(
    name="generate_report",
    description=REPORT_GENERATE_DESC,
    func=_generate_report,
    args_model=GenerateReportArgs,
)
