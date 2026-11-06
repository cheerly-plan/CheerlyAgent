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

服务端确定性流水线（解析 → 抽取 → 计算 → 撰写 → 校验 → 渲染）不调用 LLM，
所以这个工具本身零 token 成本，耗时主要是知识库检索与 PDF 渲染。
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
    from ai_agent.tools.progress import emit_progress

    emit_progress("正在生成研报…")
    result = run_report(request, skill_id=report_type or "", write_pdf=write_pdf)
    gaps = len(getattr(result, "gaps", []) or [])
    if gaps:
        emit_progress(f"研报已生成（另有 {gaps} 处数据缺口待说明）")
    else:
        emit_progress("研报已生成")
    artifacts = result.artifacts
    lines = [
        "研报已由服务端确定性流水线生成（正文数字只来自知识库表格，已过校验门禁）。",
        f"标题：{result.document.title or '（未识别到标题）'}",
        f"状态：{result.status} / 实际交付：{result.deliverable} / PDF：{result.pdf_status}",
        (
            f"校验：{result.validation.summary}"
            f"（错误 {sum(1 for item in result.validation.findings if item.severity == 'error')} 条，"
            f"待人工审核 {len(result.validation.review_queue)} 条）"
        ),
        "把下面的 Markdown 下载链接原样发给用户（方括号标题 + 圆括号路径不要改），"
        "并简要说明校验结论与缺口；不要自己撰写研报正文、不要复述或改写报告里的数字。"
        "若缺口里出现「入库等待超时」或「尚未 ready」，请明确告诉用户是原文还在解析、"
        "不是聊天超时；可建议稍后重试或到知识库查看文件状态。",
    ]
    # M2：审核状态决定 PDF 链接是否可点，必须让模型如实转述，别让用户白点。
    review_status = getattr(result, "review_status", PENDING)
    report_id = getattr(result, "report_id", "")
    if review_status == FINALIZED:
        lines.append("审核：已定稿，PDF 可直接下载。")
    else:
        lines.append(
            f"审核：尚未定稿（{review_status}），PDF 链接需要管理员在"
            f" /v1/report/reviews/{report_id} 处置审核项并定稿后才能下载；"
            "Markdown 正文与溯源附件可先预览。请如实告知用户。"
        )
    downloads = [
        ("PDF", artifacts.pdf),
        ("Markdown 正文", artifacts.markdown),
        ("溯源附件", artifacts.provenance),
        ("人工审核队列", artifacts.review),
    ]
    for label, name in downloads:
        if name:
            # 前端认 Markdown 链与裸路径；中文文件名做百分号编码，避免空格截断
            lines.append(f"- [{label}](/v1/report/files/{quote(name, safe='')})")
    if result.pdf_error:
        lines.append(f"PDF 未生成的原因：{result.pdf_error[:150]}")

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
