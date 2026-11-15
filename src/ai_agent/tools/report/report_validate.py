"""数据校验工具：核对一段财务分析里的数字与引用能否溯源到抽取事实。

名字不带 ``report_`` 前缀——它的用途不只是正式研报。问「某公司业绩怎么样」
时模型在对话里直接答一段分析，同样必须先过这一步；带前缀会让模型以为这是
研报专用工具，普通问答就跳过了校验。

返回摘要和人工审核队列，不改写正文，也不生成 PDF。
"""

from pydantic import BaseModel, Field

from ai_agent.prompts.tools import REPORT_VALIDATE_DESC
from ai_agent.report.compose import ReportDocument
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.validate import format_validation, validate_report
from ai_agent.tools.base import Tool


class ValidateFact(BaseModel):
    subject: str = Field(..., description="公司名")
    name: str = Field(..., description="指标名")
    period: str = Field(..., description="报告期，例如 2025Q3")
    value: str = Field(..., description="抽取到的原始数字")
    unit: str = Field(default="", description="单位")
    scope: str = Field(default="", description="口径 consolidated / parent")
    file_id: str = Field(default="", description="文档 ID")
    filename: str = Field(default="", description="文件名")
    page: int | None = Field(default=None, description="页码")
    page_end: int | None = Field(default=None, description="结束页码")
    section: str = Field(default="", description="章节")
    quote: str = Field(default="", description="原文片段，必须含有 value")


class ReportValidateArgs(BaseModel):
    markdown: str = Field(..., description="撰写步骤产出的 Markdown 正文")
    title: str = Field(default="", description="标题，可留空")
    period: str = Field(default="", description="主报告期，用于核对本期指标是否写入正文")
    facts: list[ValidateFact] = Field(..., description="抽取结果。校验只对照这些事实和由其算出的增减。")


def _report_validate(markdown: str, title: str, period: str, facts: list[dict]) -> str:
    rows = [
        IndicatorFact(
            subject=item["subject"],
            period=item["period"],
            name=item["name"],
            value=item["value"],
            unit=item.get("unit") or "",
            column="",
            scope=item.get("scope") or "",
            evidence=Evidence(
                file_id=item.get("file_id") or "",
                filename=item.get("filename") or "",
                page=item.get("page"),
                page_end=item.get("page_end"),
                section=item.get("section") or "",
                quote=item.get("quote") or "",
            ),
        )
        for item in facts
    ]
    document = ReportDocument(
        title=title,
        markdown=markdown,
        html="",
        deliverable="markdown",
    )
    plan = None
    if period:
        from ai_agent.report.parse import ReportPlan

        plan = ReportPlan(title=title, kind="company", deliverable="markdown", period=period)
    return format_validation(validate_report(document, rows, plan=plan))


report_validate_tool = Tool(
    name="validate_figures",
    description=REPORT_VALIDATE_DESC,
    func=_report_validate,
    args_model=ReportValidateArgs,
)
