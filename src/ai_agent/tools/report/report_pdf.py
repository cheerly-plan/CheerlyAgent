"""研报 PDF 工具。模型只传入已写好的 HTML，文件由 WeasyPrint 写到磁盘。

返回值是路径和页数，不是 PDF 字节。
"""

from pathlib import Path

from pydantic import BaseModel, Field

from ai_agent.report.compose import ReportDocument
from ai_agent.report.pdf import render_report_pdf
from ai_agent.tools.base import Tool


class ReportPdfArgs(BaseModel):
    html: str = Field(..., description="撰写步骤产出的 HTML。禁止传入 PDF 二进制。")
    title: str = Field(default="研报", description="报告标题，用于页眉和文件名")
    output_path: str = Field(..., description="PDF 落盘路径。工具不把文件内容返回给模型。")


def _report_pdf(html: str, title: str, output_path: str) -> str:
    if html.lstrip().startswith("%PDF"):
        raise ValueError("传入的是 PDF 二进制，已拒绝")
    document = ReportDocument(title=title, markdown="", html=html, deliverable="pdf")
    pdf = render_report_pdf(document)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf.content)
    return f"已生成 PDF：{path}，共 {pdf.pages} 页"


report_pdf_tool = Tool(
    name="report_pdf",
    description=(
        "把已经写好的研报 HTML 排成 PDF。"
        "页眉页脚、字体和分页由渲染器处理。"
        "禁止自己生成或拼接 PDF 二进制；只传入 HTML，并读取返回的文件路径。"
    ),
    func=_report_pdf,
    args_model=ReportPdfArgs,
)
