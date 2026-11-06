"""研报工具：生成、抽取指标、对比、校验、导出 PDF。

聊天侧只暴露 ``report_generate``（通用入口，落盘与门禁走服务端）；
``report_extract_indicators`` / ``report_compare`` / ``report_validate`` 是
确定性流水线的积木工具；``report_pdf`` 不注册（PDF 只由 generate 链路产出）。
"""

from ai_agent.tools.report.report_compare import report_compare_tool
from ai_agent.tools.report.report_extract import report_extract_tool
from ai_agent.tools.report.report_generate import generate_report_tool
from ai_agent.tools.report.report_validate import report_validate_tool

__all__ = [
    "report_compare_tool",
    "report_extract_tool",
    "generate_report_tool",
    "report_validate_tool",
]

