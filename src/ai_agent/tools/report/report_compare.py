"""研报对比工具。模型只传入已经抽好的数字，增减由 compare 计算。"""

from pydantic import BaseModel, Field

from ai_agent.prompts.tools import REPORT_COMPARE_DESC
from ai_agent.report.compare import Measure, compare_measures, format_compare
from ai_agent.tools.base import Tool


class ComparePoint(BaseModel):
    subject: str = Field(..., description="公司名，与抽取结果一致")
    name: str = Field(..., description="指标名，例如营业收入、毛利率")
    period: str = Field(..., description="报告期，例如 2025Q3、2025H1、2025")
    value: str = Field(..., description="原文中的数字，不要自行换算或四舍五入")
    unit: str = Field(default="", description="原文单位，例如 元、万元、%。没有就留空")
    scope: str = Field(
        default="",
        description="口径：consolidated 或 parent。不同口径不要混在同一条里比较",
    )
    cumulative: bool = Field(
        default=False,
        description="该值是否取自累计列（年初至报告期末）。单季与累计不可混在一条序列",
    )
    basis: str = Field(
        default="",
        description="口径代码（report_extract_indicators 返回）：cumulative/single/point/stock",
    )
    coverage_months: int = Field(
        default=0,
        description="覆盖月数（累计口径必填）：Q1=3、H1=6、Q3=9、全年=12",
    )


class ReportCompareArgs(BaseModel):
    points: list[ComparePoint] = Field(
        ...,
        description="同一指标的多期数字。只填抽取结果里出现过的值。",
    )


def _report_compare(points: list[dict]) -> str:
    rows = [
        Measure(
            subject=item["subject"],
            name=item["name"],
            period=item["period"],
            value=item["value"],
            unit=item.get("unit") or "",
            scope=item.get("scope") or "",
            cumulative=bool(item.get("cumulative", False)),
            basis=item.get("basis") or "",
            coverage_months=int(item.get("coverage_months") or 0),
        )
        for item in points
    ]
    return format_compare(compare_measures(rows))


report_compare_tool = Tool(
    name="report_compare",
    description=REPORT_COMPARE_DESC,
    func=_report_compare,
    args_model=ReportCompareArgs,
)
