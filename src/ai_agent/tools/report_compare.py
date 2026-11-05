"""研报对比工具。模型只传入已经抽好的数字，增减由 compare 计算。"""

from pydantic import BaseModel, Field

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
        )
        for item in points
    ]
    return format_compare(compare_measures(rows))


report_compare_tool = Tool(
    name="report_compare",
    description=(
        "计算已抽取指标的同比、环比、增减额、增减幅度和多期趋势。"
        "必须传入抽取结果里的原始数字，禁止自己心算后再传入。"
        "比率类指标返回百分点，不返回除以基期得到的增速。"
    ),
    func=_report_compare,
    args_model=ReportCompareArgs,
)
