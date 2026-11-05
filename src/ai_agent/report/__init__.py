"""研报流水线。抽取指标，再用代码计算同比、环比和趋势。"""

from ai_agent.report.compare import compare_measures
from ai_agent.report.extract import extract_indicators

__all__ = ["compare_measures", "extract_indicators"]
