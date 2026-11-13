"""外部能力工具：天气、联网搜索。

不依赖项目内业务域，只用到 core 配置与提示词模板；与知识库/记忆/研报正交。
"""

from ai_agent.tools.misc.history import read_history_tool, search_history_tool
from ai_agent.tools.misc.tool_result import read_tool_result_tool
from ai_agent.tools.misc.weather import weather_tool
from ai_agent.tools.misc.web_search import web_search_tool

__all__ = [
    "read_history_tool",
    "read_tool_result_tool",
    "search_history_tool",
    "weather_tool",
    "web_search_tool",
]

