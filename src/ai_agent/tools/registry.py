from ai_agent.core.logging import get_logger
from ai_agent.tools.memory import memory_search_tool, memory_write_tool
from ai_agent.tools.weather import weather_tool
from ai_agent.tools.web_search import web_search_tool

logger = get_logger(__name__)

TOOLS = [weather_tool, web_search_tool, memory_search_tool, memory_write_tool]
TOOLS_MAP = {t.name: t for t in TOOLS}
_TOOL_SCHEMAS = [t.schema() for t in TOOLS]


def tool_schemas() -> list[dict]:
    # 工具 schema 是静态的，模块级预计算，避免每轮对话重复生成
    return list(_TOOL_SCHEMAS)


def tool_label(name: str) -> str:
    """工具的用户可读展示名（未注册的工具回退为工具名本身）。"""
    tool = TOOLS_MAP.get(name)
    return tool.display_name if tool else name


def execute_tool(name: str, arguments: dict) -> str:
    tool = TOOLS_MAP.get(name)
    if tool is None:
        return f"ERROR: 未知工具 {name}"
    try:
        return tool.run(**arguments)[:3000]
    except Exception as e:
        # 异常详情（可能包含 SQL/连接信息）只进日志，不回传给模型
        logger.exception("工具执行失败 name=%s", name)
        return f"ERROR: 工具执行失败（{type(e).__name__}）"
