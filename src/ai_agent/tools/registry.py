from ai_agent.tools.memory import memory_search_tool, memory_write_tool
from ai_agent.tools.weather import weather_tool

TOOLS = [weather_tool, memory_search_tool, memory_write_tool]   
TOOLS_MAP = {t.name: t for t in TOOLS}


def schemas() -> list[dict]:
    return [t.schema() for t in TOOLS]


def execute(name: str, arguments: dict) -> str:
    tool = TOOLS_MAP.get(name)
    if tool is None:
        return f"ERROR: 未知工具 {name}"
    try:
        return tool.run(**arguments)[:3000]
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"