from pydantic import ValidationError

from ai_agent.core.logging import get_logger
from ai_agent.tools.kb_search import kb_search_tool
from ai_agent.tools.kg_search import kg_search_tool
from ai_agent.tools.memory import memory_search_tool, memory_write_tool
from ai_agent.tools.weather import weather_tool
from ai_agent.tools.web_search import web_search_tool

logger = get_logger(__name__)

TOOLS = [
    weather_tool,
    web_search_tool,
    memory_search_tool,
    memory_write_tool,
    kb_search_tool,
    kg_search_tool,
]
TOOLS_MAP = {t.name: t for t in TOOLS}
_TOOL_SCHEMAS = [t.schema() for t in TOOLS]


def tool_schemas() -> list[dict]:
    # 工具 schema 是静态的，模块级预计算，避免每轮对话重复生成
    return list(_TOOL_SCHEMAS)


def tool_label(name: str) -> str:
    """工具的用户可读展示名（未注册的工具回退为工具名本身）。"""
    tool = TOOLS_MAP.get(name)
    return tool.display_name if tool else name


def _clamp_out_of_range_args(args_model: type, arguments: dict) -> dict:
    """把越界的数值参数夹回 schema 的合法边界，避免模型偶尔传个 depth=3 就整次失败。

    本地小模型对工具参数约束的服从度不稳定（实测传 depth=3 而 schema 上限是 2），
    直接 ValidationError 会浪费一整轮调用。这里按字段声明的 ge/le 约束做一次
    兜底夹取；只夹「数值」且「有边界声明」的字段，其余（类型不对、缺必填项）
    保持原样，让后续校验继续按老逻辑报错。
    """
    schema = args_model.model_json_schema().get("properties", {})
    clamped = dict(arguments)
    for name, meta in schema.items():
        value = clamped.get(name)
        if not isinstance(value, (int, float)):
            continue
        changed = False
        if isinstance(meta.get("minimum"), (int, float)) and value < meta["minimum"]:
            clamped[name] = meta["minimum"]
            changed = True
        if isinstance(meta.get("maximum"), (int, float)) and value > meta["maximum"]:
            clamped[name] = meta["maximum"]
            changed = True
        if changed:
            logger.info(
                "工具参数越界已夹回：%s %s=%r -> %r", args_model.__name__, name, value, clamped[name]
            )
    return clamped


def execute_tool(name: str, arguments: dict) -> str:
    tool = TOOLS_MAP.get(name)
    if tool is None:
        return f"ERROR: 未知工具 {name}"
    try:
        return tool.run(**arguments)[:3000]
    except ValidationError:
        # 模型传的参数越界（depth=3 超过 le=2 之类）：夹回边界重试一次，别白费一轮
        try:
            clamped = _clamp_out_of_range_args(tool.args_model, arguments)
            return tool.run(**clamped)[:3000]
        except Exception as e:
            logger.warning("工具参数越界后夹取仍失败 name=%s: %s", name, e)
            return f"ERROR: 工具执行失败（{type(e).__name__}）"
    except Exception as e:
        # 异常详情（可能包含 SQL/连接信息）只进日志，不回传给模型
        logger.exception("工具执行失败 name=%s", name)
        return f"ERROR: 工具执行失败（{type(e).__name__}）"
