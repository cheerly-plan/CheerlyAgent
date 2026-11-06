import json
from types import UnionType
from typing import Union, get_args, get_origin

from pydantic import ValidationError

from ai_agent.core.logging import get_logger
from ai_agent.tools.kb_list import kb_list_files_tool
from ai_agent.tools.kb_search import kb_search_tool
from ai_agent.tools.kg_search import kg_search_tool
from ai_agent.tools.memory import memory_search_tool, memory_write_tool
from ai_agent.tools.report_compare import report_compare_tool
from ai_agent.tools.report_extract import report_extract_tool
from ai_agent.tools.report_generate import generate_report_tool
from ai_agent.tools.report_validate import report_validate_tool
from ai_agent.tools.weather import weather_tool
from ai_agent.tools.web_search import web_search_tool

logger = get_logger(__name__)

# 研报积木（extract / compare / validate）与 generate_report 一起注册：
# 积木无路径参数、不写盘，分别承担「抂数、算增减、校验」三个确定性环节。
# report_pdf 仍不注册——渲染含落盘与 M2 审核门禁，只能由服务端流水线触发。
TOOLS = [
    weather_tool,
    web_search_tool,
    memory_search_tool,
    memory_write_tool,
    kb_list_files_tool,
    kb_search_tool,
    kg_search_tool,
    generate_report_tool,
    report_extract_tool,
    report_compare_tool,
    report_validate_tool,
]
TOOLS_MAP = {t.name: t for t in TOOLS}
_TOOL_SCHEMAS = [t.schema() for t in TOOLS]


def tool_schemas() -> list[dict]:
    # 工具 schema 是静态的，模块级预计算，避免每轮对话重复生成
    return list(_TOOL_SCHEMAS)


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


def _expects_list(annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin is list:
        return True
    if origin in (Union, UnionType):
        return any(_expects_list(item) for item in get_args(annotation))
    return False


def _coerce_stringified_lists(args_model: type, arguments: dict) -> dict:
    """模型有时把数组参数收成一段 JSON 字符串，先还原成列表再校验。

    本地模型会写出 ``files: "[\\"年报.pdf\\"]"``。schema 要的是字符串数组，
    原样校验会整次失败，检索还没开始。单个文件名没有方括号时，当成只有一项。
    """
    coerced = dict(arguments)
    for name, field in args_model.model_fields.items():
        value = coerced.get(name)
        if not isinstance(value, str) or not _expects_list(field.annotation):
            continue
        text = value.strip()
        if not text:
            coerced[name] = None
            logger.info("工具参数空数组字符串已忽略：%s.%s", args_model.__name__, name)
            continue
        parsed: object = None
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, list):
            items = [str(item).strip() for item in parsed if str(item).strip()]
        else:
            items = [text]
        coerced[name] = items
        logger.info(
            "工具参数数组字符串已还原：%s.%s -> %s",
            args_model.__name__,
            name,
            items,
        )
    return coerced


def execute_tool(name: str, arguments: dict) -> str:
    tool = TOOLS_MAP.get(name)
    if tool is None:
        return f"ERROR: 未知工具 {name}"
    arguments = _coerce_stringified_lists(tool.args_model, arguments)
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
