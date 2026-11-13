import json
from types import UnionType
from typing import Union, get_args, get_origin

from pydantic import ValidationError

from ai_agent.core.logging import get_logger
from ai_agent.core.text import clip_text
from ai_agent.tools.kb import (
    kb_list_files_tool,
    kb_read_tool,
    kb_search_tool,
    kg_search_tool,
)
from ai_agent.tools.memory import (
    memory_expand_tool,
    memory_search_tool,
    memory_write_tool,
)
from ai_agent.tools.misc import (
    read_tool_result_tool,
    weather_tool,
    web_search_tool,
)
from ai_agent.tools.base import ToolOutput
from ai_agent.tools.report import (
    generate_report_tool,
    report_compare_tool,
    report_extract_tool,
    report_job_status_tool,
    report_validate_tool,
)

logger = get_logger(__name__)

# 研报积木（extract / compare / validate）与 generate_report 一起注册：
# 积木无路径参数、不写盘，分别承担「抂数、算增减、校验」三个确定性环节。
# report_pdf 仍不注册——渲染含落盘与 M2 审核门禁，只能由服务端流水线触发。
TOOLS = [
    weather_tool,
    web_search_tool,
    read_tool_result_tool,
    memory_search_tool,
    memory_write_tool,
    kb_list_files_tool,
    kb_search_tool,
    kg_search_tool,
    generate_report_tool,
    report_job_status_tool,
    report_extract_tool,
    report_compare_tool,
    report_validate_tool,
]
# kb_read 恒注册：按「文件名 · 第N页」读知识库当前原文并补上下文/引用。
# 通用工具结果的读回由 read_tool_result 负责（按 tool_call_id 取原消息全文）；
# 两者分工不同，都恒注册，避免老化后的结果无处读回。
from ai_agent.core.config import settings as _settings  # noqa: E402

TOOLS.append(kb_read_tool)
if _settings.kb_search_list_only:
    pass  # 清单模式只是把当前轮的 kb_search 换成"只回清单"，工具本身一样
# 渐进式记忆第 3 步：expand_memory 按需展开被索引省略的条目，恒定注册。
TOOLS.append(memory_expand_tool)
TOOLS_MAP = {t.name: t for t in TOOLS}
_TOOL_SCHEMAS = [t.schema() for t in TOOLS]

# 输出上限按工具语义分级。原先所有工具统一 [:3000] 前缀硬截断：研报的多个
# 下载链接、检索结果靠后的表格行、长 JSON 都可能被从中间切断，且没有任何
# 标记，模型会拿残片当完整结果。返回结构化/链接类结果的工具放宽上限，
# 其余沿用 Tool 声明的默认值；裁剪统一走 clip_text（保头保尾 + 显式标记）。
TOOL_OUTPUT_LIMITS: dict[str, int] = {
    "generate_report": 4000,
    # 任务完成后返回的是一整段带链接的摘要，截断会把深度链接从中间切掉
    "report_job_status": 4000,
    "kb_search": 4000,
    # 两段式里 kb_read 是模型按需取全文，读 2~3 条就 3k+，给宽一点
    "kb_read": 6000,
    "report_extract_indicators": 4000,
    # 展开的是全量画像 + 情景原文，截断会让「未列出的条目」再次消失
    "expand_memory": 4000,
    "read_tool_result": 6000,
}


def _tool_output(
    tool, raw: str, arguments: dict | None = None
) -> ToolOutput:
    limit = TOOL_OUTPUT_LIMITS.get(tool.name, tool.max_output_chars)
    clipped = raw
    if limit > 0 and len(raw) > limit:
        logger.info(
            "工具输出已裁剪 name=%s chars=%d limit=%d",
            tool.name,
            len(raw),
            limit,
        )
        clipped = clip_text(raw, limit)
    preview = (raw.splitlines()[0] if raw.splitlines() else "").strip()
    meta = {
        "tool": tool.name,
        "arguments": arguments or {},
        "raw_chars": len(raw),
        "clipped_chars": len(clipped),
        "truncated": len(clipped) < len(raw),
        "preview": preview[:120],
    }
    return ToolOutput(clipped, raw=raw, meta=meta)


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
        return _tool_output(tool, tool.run(**arguments), arguments)
    except ValidationError:
        # 模型传的参数越界（depth=3 超过 le=2 之类）：夹回边界重试一次，别白费一轮
        try:
            clamped = _clamp_out_of_range_args(tool.args_model, arguments)
            return _tool_output(tool, tool.run(**clamped), clamped)
        except Exception as e:
            logger.warning("工具参数越界后夹取仍失败 name=%s: %s", name, e)
            return f"ERROR: 工具执行失败（{type(e).__name__}）"
    except Exception as e:
        # 异常详情（可能包含 SQL/连接信息）只进日志，不回传给模型
        logger.exception("工具执行失败 name=%s", name)
        return f"ERROR: 工具执行失败（{type(e).__name__}）"
