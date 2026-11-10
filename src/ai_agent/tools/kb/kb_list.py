"""知识库文件清单工具。

替代原「目录注入」：不再把文件名/标签/类型主动塞进 system 上下文，而是让模型在
需要确认库里有哪些文件、各是什么内容时主动调用，再据此决定 kb_search 的 files
参数。文件描述来自 agent_kb_file.description（上传/管理时可维护）。

文件多时由本工具自己限长：一次最多返回 _LIST_LIMIT 条，避免被 execute_tool
的全局截断从某一行中间切断。
"""

from pydantic import BaseModel, Field

from ai_agent.storage.store import store
from ai_agent.prompts.tools import KB_LIST_DESC
from ai_agent.tools.base import Tool

# 与系统提示词里「小库全列」的阈值对齐：再多就要求带关键字，而不是把全库倒出来
_LIST_LIMIT = 30


class KbListArgs(BaseModel):
    q: str | None = Field(
        default=None,
        description=(
            "关键字，匹配文件名、主体名或内容描述。"
            "文件较多时必须填写；留空且超过 30 份时只返回总数，不列出文件名。"
        ),
    )


def _kb_list_files(q: str | None = None) -> str:
    needle = (q or "").strip()
    # 过滤放在工具层：store 的 q 只匹配文件名，主体名和描述要在这里一起看
    rows = store.list_kb_files()
    ready = [r for r in rows if r.get("status") == "ready"]
    if needle:
        ready = [r for r in ready if _matches(r, needle)]
    if not ready:
        if needle:
            return (
                "没有文件名、主体或描述匹配该关键字的已就绪文件。"
                "请换个关键字，或直接把主体名填进 kb_search 的 files。"
            )
        return "知识库当前没有处理完成的文件，请提示用户联系管理员上传资料。"
    if not needle and len(ready) > _LIST_LIMIT:
        return (
            f"知识库当前有 {len(ready)} 份已就绪文件，数量较多，未全部列出。"
            "请用 q 传文件名、主体名或内容描述里的关键字再查；"
            "知道主体时也可以直接把主体名填进 kb_search 的 files。"
        )
    shown = ready[:_LIST_LIMIT]
    omitted = len(ready) - len(shown)
    lines = ["知识库当前已就绪的文件："]
    lines.extend(_format_line(r) for r in shown)
    if omitted:
        lines.append(f"还有 {omitted} 份未列出，请把关键字写窄。")
    return "\n".join(lines)


def _matches(row: dict, needle: str) -> bool:
    """文件名、主体、别名、描述任一包含关键字即命中。比较不区分大小写。"""
    key = needle.casefold()
    fields = [
        str(row.get("filename") or ""),
        str(row.get("subject") or ""),
        str(row.get("description") or ""),
    ]
    aliases = row.get("subject_aliases") or []
    if isinstance(aliases, (list, tuple)):
        fields.extend(str(alias) for alias in aliases)
    return any(key in field.casefold() for field in fields)


def _format_line(row: dict) -> str:
    name = str(row.get("filename") or "(未命名)").strip()
    desc = str(row.get("description") or "").strip()
    period = str(row.get("period") or "").strip()
    line = f"- {name}"
    if period:
        line += f"（报告期 {period}）"
    if desc:
        line += f"：{desc}"
    return line


kb_list_files_tool = Tool(
    name="kb_list_files",
    description=KB_LIST_DESC,
    func=_kb_list_files,
    args_model=KbListArgs,
)
