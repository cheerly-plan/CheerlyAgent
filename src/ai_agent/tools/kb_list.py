"""知识库文件清单工具。

替代原「目录注入」：不再把文件名/标签/类型主动塞进 system 上下文，而是让模型在
需要确认库里有哪些文件、各是什么内容时主动调用，再据此决定 kb_search 的 files
参数。文件描述来自 agent_kb_file.description（上传/管理时可维护）。
"""

from pydantic import BaseModel, Field

from ai_agent.memory.store import store
from ai_agent.prompts.tools import KB_LIST_DESC
from ai_agent.tools.base import Tool


class KbListArgs(BaseModel):
    q: str | None = Field(
        default=None,
        description="按文件名关键字过滤；留空则返回全部已就绪（可检索）的文件",
    )


def _kb_list_files(q: str | None = None) -> str:
    rows = store.list_kb_files(q=q.strip() if q else "")
    ready = [r for r in rows if r.get("status") == "ready"]
    if not ready:
        return "知识库当前没有处理完成的文件，请提示用户联系管理员上传资料。"
    lines = ["知识库当前已就绪的文件："]
    for r in ready:
        name = str(r.get("filename") or "(未命名)").strip()
        desc = str(r.get("description") or "").strip()
        line = f"- {name}"
        if desc:
            line += f"：{desc}"
        lines.append(line)
    return "\n".join(lines)


kb_list_files_tool = Tool(
    name="kb_list_files",
    display_name="知识库文件清单",
    description=KB_LIST_DESC,
    func=_kb_list_files,
    args_model=KbListArgs,
)
