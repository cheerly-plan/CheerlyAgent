from typing import Literal

from pydantic import BaseModel, Field

from ai_agent.memory.profile import apply_profile_text
from ai_agent.memory.store import store
from ai_agent.tools.base import Tool


class MemorySearchArgs(BaseModel):
    query: str = Field(description="要检索的关键词或问题")


class MemoryWriteArgs(BaseModel):
    content: str = Field(
        description="一条只写一件事。身份/偏好用完整陈述，不要把当前任务写进来。"
    )
    kind: Literal["profile", "episode"] = Field(
        default="profile",
        description=(
            "profile: 长期成立的姓名、职业、身份、语言、偏好、约束。\n"
            "episode: 仅当用户明确要求记住某件近况时使用。"
        ),
    )


def search_memory(query: str) -> str:
    hits = store.search_episode(query)
    if not hits:
        return "没有找到相关情景记忆"
    lines = [f"- {hit.text}" for hit in hits]
    return "相关情景记忆：\n" + "\n".join(lines)


def write_memory(content: str, kind: str = "profile") -> str:
    if kind == "profile":
        return apply_profile_text(content)
    item = store.write_episode(content)
    return f"已写入情景记忆: {item.text}"


memory_search_tool = Tool(
    name="search_memory",
    description=(
        "检索当前用户的跨会话情景记忆，不是当前对话。"
        "【触发条件】用户询问过去的事情、需要历史细节或过往结论，且当前上下文不够时调用。"
        "【禁止条件】系统已注入的用户画像、当前会话里已有的内容，不要再搜。"
    ),
    func=search_memory,
    args_model=MemorySearchArgs,
)

memory_write_tool = Tool(
    name="write_memory",
    description=(
        "仅在用户明确要求记住，或清楚陈述姓名、职业、身份、长期偏好时写入。"
        "当前任务、正在学习或讨论的技术方案不要在此写入，系统会在本轮结束后自动抽取。"
        "一条只写一件事：身份/偏好用 profile；只有用户明确要求记住近况时才用 episode。"
        "例如「我是画家，请记住」只写 profile='我是画家'。"
        "【禁止条件】用户只是提问、要建议、讨论方案时，绝对不要调用。"
    ),
    func=write_memory,
    args_model=MemoryWriteArgs,
)
