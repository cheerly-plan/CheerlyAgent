from typing import Literal

from pydantic import BaseModel, Field

from ai_agent.memory.store import store
from ai_agent.tools.base import Tool


class MemorySearchArgs(BaseModel):
    query: str = Field(description="要检索的关键词或问题")


class MemoryWriteArgs(BaseModel):
    content: str = Field(
        description="精炼的记忆内容。一条只写一件事，禁止将身份和近况混写。"
    )
    kind: Literal["profile", "episode"] = Field(
        default="episode",
        description=(
            "profile: 长期成立的姓名、职业、身份、语言、偏好、约束。\n"
            "episode: 当前阶段、正在做的任务、近期事件。\n"
            "拿不准时用 episode。"
        ),
    )


def search_memory(query: str) -> str:
    hits = store.search(query)
    if not hits:
        return "没有找到相关情景记忆"
    lines = [f"- {hit.text}" for hit in hits]
    return "相关情景记忆：\n" + "\n".join(lines)


def write_memory(content: str, kind: str = "episode") -> str:
    # 【代码层兜底建议】：如果大模型偶尔不听话，把长句塞进 content
    # 可以在这里加个简单的拦截（根据你的实际业务决定是否开启）
    # if len(content) > 50 and ("，" in content or "," in content):
    #     return "Error: 记忆内容过长或包含多个事件，请务必拆分成多次调用 write_memory。"

    if kind == "profile":
        item = store.write_profile(content)
        return f"已写入用户画像: {item.text}"
    item = store.write(content)
    return f"已写入情景记忆: {item.text}"


memory_search_tool = Tool(
    name="search_memory",
    description=(
        "检索当前用户的历史记忆。"
        "【触发条件】用户询问过去的事情、需要历史细节或过往结论时调用。"
        "【禁止条件】当前系统已注入用户画像，或上下文信息已充足时，不要调用。"
    ),
    func=search_memory,
    args_model=MemorySearchArgs,
)

memory_write_tool = Tool(
    name="write_memory",
    description=(
        "写入当前用户的长期记忆。"
        "【核心规则】一条消息同时包含身份和近况时，必须调用两次并按 kind 拆开，禁止整句写入。"
        "例如「我是画家，最近在画星河」：必须调两次，一次写 profile='我是画家'，一次写 episode='正在画星河'。"
        "【触发条件】用户陈述姓名、职业、身份、长期偏好，即使没说「请记住」也要写。"
        "【禁止条件】用户只是在提问或要建议，且没有陈述新的身份/偏好时，绝对不要写入。"
    ),
    func=write_memory,
    args_model=MemoryWriteArgs,
)
