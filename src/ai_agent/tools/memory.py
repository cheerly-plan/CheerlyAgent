from typing import Literal

from pydantic import BaseModel, Field

from ai_agent.memory.profile import apply_profile_text
from ai_agent.memory.store import store
from ai_agent.prompts.tools import (
    MEMORY_SEARCH_DESC,
    MEMORY_SEARCH_QUERY_DESC,
    MEMORY_WRITE_CONTENT_DESC,
    MEMORY_WRITE_DESC,
    MEMORY_WRITE_KIND_DESC,
)
from ai_agent.tools.base import Tool


class MemorySearchArgs(BaseModel):
    query: str = Field(description=MEMORY_SEARCH_QUERY_DESC)


class MemoryWriteArgs(BaseModel):
    content: str = Field(description=MEMORY_WRITE_CONTENT_DESC)
    kind: Literal["profile", "episode"] = Field(
        default="profile",
        description=MEMORY_WRITE_KIND_DESC,
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
    display_name="记忆检索",
    description=MEMORY_SEARCH_DESC,
    func=search_memory,
    args_model=MemorySearchArgs,
)

memory_write_tool = Tool(
    name="write_memory",
    display_name="记忆写入",
    description=MEMORY_WRITE_DESC,
    func=write_memory,
    args_model=MemoryWriteArgs,
)
