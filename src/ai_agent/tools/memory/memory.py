from typing import Literal

from pydantic import BaseModel, Field

from ai_agent.memory.progressive import relevance
from ai_agent.memory.profile import apply_profile_text
from ai_agent.storage.store import store
from ai_agent.prompts.tools import (
    MEMORY_EXPAND_DESC,
    MEMORY_EXPAND_KIND_DESC,
    MEMORY_EXPAND_QUERY_DESC,
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


class MemoryExpandArgs(BaseModel):
    query: str = Field(description=MEMORY_EXPAND_QUERY_DESC)
    kind: Literal["profile", "episode", "all"] = Field(
        default="all",
        description=MEMORY_EXPAND_KIND_DESC,
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


def expand_memory(query: str, kind: str = "all") -> str:
    """按需展开记忆全文，补齐【记忆索引】里被省略的画像条目。

    与 ``search_memory`` 的关键区别：后者只查 episode，查不到 profile；
    这里两条都查。profile 走 ``list_profile`` 拿全量后按确定性相关性排序
    （不调向量库），episode 走既有的 ``search_episode``。

    两段各自 fail-open：一段读取失败只标注该段不可用，不影响另一段，
    也不让整次工具调用抛异常打断 agent 循环。
    """
    parts: list[str] = []
    if kind in ("profile", "all"):
        try:
            items = store.list_profile()
            ranked = sorted(items, key=lambda item: -relevance(query, item))
        except Exception:
            parts.append("用户画像读取暂不可用")
        else:
            if ranked:
                body = "\n".join(
                    f"- [{item.id}] {item.text}" for item in ranked
                )
                parts.append(f"完整用户画像条目（按相关性排序，共 {len(ranked)} 条）：\n{body}")
    if kind in ("episode", "all"):
        try:
            hits = store.search_episode(query)
        except Exception:
            parts.append("情景记忆检索暂不可用")
        else:
            if hits:
                body = "\n".join(f"- [{hit.id}] {hit.text}" for hit in hits)
                parts.append(f"相关情景记忆全文（共 {len(hits)} 条）：\n{body}")
    # 第 5 步：L4 归档只在显式展开时出现，永不自动进 system 人格块。
    if kind == "all":
        try:
            from ai_agent.memory.archive import search_archive

            entries = search_archive(query)
        except Exception:
            parts.append("L4 归档读取暂不可用")
        else:
                if entries:
                    body = "\n".join(
                        f"- [{entry.get('id', '')}] {entry.get('text', '')}"
                        for entry in entries
                    )
                    parts.append(f"L4 归档（按需展开，共 {len(entries)} 条）：\n{body}")
    return "\n\n".join(parts) or "没有找到可展开的记忆条目"


memory_search_tool = Tool(
    name="search_memory",
    description=MEMORY_SEARCH_DESC,
    func=search_memory,
    args_model=MemorySearchArgs,
)

memory_write_tool = Tool(
    name="write_memory",
    description=MEMORY_WRITE_DESC,
    func=write_memory,
    args_model=MemoryWriteArgs,
)

memory_expand_tool = Tool(
    name="expand_memory",
    description=MEMORY_EXPAND_DESC,
    func=expand_memory,
    args_model=MemoryExpandArgs,
)
