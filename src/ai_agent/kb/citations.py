"""检索引用收集：把 kb_search 命中的切片「旁路」交给上层（SSE + 落库）。

工具函数只能返回字符串（那是给模型看的正文），引用结构没法从返回值里带出来，
又不能让模型自己吐 JSON 再解析（不稳定、还占 token）。这里用一个 ContextVar
做旁路：loop 执行工具前开一个收集器，工具往里塞引用，执行完取走。

为什么是 ContextVar 而不是全局列表：聊天并发时两个会话可能同时执行工具，
全局列表会把两边的引用串在一起。ContextVar 按执行上下文隔离，天然安全；
而且设置点必须和工具执行同栈（loop 里 execute_tool 那一层），
放到更外层会被生成器跨线程推进打散。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

# 单条引用保留的正文长度：切片本体约 500 字，留点余量基本能整段存下，
# 前端点开就能看到原文，不必再回查接口
_EXCERPT_CLIP = 600
# 表格按行切过，整表才渲染得出来；600 字会切到半行，竖线对不上列。
# 上限放宽，超了也按完整行截，避免 Markdown 表格残缺。
_TABLE_EXCERPT_CLIP = 2500

# 一条回答最多记这么多条引用：工具循环可能被反复调用，超过这个数量
# 说明检索在发散，前端列表也放不下
MAX_SOURCES = 12

_COLLECTOR: ContextVar[list[Citation] | None] = ContextVar(
    "kb_citations", default=None
)


def _clip_excerpt(text: str, *, kind: str) -> str:
    body = (text or "").strip()
    limit = _TABLE_EXCERPT_CLIP if kind == "table" else _EXCERPT_CLIP
    if len(body) <= limit:
        return body
    if kind == "table":
        # 半行竖线会把表格渲染拆烂，退到上一个换行
        cut = body[:limit]
        nl = cut.rfind("\n")
        if nl >= 8:
            cut = cut[:nl]
        return f"{cut}\n…"
    return f"{body[:limit]}…"


@dataclass
class Citation:
    """一条可展示的引用：定位信息 + 摘录正文。"""

    file_id: str
    filename: str
    seq: int
    page: int | None
    kind: str
    where: str
    text: str
    # 跨页切片的末页；单页片等于 page（前端按 page_start==page_end 决定
    # 显示「P134」还是「P134–P135」）
    page_end: int | None = None
    # 所属一级节，如「第十节 管理层讨论与分析」
    section: str | None = None

    @property
    def key(self) -> str:
        # 同一份文件的同一个切片只保留一次（多次检索常常命中同一片）
        return f"{self.file_id}#{self.seq}"

    def payload(self) -> dict:
        excerpt = _clip_excerpt(self.text, kind=self.kind)
        return {
            "key": self.key,
            "file_id": self.file_id,
            "filename": self.filename,
            "seq": self.seq,
            # page 保留给旧前端；新前端用 page_start/page_end 渲染区间
            "page": self.page,
            "page_start": self.page,
            "page_end": self.page_end if self.page_end is not None else self.page,
            "section": self.section,
            "kind": self.kind,
            "where": self.where,
            "excerpt": excerpt,
        }


@contextmanager
def collect_citations() -> Iterator[list[Citation]]:
    """在工具执行期间收集引用；退出时恢复上一层（支持嵌套调用）。"""
    bucket: list[Citation] = []
    token = _COLLECTOR.set(bucket)
    try:
        yield bucket
    finally:
        _COLLECTOR.reset(token)


def add_citations(items: Iterable[Citation]) -> None:
    """工具侧调用：把引用塞进当前收集器。没有收集器时静默丢弃
    （例如单测直接调工具、或未来在别的上下文里复用检索）。"""
    bucket = _COLLECTOR.get()
    if bucket is None:
        return
    bucket.extend(items)


def merge_sources(current: list[dict], fresh: Iterable[Citation]) -> list[dict]:
    """把新收集到的引用合并进本轮的累积列表（按 key 去重，保序，封顶）。"""
    seen = {item["key"] for item in current}
    merged = list(current)
    for citation in fresh:
        payload = citation.payload()
        if payload["key"] in seen:
            continue
        seen.add(payload["key"])
        merged.append(payload)
    return merged[:MAX_SOURCES]
