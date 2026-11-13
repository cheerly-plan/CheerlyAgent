"""工具结果读取的运行时上下文。

老化（``agents/messages.py::age_tool_results``）只替换**送模型副本**里的旧工具
原文；原始 ``messages`` 仍然完整。通用工具 ``read_tool_result`` 需要按
``tool_call_id`` 从原始消息里取回全文，但工具函数签名拿不到消息列表，所以
由 run_loop 在每次执行工具前把「call_id -> {name, content}」放进 ContextVar。

只覆盖当前上下文里的工具结果：更早、已经被整轮裁掉的结果不在 messages 里，
模型手上也不会留下引用，工具会明确提示重新查询，而不是去猜。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator

from ai_agent.llm.types import ChatMessage

ToolResult = dict[str, str]

_tool_results: ContextVar[dict[str, ToolResult] | None] = ContextVar(
    "tool_results", default=None
)
_current_session: ContextVar[str] = ContextVar("tool_result_session", default="")


def tool_result_index(messages: list[ChatMessage]) -> dict[str, ToolResult]:
    """从消息列表建 ``tool_call_id -> {name, content}`` 索引。

    原始消息里的 content 是工具刚返回的完整文本；老化只发生在送模型的副本
    上，所以这里取到的一定是全量原文。
    """
    out: dict[str, ToolResult] = {}
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        call_id = str(msg.get("tool_call_id") or "").strip()
        if not call_id:
            continue
        out[call_id] = {
            "name": str(msg.get("name") or "tool"),
            "content": str(msg.get("content") or ""),
        }
    return out


@contextmanager
def tool_results_scope(results: dict[str, ToolResult]) -> Iterator[None]:
    token = _tool_results.set(results)
    try:
        yield
    finally:
        _tool_results.reset(token)


@contextmanager
def session_scope(chat_session_id: str) -> Iterator[None]:
    """给 read_tool_result 的池子查找提供当前会话 id。"""
    token = _current_session.set(chat_session_id or "")
    try:
        yield
    finally:
        _current_session.reset(token)


def current_session_id() -> str:
    return _current_session.get()


def get_tool_result(call_id: str) -> ToolResult | None:
    results = _tool_results.get()
    if not results:
        return None
    return results.get((call_id or "").strip())