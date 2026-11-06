"""工具执行中的进度回传（给 SSE tool_progress 用）。

长工具（如 generate_report 等入库）在同线程里阻塞时，Agent 循环读不到
中间态；循环把工具丢到工作线程后，通过本模块的回调把「正在解析哪几份」
推回主循环再 yield 给前端。
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from collections.abc import Callable

ProgressFn = Callable[[str], None]

_progress: ContextVar[ProgressFn | None] = ContextVar("tool_progress", default=None)


def set_progress(fn: ProgressFn | None) -> Token:
    return _progress.set(fn)


def reset_progress(token: Token) -> None:
    _progress.reset(token)


def emit_progress(message: str) -> None:
    text = (message or "").strip()
    if not text:
        return
    fn = _progress.get()
    if fn is not None:
        fn(text)
