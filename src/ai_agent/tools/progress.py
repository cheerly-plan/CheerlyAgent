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


def current_progress() -> ProgressFn | None:
    """取当前线程绑定的进度回调。

    ContextVar 不跨线程继承：深层并发（如深度分析章节/评审 worker）要推进度，
    必须在主线程先取出回调再显式传进去。回调本身约定线程安全（聊天侧是
    ``queue.put``，任务表侧有锁），所以 worker 直接调用即可。
    """
    return _progress.get()


def emit_progress(message: str) -> None:
    text = (message or "").strip()
    if not text:
        return
    fn = _progress.get()
    if fn is not None:
        fn(text)
