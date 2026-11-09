"""工具执行中的进度/结构化事件回传（给 SSE tool_progress 与自定义事件用）。

长工具（如 generate_report 等入库）在同线程里阻塞时，Agent 循环读不到
中间态；循环把工具丢到工作线程后，通过本模块的回调把「正在解析哪几份」
推回主循环再 yield 给前端。

``emit_event`` 是同一通道的结构化出口：工具把自己产生的、前端要单独渲染的
事件（如研报后台任务 ``{"type": "report_job", "job_id": …}``）原样抛出去，
而不是塞进进度文案里让前端正则去猜。
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from collections.abc import Callable

ProgressFn = Callable[[str], None]
EventFn = Callable[[dict], None]

_progress: ContextVar[ProgressFn | None] = ContextVar("tool_progress", default=None)
_event: ContextVar[EventFn | None] = ContextVar("tool_event", default=None)


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


def set_event_emitter(fn: EventFn | None) -> Token:
    return _event.set(fn)


def reset_event_emitter(token: Token) -> None:
    _event.reset(token)


def emit_event(event: dict) -> None:
    """把一条结构化事件交给 Agent 循环（没有订阅者时静默丢弃）。"""
    if not isinstance(event, dict) or not event.get("type"):
        return
    fn = _event.get()
    if fn is not None:
        fn(event)


def emit_progress(message: str) -> None:
    text = (message or "").strip()
    if not text:
        return
    fn = _progress.get()
    if fn is not None:
        fn(text)
