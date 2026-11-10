"""SSE 传输层保活：长静默期间输出注释帧，data 事件不丢不重。

研报深度层单次工具调用可跑十几分钟，期间没有 text 也没有 tool_progress；
反代/网关按 idle 计时会掐断连接。心跳放在传输层，用标准 SSE 注释帧，
前端 EventSource 自动忽略，data 事件契约逐字不变。
"""

import asyncio
import json
import time

from ai_agent.api.routers import chat as chat_router
from ai_agent.core.config import settings


def _slow_events(gap: float):
    yield chat_router._sse_frame({"type": "tool_progress", "message": "开始"})
    time.sleep(gap)
    yield chat_router._sse_frame({"type": "tool_progress", "message": "结束"})


def _drain(response) -> list[str]:
    async def collect():
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(collect())


def test_heartbeat_during_long_silence(monkeypatch):
    monkeypatch.setattr(settings, "sse_heartbeat_seconds", 0.03)
    chunks = _drain(chat_router._stream_response(_slow_events(0.18)))
    keepalives = [c for c in chunks if c.startswith(":")]
    data = [c for c in chunks if c.startswith("data: ")]

    assert len(keepalives) >= 2, chunks
    assert all(c == ": keepalive\n\n" for c in keepalives)
    # 心跳不能把已预拉的首个事件吞掉，也不能让后续事件重复
    assert [json.loads(c[6:])["message"] for c in data] == ["开始", "结束"]


def test_heartbeat_disabled_keeps_legacy_behaviour(monkeypatch):
    monkeypatch.setattr(settings, "sse_heartbeat_seconds", 0)
    chunks = _drain(chat_router._stream_response(_slow_events(0.05)))
    assert not any(c.startswith(":") for c in chunks)
    assert len([c for c in chunks if c.startswith("data: ")]) == 2


def test_heartbeat_covers_silence_after_first_event(monkeypatch):
    """首个事件由 handler 同步预拉，之后的静默才是反代掐断的高发段。"""
    monkeypatch.setattr(settings, "sse_heartbeat_seconds", 0.02)

    def events():
        yield chat_router._sse_frame({"type": "run_started"})
        time.sleep(0.09)
        yield chat_router._sse_frame({"type": "done"})

    chunks = _drain(chat_router._stream_response(events()))
    assert any(c == ": keepalive\n\n" for c in chunks)
    assert [json.loads(c[6:])["type"] for c in chunks if c.startswith("data: ")] == [
        "run_started",
        "done",
    ]
