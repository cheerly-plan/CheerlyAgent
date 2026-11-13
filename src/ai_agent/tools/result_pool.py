"""工具结果池：按 (user, chat_session, tool_call_id) 保存可回放的工具原文。

为什么需要它：
- ``age_tool_results`` 只替换送模型副本，原始 messages 里还有全文；但
  ``_fit_context`` 整轮裁剪后 messages 里的旧结果也会消失。
- ``read_tool_result`` 需要在这之后仍能按 call_id 读回原文。

实现取舍：
- 进程内 LRU + TTL，按用户/会话隔离；重启后池子为空，回落到 messages/
  checkpoint 里的数据，不把「池子」当成持久化真相。
- ``put`` 保存的是**裁剪前**的原始输出（由 loop 从 ToolOutput.raw 取），
  所以 read_tool_result 不受单工具 3k/6k 输出上限影响；单条超过配置上限
  时截断并在 meta 里标记，避免大表/OCR 把进程内存撑爆。
- 线程安全：聊天工具执行在工作线程，池子用锁 + OrderedDict 做 LRU。
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from time import monotonic
from typing import Callable

from ai_agent.core.config import settings

Key = tuple[str, str, str]


class ToolResultPool:
    def __init__(
        self,
        *,
        max_entries: int = 512,
        ttl_seconds: int = 7200,
        max_content_chars: int = 200_000,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._items: OrderedDict[Key, dict] = OrderedDict()
        self._lock = Lock()
        self._clock = clock
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_content_chars = max(1, int(max_content_chars))

    def put(
        self,
        user_id: str,
        chat_session_id: str,
        tool_call_id: str,
        name: str,
        content: str,
        meta: dict | None = None,
    ) -> None:
        call_id = (tool_call_id or "").strip()
        if not call_id:
            return
        body = content or ""
        truncated = len(body) > self.max_content_chars
        if truncated:
            body = body[: self.max_content_chars]
        key = (user_id or "", chat_session_id or "", call_id)
        with self._lock:
            self._expire()
            self._items[key] = {
                "name": name or "tool",
                "content": body,
                "meta": dict(meta or {}),
                "truncated": truncated,
                "created_at": self._clock(),
            }
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def get(
        self, user_id: str, chat_session_id: str, tool_call_id: str
    ) -> dict | None:
        call_id = (tool_call_id or "").strip()
        if not call_id:
            return None
        key = (user_id or "", chat_session_id or "", call_id)
        with self._lock:
            self._expire()
            item = self._items.get(key)
            if item is None:
                return None
            self._items.move_to_end(key)
            return dict(item)

    def clear(
        self, user_id: str | None = None, chat_session_id: str | None = None
    ) -> int:
        with self._lock:
            if user_id is None:
                count = len(self._items)
                self._items.clear()
                return count
            keys = [
                key
                for key in self._items
                if key[0] == user_id
                and (chat_session_id is None or key[1] == chat_session_id)
            ]
            for key in keys:
                del self._items[key]
            return len(keys)

    def _expire(self) -> None:
        if self.ttl_seconds <= 0:
            return
        now = self._clock()
        stale = [
            key
            for key, item in self._items.items()
            if now - float(item.get("created_at", 0)) >= self.ttl_seconds
        ]
        for key in stale:
            del self._items[key]


tool_result_pool = ToolResultPool(
    max_entries=settings.tool_result_pool_max_entries,
    ttl_seconds=settings.tool_result_pool_ttl_seconds,
    max_content_chars=settings.tool_result_pool_max_content_chars,
)
