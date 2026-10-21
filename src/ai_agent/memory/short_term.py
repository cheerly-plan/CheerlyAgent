from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic

from ai_agent.core.config import settings
from ai_agent.llm.types import ChatMessage


def compact(messages: list[ChatMessage], keep: int) -> list[ChatMessage]:
    if not messages:
        return []
    system: list[ChatMessage] = []
    rest = messages
    if messages[0]["role"] == "system":
        system = [messages[0]]
        rest = messages[1:]
    recent = rest[-keep:] if keep >= 0 else rest
    start = 0
    while start < len(recent) and recent[start].get("role") == "tool":
        start += 1
    return [*system, *recent[start:]]


@dataclass(slots=True)
class _Entry:
    messages: list[ChatMessage]
    used_at: float


class ShortTermMemory:
    def __init__(
        self,
        max_turns: int = settings.short_term_memory_max_turns,
        max_sessions: int = settings.short_term_memory_max_sessions,
        ttl_seconds: int = settings.short_term_memory_ttl_seconds,
        cleanup_interval_seconds: int = settings.short_term_memory_cleanup_interval_seconds,
        clock: Callable[[], float] = monotonic,
    ):
        self._sessions: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self._lock = Lock()
        self._stop = Event()
        self._cleaner: Thread | None = None
        self._clock = clock
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds

    def load(self, chat_session_id: str, user_id: str = "default") -> list[ChatMessage]:
        key = (user_id, chat_session_id)
        with self._lock:
            self._expire()
            entry = self._sessions.get(key)
            if entry is None:
                return []
            entry.used_at = self._clock()
            self._sessions.move_to_end(key)
            return list(entry.messages)

    def save(
        self,
        chat_session_id: str,
        messages: list[ChatMessage],
        user_id: str = "default",
    ) -> None:
        key = (user_id, chat_session_id)
        compacted = compact(messages, self.max_turns)
        with self._lock:
            self._sessions[key] = _Entry(compacted, self._clock())
            self._sessions.move_to_end(key)
            self._evict()

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def start_cleanup(self) -> None:
        if self.ttl_seconds <= 0 or self.cleanup_interval_seconds <= 0:
            return
        if self._cleaner is not None and self._cleaner.is_alive():
            return
        self._stop.clear()
        self._cleaner = Thread(
            target=self._cleanup_loop,
            name="short-term-cleanup",
            daemon=True,
        )
        self._cleaner.start()

    def stop_cleanup(self) -> None:
        self._stop.set()
        cleaner = self._cleaner
        if cleaner is not None:
            cleaner.join(timeout=1)
        self._cleaner = None

    def _cleanup_loop(self) -> None:
        interval = self.cleanup_interval_seconds
        while not self._stop.wait(interval):
            with self._lock:
                self._expire()

    def _expire(self) -> None:
        ttl = self.ttl_seconds
        if ttl <= 0:
            return
        now = self._clock()
        stale = [
            key
            for key, entry in self._sessions.items()
            if now - entry.used_at >= ttl
        ]
        for key in stale:
            del self._sessions[key]

    def _evict(self) -> None:
        self._expire()
        limit = self.max_sessions
        if limit <= 0:
            return
        while len(self._sessions) > limit:
            self._sessions.popitem(last=False)


short_term_memory = ShortTermMemory()
