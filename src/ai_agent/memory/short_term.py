from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic

from ai_agent.core.config import settings
from ai_agent.llm.types import ChatMessage

SUMMARY_USER_PREFIX = "此前对话摘要：\n"
SUMMARY_ASSISTANT = "已了解此前对话摘要，会据此继续当前会话。"
_CRUDE_SUMMARY_MAX_CHARS = 2000


def take_system(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    if messages and messages[0].get("role") == "system":
        return [messages[0]], messages[1:]
    return [], messages


def take_summary(
    messages: list[ChatMessage],
) -> tuple[str, list[ChatMessage]]:
    _, rest = take_system(messages)
    if (
        len(rest) >= 2
        and rest[0].get("role") == "user"
        and str(rest[0].get("content") or "").startswith(SUMMARY_USER_PREFIX)
        and rest[1].get("role") == "assistant"
    ):
        summary = str(rest[0].get("content") or "")[len(SUMMARY_USER_PREFIX) :]
        return summary.strip(), rest[2:]
    return "", rest


def iter_turns(messages: list[ChatMessage]) -> list[list[ChatMessage]]:
    turns: list[list[ChatMessage]] = []
    current: list[ChatMessage] = []
    for msg in messages:
        if msg.get("role") == "user":
            if current and current[0].get("role") == "user":
                turns.append(current)
            current = [msg]
            continue
        if current:
            current.append(msg)
    if current and current[0].get("role") == "user":
        turns.append(current)
    return turns


def flatten(turns: list[list[ChatMessage]]) -> list[ChatMessage]:
    return [msg for turn in turns for msg in turn]


def compact(messages: list[ChatMessage], keep: int) -> list[ChatMessage]:
    system, rest = take_system(messages)
    _, rest = take_summary(rest)
    turns = iter_turns(rest)
    recent = turns if keep < 0 else turns[-keep:]
    return [*system, *flatten(recent)]


def assemble(
    system: str, summary: str, recent: list[ChatMessage]
) -> list[ChatMessage]:
    messages: list[ChatMessage] = [{"role": "system", "content": system}]
    text = summary.strip()
    if text:
        messages.append({"role": "user", "content": SUMMARY_USER_PREFIX + text})
        messages.append({"role": "assistant", "content": SUMMARY_ASSISTANT})
    messages.extend(recent)
    return messages


def crude_summary(previous: str, overflow: list[ChatMessage]) -> str:
    lines: list[str] = []
    if previous.strip():
        lines.append(previous.strip())
    for msg in overflow:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content or role not in {"user", "assistant"}:
            continue
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}：{content}")
    return "\n".join(lines)[:_CRUDE_SUMMARY_MAX_CHARS]


def compress(
    messages: list[ChatMessage],
    keep: int,
    old_summary: str = "",
    summarize: Callable[[str, list[ChatMessage]], str] | None = None,
) -> tuple[list[ChatMessage], str]:
    found, rest = take_summary(messages)
    summary = old_summary.strip() or found
    turns = iter_turns(rest)
    if keep < 0 or len(turns) <= keep:
        return flatten(turns), summary
    overflow = flatten(turns[:-keep])
    recent = flatten(turns[-keep:])
    if summarize is None:
        return recent, crude_summary(summary, overflow)
    try:
        text = (summarize(summary, overflow) or "").strip()
    except Exception:
        text = ""
    return recent, text or crude_summary(summary, overflow)


@dataclass(slots=True)
class ShortTermState:
    messages: list[ChatMessage]
    summary: str


@dataclass(slots=True)
class _Entry:
    messages: list[ChatMessage]
    summary: str
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

    def load(self, chat_session_id: str, user_id: str = "default") -> ShortTermState | None:
        key = (user_id, chat_session_id)
        with self._lock:
            self._expire()
            entry = self._sessions.get(key)
            if entry is None:
                return None
            entry.used_at = self._clock()
            self._sessions.move_to_end(key)
            return ShortTermState(list(entry.messages), entry.summary)

    def save(
        self,
        chat_session_id: str,
        messages: list[ChatMessage],
        user_id: str = "default",
        summary: str = "",
    ) -> None:
        key = (user_id, chat_session_id)
        _, recent = take_summary(compact(messages, self.max_turns))
        with self._lock:
            self._sessions[key] = _Entry(recent, summary.strip(), self._clock())
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
