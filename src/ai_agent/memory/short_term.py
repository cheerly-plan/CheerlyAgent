from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic

from ai_agent.core.config import settings
from ai_agent.llm.types import ChatMessage

# 摘要伪装成一轮 user/assistant 对话注入上下文。两个常量迁到了
# prompts.memory，这里 import 进来是为了保持 ai_agent.memory.short_term
# 这个导入路径继续可用（tests 与 loop 都从这儿取）。
# SUMMARY_USER_PREFIX 的字面值还被 take_summary() 用来按前缀反解摘要，
# 改一个字就会静默丢掉全部历史摘要。
from ai_agent.prompts.memory import SUMMARY_ASSISTANT, SUMMARY_USER_PREFIX

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


def messages_chars(messages: list[ChatMessage]) -> int:
    """上下文长度的粗估（字符数）：正文 + tool_calls 参数。

    送模型前的裁剪与收尾时的摘要共用这一个口径，避免「一边按字符裁、
    一边按轮数摘要」导致被裁掉的历史既不进窗口也不进摘要。
    """
    total = 0
    for msg in messages:
        total += len(str(msg.get("content") or ""))
        for call in msg.get("tool_calls") or []:
            total += len(str(call.get("arguments") or ""))
    return total


def fit_turns(
    messages: list[ChatMessage], max_chars: int
) -> tuple[list[list[ChatMessage]], list[ChatMessage]]:
    """按字符预算从最旧整轮丢弃，返回 (保留轮次, 被丢的消息)。

    system 与摘要对不参与丢弃；最后一个 turn 是当前回合，永不丢。
    被丢的消息由调用方决定是否并入摘要——它们不该静默蒸发。
    """
    system, rest = take_system(messages)
    summary, rest = take_summary(rest)
    turns = iter_turns(rest)
    if max_chars <= 0 or messages_chars(messages) <= max_chars:
        return turns, []
    sys_text = str(system[0].get("content") or "") if system else ""
    overflow: list[ChatMessage] = []
    while len(turns) > 1:
        overflow.extend(turns.pop(0))
        if messages_chars(assemble(sys_text, summary, flatten(turns))) <= max_chars:
            break
    return turns, overflow


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
    max_chars: int = 0,
) -> tuple[list[ChatMessage], str]:
    """压缩出「送下一轮模型的窗口 + 覆盖全部被丢历史的摘要」。

    ``keep`` 是轮数窗口，``max_chars`` 是与 ``_fit_context`` 同一个字符预算。
    两者都要走：只按轮数压缩时，窗口内几轮大工具输出仍会超预算，送模型前
    被整轮丢掉的历史既没进摘要也没进长期记忆——用户视角就是模型突然忘事。
    现在预算挤掉的轮次和轮数窗口外的轮次一起进 ``overflow`` 交摘要。
    """
    found, rest = take_summary(messages)
    summary = old_summary.strip() or found
    turns = iter_turns(rest)
    overflow: list[ChatMessage] = []
    if keep >= 0 and len(turns) > keep:
        overflow = flatten(turns[:-keep])
        turns = turns[-keep:]
    if max_chars > 0:
        system, _ = take_system(messages)
        sys_text = str(system[0].get("content") or "") if system else ""
        turns, dropped = fit_turns(
            assemble(sys_text, summary, flatten(turns)), max_chars
        )
        overflow = [*overflow, *dropped]
    recent = flatten(turns)
    if not overflow:
        return recent, summary
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
        max_chars: int = settings.llm_context_char_budget,
        clock: Callable[[], float] = monotonic,
    ):
        self._sessions: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self._lock = Lock()
        self._stop = Event()
        self._cleaner: Thread | None = None
        self._clock = clock
        self.max_turns = max_turns
        self.max_chars = max_chars
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
        # 与送模型前的裁剪同口径：轮数窗口之后再按字符预算裁一次，
        # 保证热缓存里的窗口不会比模型实际看到的更宽（超预算照样被丢）。
        turns, _ = fit_turns(compact(messages, self.max_turns), self.max_chars)
        recent = flatten(turns)
        with self._lock:
            self._sessions[key] = _Entry(recent, summary.strip(), self._clock())
            self._sessions.move_to_end(key)
            self._evict()

    def invalidate(self, user_id: str, chat_session_id: str | None = None) -> None:
        """会话历史被删除/清空时同步失效热缓存。

        不失效的话，DB 已清空但进程内 TTL 内的旧窗口还能被 load 命中，
        用户会看到「删掉的会话自己回来了」。传 session_id 只清一个会话；
        不传则清该用户全部会话（对应 clear_chat 的全局清空）。
        """
        with self._lock:
            if chat_session_id is not None:
                self._sessions.pop((user_id, chat_session_id), None)
                return
            for key in [k for k in self._sessions if k[0] == user_id]:
                del self._sessions[key]

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
