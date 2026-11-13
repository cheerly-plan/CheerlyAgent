from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic

from ai_agent.core.config import settings
from ai_agent.llm import metrics
from ai_agent.llm.tokenizer import (
    clip_text_to_tokens,
    count_messages_tokens,
    count_tokens,
)
from ai_agent.llm.types import ChatMessage

# 摘要伪装成一轮 user/assistant 对话注入上下文。两个常量迁到了
# prompts.memory，这里 import 进来是为了保持 ai_agent.memory.short_term
# 这个导入路径继续可用（tests 与 loop 都从这儿取）。
# SUMMARY_USER_PREFIX 的字面值还被 take_summary() 用来按前缀反解摘要，
# 改一个字就会静默丢掉全部历史摘要。
from ai_agent.prompts.memory import SUMMARY_ASSISTANT, SUMMARY_USER_PREFIX

_CRUDE_SUMMARY_MAX_TOKENS = 1024


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


def messages_tokens(messages: list[ChatMessage]) -> int:
    """上下文 token 数：Qwen tokenizer 计数 + 每条消息的固定开销。

    送模型前的裁剪、摘要触发和短期缓存共用这一个口径。
    """
    return count_messages_tokens(messages)


def fit_turns(
    messages: list[ChatMessage], max_tokens: int
) -> tuple[list[list[ChatMessage]], list[ChatMessage]]:
    """按 token 预算从最旧整轮丢弃，返回 (保留轮次, 被丢的消息)。

    system 与摘要对不参与丢弃；最后一个 turn 是当前回合，永不丢。
    被丢的消息由调用方决定是否并入摘要——它们不该静默蒸发。
    """
    system, rest = take_system(messages)
    summary, rest = take_summary(rest)
    turns = iter_turns(rest)
    if max_tokens <= 0:
        return turns, []
    sys_text = str(system[0].get("content") or "") if system else ""
    # 预计算 system/摘要和每轮 token，丢弃时做减法；避免每丢一轮都重新
    # tokenize 整段历史（O(n²) tokenizer 调用）。
    base = messages_tokens(assemble(sys_text, summary, []))
    if not sys_text:
        # assemble 会补一个空 system；原 messages 没有 system 时扣掉它的开销
        base -= 4
    turn_tokens = [messages_tokens(turn) for turn in turns]
    total = base + sum(turn_tokens)
    if total <= max_tokens:
        return turns, []
    overflow: list[ChatMessage] = []
    while len(turns) > 1 and total > max_tokens:
        overflow.extend(turns.pop(0))
        total -= turn_tokens.pop(0)
    return turns, overflow


def assemble(
    system: str, summary: str, recent: list[ChatMessage]
) -> list[ChatMessage]:
    messages: list[ChatMessage] = [{"role": "system", "content": system}]
    text = summary.strip()
    if text:
        # 摘要每轮注入直到会话结束，不设上限会在长会话里越滚越大，早期摘要
        # 里的遗漏/错误持续污染后续回复。这里按 token 预算保头裁剪（摘要
        # 由 LLM 精炼、重要信息靠前，保头丢尾是安全的确定性兜底）。
        limit = settings.summary_max_tokens
        if limit > 0 and count_tokens(text) > limit:
            clipped = clip_text_to_tokens(text, limit, "…")
            if clipped != text:
                metrics.incr(metrics.SUMMARY_SHRINK)
                text = clipped
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
    # 与正文预算统一走 Qwen tokenizer 口径：中文约 1 字/token、ASCII 约
    # 4 字符/token，字符硬切会低估英文/数字的占用。保头 + 显式省略标记，
    # 让降级摘录也不会静默丢尾。未超限时原样返回，不追加标记。
    text = "\n".join(lines)
    if count_tokens(text) > _CRUDE_SUMMARY_MAX_TOKENS:
        return clip_text_to_tokens(text, _CRUDE_SUMMARY_MAX_TOKENS, "…")
    return text


def compress(
    messages: list[ChatMessage],
    old_summary: str = "",
    summarize: Callable[[str, list[ChatMessage]], str] | None = None,
    max_tokens: int = 0,
    old_cursor: int = 0,
) -> tuple[list[ChatMessage], str, int]:
    """压缩出「送下一轮模型的窗口 + 覆盖全部被丢历史的摘要」。

    不按固定轮数触发；只有总 token 超过 ``max_tokens`` 时，才把最旧轮次
    挤进 ``overflow`` 交摘要。``max_tokens`` 与 ``_fit_context`` 同一口径。
    """
    found, rest = take_summary(messages)
    summary = old_summary.strip() or found
    turns = iter_turns(rest)
    overflow: list[ChatMessage] = []
    if max_tokens > 0:
        system, _ = take_system(messages)
        sys_text = str(system[0].get("content") or "") if system else ""
        turns, overflow = fit_turns(
            assemble(sys_text, summary, flatten(turns)), max_tokens
        )
    recent = flatten(turns)
    cursor = old_cursor
    if overflow:
        seqs = [
            int(msg["seq"]) for msg in overflow if msg.get("seq") is not None
        ]
        if seqs:
            cursor = max(cursor, max(seqs))
    if not overflow:
        return recent, summary, cursor
    if summarize is None:
        return recent, crude_summary(summary, overflow), cursor
    try:
        text = (summarize(summary, overflow) or "").strip()
    except Exception:
        text = ""
    return recent, text or crude_summary(summary, overflow), cursor


@dataclass(slots=True)
class ShortTermState:
    messages: list[ChatMessage]
    summary: str
    summary_until_seq: int = 0


@dataclass(slots=True)
class _Entry:
    messages: list[ChatMessage]
    summary: str
    used_at: float
    summary_until_seq: int = 0


class ShortTermMemory:
    def __init__(
        self,
        max_sessions: int = settings.short_term_memory_max_sessions,
        ttl_seconds: int = settings.short_term_memory_ttl_seconds,
        cleanup_interval_seconds: int = settings.short_term_memory_cleanup_interval_seconds,
        max_tokens: int = settings.context_message_token_budget(),
        clock: Callable[[], float] = monotonic,
    ):
        self._sessions: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self._lock = Lock()
        self._stop = Event()
        self._cleaner: Thread | None = None
        self._clock = clock
        self.max_tokens = max_tokens
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
            return ShortTermState(
                list(entry.messages), entry.summary, entry.summary_until_seq
            )

    def save(
        self,
        chat_session_id: str,
        messages: list[ChatMessage],
        user_id: str = "default",
        summary: str = "",
        summary_until_seq: int = 0,
        trim: bool = True,
    ) -> None:
        key = (user_id, chat_session_id)
        # trim=False 用于收尾快路径：先保留全量窗口，避免后台摘要还没跑完
        # 时下一轮只看到「已从缓存裁掉、但尚未进摘要」的轮次。
        turns, _ = fit_turns(messages, self.max_tokens if trim else 0)
        recent = flatten(turns)
        with self._lock:
            self._sessions[key] = _Entry(
                recent,
                summary.strip(),
                self._clock(),
                max(0, int(summary_until_seq)),
            )
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
