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


class ShortTermMemory:
    def __init__(self, max_turns: int = settings.short_term_memory_max_turns):
        self._sessions: dict[str, list[ChatMessage]] = {}
        self.max_turns = max_turns

    def load(self, session_id: str) -> list[ChatMessage]:
        return list(self._sessions.get(session_id, []))

    def save(self, session_id: str, messages: list[ChatMessage]) -> None:
        self._sessions[session_id] = messages

    def clear(self) -> None:
        self._sessions.clear()


short_term_memory = ShortTermMemory()
