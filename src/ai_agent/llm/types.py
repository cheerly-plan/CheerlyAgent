from typing import Any, Literal, TypedDict

Role = Literal["system", "user", "assistant", "tool"]
ChatEventType = Literal[
    "run_started",
    "text",
    "tool_start",
    "tool_end",
    "done",
    "error",
    "title_updated",
]


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict[str, Any]


class ChatMessage(TypedDict, total=False):
    role: Role
    content: str
    tool_calls: list[ToolCall]
    tool_call_id: str
    name: str


class ChatResult(TypedDict):
    content: str
    tool_calls: list[ToolCall]


class ChatEvent(TypedDict, total=False):
    type: ChatEventType
    content: str
    name: str
    label: str
    message: str
    session_id: str
    title: str
    run_id: str
