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
    # 消息级附属信息（目前只有 sources：这一轮回答引用的知识库切片）。
    # 只在落库/回放/SSE 用，_to_openai_messages 是白名单构造，不会漏给模型
    meta: dict[str, Any]


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
    # 本轮已收集到的知识库引用（tool_end / done 携带，前端整表替换）
    sources: list[dict[str, Any]]
