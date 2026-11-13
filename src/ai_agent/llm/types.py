from typing import Any, Literal, TypedDict

Role = Literal["system", "user", "assistant", "tool"]
ChatEventType = Literal[
    "run_started",
    "text",
    "tool_start",
    "tool_progress",
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
    # 仅从 DB 加载时回填的行号，用于摘要游标；_to_openai_messages 会忽略。
    seq: int
    # 消息级附属信息（目前只有 sources：这一轮回答引用的知识库切片）。
    # 只在落库/回放/SSE 用，_to_openai_messages 是白名单构造，不会漏给模型
    meta: dict[str, Any]
    # 本消息对应的网关计费口径 token 数。生产侧 assistant 消息带这两个键，
    # 历史数据为 None（当时没采集）。_to_openai_messages 同样不会下发。
    prompt_tokens: int | None
    completion_tokens: int | None


class ChatResult(TypedDict):
    content: str
    tool_calls: list[ToolCall]
    # 网关返回的 usage；网关不返回（本地小模型、部分代理）时缺省。
    prompt_tokens: int | None
    completion_tokens: int | None


class ChatEvent(TypedDict, total=False):
    type: ChatEventType
    content: str
    name: str
    message: str
    session_id: str
    title: str
    run_id: str
    # 本轮已收集到的知识库引用（tool_end / done 携带，前端整表替换）
    sources: list[dict[str, Any]]
