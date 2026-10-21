from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UserInfo(BaseModel):
    user_id: str
    chat_session_id: str
    user_name: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = Field(default="default")
    chat_session_id: str = "default"


class ChatMemoryMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatMemoryResponse(BaseModel):
    user_id: str
    chat_session_id: str
    messages: list[ChatMemoryMessage]
    has_more: bool = False
    next_before_seq: int | None = None


class SessionCreateRequest(BaseModel):
    user_id: str = Field(default="default")
    idempotency_key: str = Field(min_length=8, max_length=128)


class SessionInfo(BaseModel):
    session_id: str
    title: str
    created_at: str | None = None
    updated_at: str | None = None


class SessionListResponse(BaseModel):
    user_id: str
    sessions: list[SessionInfo]


class SessionRenameRequest(BaseModel):
    user_id: str = Field(default="default")
    session_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=30)


class SessionDeleteRequest(BaseModel):
    user_id: str = Field(default="default")
    session_id: str = Field(min_length=1)
