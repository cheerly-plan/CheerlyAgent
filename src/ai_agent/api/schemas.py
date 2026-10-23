from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

# user_id / session_id 会被写库、写日志，限制为无空白字符的短字符串，
# 同时防止日志注入（换行符）与超长字符串滥用存储。
SafeId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^\S+$")]


class UserInfo(BaseModel):
    user_id: str
    chat_session_id: str
    user_name: str


class ChatRequest(BaseModel):
    # regenerate=True 时后端从历史读取用户消息，message 可为空
    message: str = Field(default="", max_length=8000)
    chat_session_id: SafeId = "default"
    # regenerate=True：重新生成最后一条回答，不重复落库用户消息
    regenerate: bool = False


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
    idempotency_key: str = Field(min_length=8, max_length=128, pattern=r"^\S+$")


class SessionInfo(BaseModel):
    session_id: str
    title: str
    created_at: str | None = None
    updated_at: str | None = None


class SessionTitleResponse(BaseModel):
    session_id: str
    title: str


class SessionListResponse(BaseModel):
    user_id: str
    sessions: list[SessionInfo]


class SessionRenameRequest(BaseModel):
    session_id: SafeId
    title: str = Field(min_length=1, max_length=30)


class SessionDeleteRequest(BaseModel):
    session_id: SafeId


# 登录用户名即 user_id，沿用 SafeId 的约束（无空白、长度有限）
Username = Annotated[str, Field(min_length=2, max_length=32, pattern=r"^\S+$")]


class RegisterRequest(BaseModel):
    username: Username
    password: str = Field(min_length=8, max_length=64)
    display_name: str = Field("", max_length=32)


class LoginRequest(BaseModel):
    username: Username
    # 兼容历史 6 位密码用户，注册已要求 8 位
    password: str = Field(min_length=6, max_length=64)


class AuthUser(BaseModel):
    user_id: str
    user_name: str


class AuthResponse(BaseModel):
    token: str
    expires_at: str
    user: AuthUser
