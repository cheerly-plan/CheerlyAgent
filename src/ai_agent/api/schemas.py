from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

# user_id / session_id 会被写库、写日志，限制为无空白字符的短字符串，
# 同时防止日志注入（换行符）与超长字符串滥用存储。
SafeId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^\S+$")]


class UserInfo(BaseModel):
    user_id: str
    chat_session_id: str
    user_name: str
    # 管理员可上传/删除知识库文件（前端据此隐藏入口，后端 403 才是真防线）
    is_admin: bool = False


class KbInitRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    # 可选标签：建档时顺手打标；管理页也可事后改（PATCH /tags）
    tags: list[str] = Field(default_factory=list, max_length=8)


class KbTagsRequest(BaseModel):
    """全量覆盖式设置文件标签：传空数组即清空。"""

    tags: list[str] = Field(max_length=8)


class KbRenameRequest(BaseModel):
    """管理员重命名文件：DB 展示名与磁盘文件同步改，stem 变了再触发重嵌。"""

    filename: str = Field(min_length=1, max_length=255)


class KbChunkEditRequest(BaseModel):
    """人工修订切片正文：保存即单片重嵌入。"""

    text: str = Field(min_length=1, max_length=2000)


class KgBuildRequest(BaseModel):
    """管理员选文件批量生成图谱。"""

    file_ids: list[str] = Field(min_length=1, max_length=20)


class ChatRequest(BaseModel):
    # regenerate=True 时后端从历史读取用户消息，message 可为空
    message: str = Field(default="", max_length=8000)
    chat_session_id: SafeId = "default"
    # regenerate=True：重新生成最后一条回答，不重复落库用户消息
    regenerate: bool = False
    # 前端「智能搜索」开关：True 时本轮提供 web_search 工具
    web_search: bool = False
    # 前端「知识库」开关：True 时本轮提供 kb_search 工具（默认关闭，不挂载）
    use_kb: bool = False
    # 会话级知识库范围：用户在输入框里选定只看这几份文件；
    # 为空 = 全库。约束的是 kb_search 工具的默认检索范围
    kb_file_ids: list[str] = Field(default_factory=list, max_length=20)


class ChatMemoryMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    # 消息级附属信息（sources：这一轮回答引用的知识库切片），前端渲染引用用
    meta: dict[str, Any] | None = None


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
    # 管理员才显示知识库入口（后端 403 才是真防线）
    is_admin: bool = False


class AuthResponse(BaseModel):
    token: str
    expires_at: str
    user: AuthUser

class ResumeRequest(BaseModel):
    run_id: str = Field(min_length=8, max_length=64)
    chat_session_id: SafeId = "default"
