from typing import Annotated, Any, Literal

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


class KbSearchRequest(BaseModel):
    """知识库页检索演示。file_ids 为空表示全库，不按问题推断范围。"""

    query: str = Field(min_length=1, max_length=500)
    file_ids: list[str] = Field(default_factory=list, max_length=50)
    top_k: int = Field(default=5, ge=1, le=8)


class KbCninfoFetchRequest(BaseModel):
    """从巨潮按股票代码 + 报告期拉取定期报告 PDF 并入库。"""

    code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")
    period: str = Field(
        min_length=4,
        max_length=7,
        description="YYYY / YYYYQn / YYYYH1，如 2025Q3",
    )
    max_files: int = Field(default=1, ge=1, le=10)


class KbUrlIngestRequest(BaseModel):
    """管理员粘贴巨潮 PDF 直链入库（仅 static.cninfo.com.cn）。"""

    url: str = Field(min_length=12, max_length=500)
    filename: str = Field(
        min_length=1,
        max_length=120,
        description="规范名（可不带 .pdf），如 中国银行股份有限公司2025年第三季度报告",
    )
    tags: list[str] = Field(default_factory=list, max_length=8)


class KgBuildRequest(BaseModel):
    """管理员选文件批量生成图谱。"""

    file_ids: list[str] = Field(min_length=1, max_length=20)
    # 可选：给这批文件统一套一套抽取配置（空 = 各文件用自己已保存的配置，
    # 都没存过就走全局默认的通用配置）
    doc_type: str = Field(default="", max_length=40)
    level: str = Field(default="", max_length=10)


class KgRebuildRequest(BaseModel):
    """重建单文件图谱时可选携带抽取配置（文档类型 + 粒度）。

    doc_type 是配置 key（kb/skills 内置：prospectus / annual_report /
    quarterly_report / legal_opinion / due_diligence / announcement / general，
    或 data/kb_doc_profiles.json 覆盖/追加），不是中文名——后端按 key 找配置块。
    空串 = 清成全局默认的通用配置。
    """

    doc_type: str = Field(default="", max_length=40)
    level: str = Field(default="", max_length=10)


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

class InspectUser(BaseModel):
    user_id: str
    user_name: str
    username: str
    is_admin: bool = False


class InspectUserList(BaseModel):
    users: list[InspectUser]


class InspectSession(BaseModel):
    session_id: str
    title: str
    updated_at: str


class InspectSessionList(BaseModel):
    user_id: str
    sessions: list[InspectSession]


class InspectStep(BaseModel):
    """一轮提问之后的单步过程。

    tool 是模型发出的一次工具调用，name 是工具名，例如 kb_search。
    tool_result 是这次调用的返回，arguments 带回同一次调用的参数。
    llm 是模型当步写下的正文。
    """

    kind: str
    name: str = ""
    tool_call_id: str = ""
    content: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)


class InspectTurn(BaseModel):
    index: int
    question: str
    steps: list[InspectStep]


class InspectTurns(BaseModel):
    user_id: str
    session_id: str
    title: str
    turns: list[InspectTurn]


class ResumeRequest(BaseModel):
    run_id: str = Field(min_length=8, max_length=64)
    chat_session_id: SafeId = "default"


class ReportRequest(BaseModel):
    """生成一份研报。

    服务端按确定性流水线跑：解析需求 → 检索抽取 → 计算增减 → 撰写 →
    校验 → 渲染。正文里的数字只来自知识库表格切片，抽不到的写进 gaps，
    不补估计值。返回 Markdown 正文、溯源附件、校验结论和人工审核队列；
    PDF 另落盘，可通过返回的 download_url 下载。
    """

    request: str = Field(
        min_length=1,
        max_length=500,
        description="一句需求，例如「生成中国能源建设2026Q3财报深度分析PDF研报」",
    )
    # 不传时按需求文本推断（含 PDF 字样 → pdf，否则 markdown）
    deliverable: Literal["pdf", "markdown"] | None = Field(
        default=None, description="交付形态，留空按需求文本推断"
    )
    # 报告类型（skill 层路由）：留空自动路由（银行/通用公司/行业）
    report_type: str | None = Field(
        default=None,
        max_length=40,
        description=(
            "报告类型，留空按需求自动路由。"
            "可选：bank_financial_report / company_financial_report / industry_report"
        ),
    )
    write_pdf: bool = Field(
        default=True, description="False 时跳过渲染，只出 Markdown（省渲染开销）"
    )


class ReportDecisionRequest(BaseModel):
    """处置一份研报的人工审核队列（M2）。

    item_ids 留空 = 整批处置；approve 全部完成后才可调用 finalize。
    reset 把已处置的条目退回 pending，用于审错后的重新核对。
    """

    action: Literal["approve", "reject", "reset"] = Field(
        description="approve 批准 / reject 驳回 / reset 退回待审"
    )
    item_ids: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="要处置的审核项 id；留空表示整批",
    )
    note: str = Field(default="", max_length=500, description="审核备注，写进审计记录")
