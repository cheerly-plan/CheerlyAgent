from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str
    api_key: str
    model: str
    small_model: str = ""
    max_turns: int = 8
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    # 服务端访问令牌：非空时 /v1/** 要求携带 X-API-Key 或 Bearer Token
    auth_token: str = ""
    # 聊天流并发上限，超出返回 429
    max_concurrent_chats: int = 20
    # 数据库连接池上限，需 ≥ 聊天并发 + 标题线程池 + 收尾线程池之和
    db_pool_max_size: int = 28
    # 允许的跨域来源，逗号分隔，如 "http://localhost:5173,https://chat.example.com"
    cors_origins: str = "http://localhost:5173"
    # 仅当部署在可信反向代理之后才开启：开启后限流取 X-Forwarded-For 最右侧 IP
    # （由自己的反代追加，客户端无法伪造）；默认关闭，直接取 socket 对端地址，
    # 防止伪造 XFF 绕过限流
    trust_forwarded_for: bool = False
    # trust_forwarded_for 开启时，视哪些地址为可信反代（uvicorn forwarded_allow_ips）
    forwarded_allow_ips: str = "127.0.0.1"
    # 开放注册开关：本机开发可开；对外暴露前建议关闭（403），改为管理员建号
    registration_enabled: bool = True
    # 断点续跑模式：True = prefill 续写（半截 assistant 留在上下文尾部，
    # 模型接着往下写，观感是"继续生成"）；False = 整段重生成（更兼容，
    # 适用于上下文以 assistant 结尾会被 API 忽略/报错的情况）
    resume_prefill: bool = True

    # 联网搜索工具：按配置的 key 自动选择搜索源（都未配置时工具返回
    # 引导文案）。tavily 免费额度 1000 次/月；bocha 为国内服务，直连可达；
    # serper 走 Google。多个 key 同时配置时按 tavily > bocha > serper 优先
    tavily_api_key: str = ""
    bocha_api_key: str = ""
    serper_api_key: str = ""
    web_search_max_results: int = 5
    web_search_timeout_seconds: float = 15

    database_url: str
    embedding_model: str
    embedding_dim: int
    memory_top_k: int
    memory_score_threshold: float
    memory_merge_threshold: float = 0.7
    memory_merge_timeout_seconds: float = 8
    memory_model: str = ""

    # Jev 门控（System One 决策模型）：长期记忆抽取的前置判断——Noul 概率
    # 低于 memory_gate_threshold 时跳过本轮抽取，省掉一次小模型调用。
    # 未配 key 时门控关闭，行为退化为每轮直接抽取；门控任何失败都
    # fail-open 放行，不影响记忆链路
    jev_api_key: str = ""
    jev_base_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-latest"
    memory_gate_threshold: float = 0.4
    jev_timeout_seconds: float = 4

    def resolve_llm(self, business_model: str = "") -> str:
        return business_model or self.small_model or self.model

    def llm_for_memory(self) -> str:
        return self.resolve_llm(self.memory_model)

    short_term_memory_max_turns: int
    short_term_memory_max_tokens: int
    short_term_memory_max_sessions: int = 1024
    short_term_memory_ttl_seconds: int = 3600
    short_term_memory_cleanup_interval_seconds: int = 300

    title_model: str = ""
    short_term_summary_model: str = ""
    short_term_summary_timeout_seconds: float = 12
    title_timeout_seconds: float = 20
    session_cleanup_interval_seconds: int = 300
    session_empty_max_age_seconds: int = 3600


settings = Settings()
