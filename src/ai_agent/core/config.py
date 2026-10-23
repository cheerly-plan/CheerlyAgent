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
    # 聊天流并发上限，超出直接返回 429
    max_concurrent_chats: int = 20

    database_url: str
    embedding_model: str
    embedding_dim: int
    memory_top_k: int
    memory_score_threshold: float
    memory_merge_threshold: float = 0.7
    memory_merge_timeout_seconds: float = 8
    memory_model: str = ""

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
