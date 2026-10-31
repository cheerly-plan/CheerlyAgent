class AgentError(Exception):
    """业务异常基类。"""


class LLMError(AgentError):
    """模型调用失败。

    status_code 保留上游 HTTP 状态码（拿不到时为 None），用于区分
    「重试也没用」的鉴权/开通类错误（401/403）和临时故障。
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def is_permission_denied(self) -> bool:
        return self.status_code in (401, 403)

    @property
    def is_rate_limited(self) -> bool:
        return self.status_code == 429
