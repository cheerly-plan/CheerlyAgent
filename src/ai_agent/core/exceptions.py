class AgentError(Exception):
    """业务异常基类。"""


class LLMError(AgentError):
    """模型调用失败。"""