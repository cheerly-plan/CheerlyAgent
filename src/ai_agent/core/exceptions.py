class AgentError(Exception):
    """业务异常基类。"""


class ConfigError(AgentError):
    """配置缺失或非法。"""


class LLMError(AgentError):
    """模型调用失败。"""


class ToolError(AgentError):
    """工具执行失败。"""