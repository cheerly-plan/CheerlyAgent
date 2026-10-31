"""对话 run 的资源预算。

把「最多几轮」从散落的魔法数字收拢成显式概念：执行体（run_loop）
只认 Budget，不直接读 settings。后续若引入 Plan-and-Execute /
Reflexion 等范式，在这里扩展字段（max_tool_calls、deadline 等），
执行体接口不变。
"""

from dataclasses import dataclass

from ai_agent.core.config import settings


@dataclass(frozen=True)
class Budget:
    """max_steps：模型决策次数上限（每次 LLM 调用算一步），防工具循环失控。"""

    max_steps: int

    @classmethod
    def from_settings(cls) -> "Budget":
        return cls(max_steps=settings.max_turns)
