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
    """一次 run 的资源上限。

    * ``max_steps``          模型决策次数上限（每次 LLM 调用算一步）
    * ``max_tool_calls``     单个工具在本 run 内的调用次数上限（0 = 不限）

    两者管的不是一件事：一个步骤里模型可以并行发起多个工具调用，
    ``max_steps`` 拦不住「10 步里调 13 次工具」。``max_tool_calls``
    按工具分别计数，防止某个工具被反复重试烧光整轮预算。
    """

    max_steps: int
    max_tool_calls: int = 0

    @classmethod
    def from_settings(cls) -> "Budget":
        return cls(
            max_steps=settings.max_turns,
            max_tool_calls=settings.max_tool_calls_per_run,
        )
