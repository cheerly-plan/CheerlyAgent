from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., str],
        args_model: type[BaseModel],
        max_output_chars: int = 3000,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.args_model = args_model
        # 单个工具的默认输出上限；registry 里的 TOOL_OUTPUT_LIMITS 可按工具名覆盖，
        # 便于运维集中调整，而不必散到各工具模块。
        self.max_output_chars = max_output_chars

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    def run(self, **kwargs) -> str:
        args = self.args_model.model_validate(kwargs)
        return self.func(**args.model_dump())