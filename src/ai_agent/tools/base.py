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
        display_name: str | None = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.args_model = args_model
        # 用户可读的展示名（SSE tool_start 事件回传给前端显示），缺省用工具名
        self.display_name = display_name or name

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