"""会话级知识库检索范围。

前端在输入框里选定「只查这几份文件」后，范围随 ChatRequest 传进来。
工具调用发生在同一个生成器上下文里（chat_sse_events → run_stream →
execute_tool），用 contextvar 传递比改 Tool.func 的签名干净：
kb_search 的 file_ids 参数仍然留给模型自己用，两者是"或"的关系。
"""

from contextvars import ContextVar, Token
from collections.abc import Iterable

_SCOPE: ContextVar[tuple[str, ...]] = ContextVar("kb_scope", default=())


def set_kb_scope(file_ids: Iterable[str] | None) -> Token:
    cleaned = tuple(str(fid).strip() for fid in (file_ids or []) if str(fid).strip())
    return _SCOPE.set(cleaned)


def reset_kb_scope(token: Token) -> None:
    _SCOPE.reset(token)


def kb_scope() -> tuple[str, ...]:
    return _SCOPE.get()
