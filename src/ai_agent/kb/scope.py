"""会话级知识库检索范围。

前端在输入框里选定「只查这几份文件」后，范围随 ChatRequest 传进来。
工具调用发生在同一个生成器上下文里（chat_sse_events → run_stream →
execute_tool），用 contextvar 传递比改 Tool.func 的签名干净。

模型用文件名圈定范围（tools 的 files 参数），服务端再解析成 id；
用户圈定的会话范围只在模型没指定文件时生效。
"""

import uuid
from collections.abc import Iterable
from contextvars import ContextVar, Token

_SCOPE: ContextVar[tuple[str, ...]] = ContextVar("kb_scope", default=())


def set_kb_scope(file_ids: Iterable[str] | None) -> Token:
    cleaned = tuple(str(fid).strip() for fid in (file_ids or []) if str(fid).strip())
    return _SCOPE.set(cleaned)


def reset_kb_scope(token: Token) -> None:
    _SCOPE.reset(token)


def kb_scope() -> tuple[str, ...]:
    return _SCOPE.get()


def effective_kb_query(query: str) -> str:
    """检索词原样返回（查询改写已移除，模型传入的 query 即最终检索词）。"""
    return (query or "").strip()


def parse_scope_uuids(raw_ids: Iterable[str] | None) -> list[uuid.UUID] | None:
    parsed: list[uuid.UUID] = []
    for raw in raw_ids or []:
        try:
            parsed.append(uuid.UUID(str(raw).strip()))
        except ValueError:
            continue
    return parsed or None


def resolve_tool_file_ids(files: list[str] | None) -> list[uuid.UUID] | None:
    """模型给的文件名优先；对不上或没给，再退回会话范围 / 全库。"""
    from ai_agent.kb.retrieval import resolve_files_by_name

    if files:
        named = resolve_files_by_name(files)
        if named:
            return named
    return parse_scope_uuids(kb_scope())
