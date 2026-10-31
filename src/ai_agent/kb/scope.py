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

from ai_agent.core.logging import get_logger

logger = get_logger(__name__)

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


def resolve_tool_file_ids(
    files: list[str] | None, query: str | None = None
) -> list[uuid.UUID] | None:
    """模型给的文件名/主体名优先；没给就先用文档路由，再退回会话范围 / 全库。

    返回语义（工具层据此区分）：
    - 非空 list：检索范围已确定（模型点名解析成功，或文档路由选出候选）
    - None：既没点名、路由也没结果、会话范围为空 → 调用方按全库处理
    - 空 list []：模型指定了 files 但一个都解析不到（编造的文件名/主体名）
      → 调用方应给出「未找到」提示，而不是静默退回全库查无关内容

    文档路由放在这条链路上而不是提示词里：files 是否填、填得准不准本来
    就不该由模型负责——摘要向量（见 kb/router.py）在检索层就能选出候选
    文档，文件清单因此不必再占用上下文。
    """
    from ai_agent.kb.retrieval import resolve_files_by_name

    if files:
        named = resolve_files_by_name(files)
        if named:
            return named
        # files 传了但一个都解析不到：明确返回空列表，让工具层拒绝并引导，
        # 不要静默退回全库（那会把无关内容当答案，又回到隐式全库扫描的老问题）。
        return []
    routed = _route(query)
    if routed:
        return routed
    return parse_scope_uuids(kb_scope())


def _route(query: str | None) -> list[uuid.UUID]:
    """按文档摘要做语义路由。任何异常都退回空列表（等价于路由没结果）。

    路由失败不能影响对话：底层接口 TP99 抖动一次，上层还有会话范围和
    全库两条路可走，抛出去只会让 kb_search 直接报 ERROR。
    """
    text = (query or "").strip()
    if not text:
        return []
    try:
        from ai_agent.kb.router import route_files

        return route_files(text)
    except Exception:
        logger.warning("文档路由失败，退回会话范围 / 全库", exc_info=True)
        return []
