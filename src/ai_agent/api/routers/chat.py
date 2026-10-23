import json
import threading
from collections.abc import Iterator

from ai_agent.agents.loop import run_stream
from ai_agent.api.dependencies import get_current_user
from ai_agent.api.schemas import ChatMemoryMessage, ChatMemoryResponse, ChatRequest, SafeId
from ai_agent.core.config import settings
from ai_agent.core.exceptions import AgentError
from ai_agent.memory.db import UserRecord
from ai_agent.memory.store import store
from ai_agent.tasks.title import start_title_job
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/v1", tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# 聊天流并发上限：每个进行中的 SSE 占用一个线程池线程并消耗 LLM 配额，
# 超出直接拒绝，避免打满线程池/烧穿 token 额度。
_CHAT_SLOTS = threading.BoundedSemaphore(settings.max_concurrent_chats)


def _pack(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def sse_events(user_id: str, session_id: str, content: str) -> Iterator[str]:
    # 并发槽已在路由处理器里 acquire（超限返回 429）；这里只负责释放
    try:
        answer_parts: list[str] = []
        for event in run_stream(
            content,
            user_id=user_id,
            chat_session_id=session_id,
            finalize_background=True,
        ):
            if event.get("type") == "text":
                answer_parts.append(event.get("content") or "")
            yield _pack(event)

        # 标题任务后台执行并直接写库；前端通过 GET /v1/session/title 轮询结果，
        # SSE 流在 done 后立即结束，不再为标题阻塞最长 22 秒。
        start_title_job(user_id, session_id, content, "".join(answer_parts))
    finally:
        _CHAT_SLOTS.release()


def _stream(user_id: str, session_id: str, content: str) -> StreamingResponse:
    return StreamingResponse(
        sse_events(user_id, session_id, content),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _acquire_chat_slot() -> None:
    """并发槽必须在返回 StreamingResponse 前获取，超限才能真正响应 429。"""
    if not _CHAT_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="当前对话并发已达上限，请稍后再试")


@router.post("/chat")
def chat(
    req: ChatRequest, user: UserRecord = Depends(get_current_user)
) -> StreamingResponse:
    if not req.regenerate and not req.message.strip():
        raise HTTPException(status_code=422, detail="消息不能为空")
    # 内置 default 会话无需预先创建，其余会话必须真实存在，防止垃圾历史堆积
    if req.chat_session_id != "default":
        try:
            session = store.get_session(user.user_id, req.chat_session_id)
        except Exception as e:
            raise AgentError("会话查询失败") from e
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
    _acquire_chat_slot()
    if req.regenerate:
        return _stream_regenerate(user.user_id, req.chat_session_id)
    return _stream(user.user_id, req.chat_session_id, req.message)


def _stream_regenerate(user_id: str, session_id: str) -> StreamingResponse:
    """重新生成最后一条回答：不重复保存用户消息，末尾旧回答先从历史剔除。"""

    def events() -> Iterator[str]:
        try:
            for event in run_stream(
                "",
                user_id=user_id,
                chat_session_id=session_id,
                finalize_background=True,
                regenerate=True,
            ):
                yield _pack(event)
        finally:
            _CHAT_SLOTS.release()

    return StreamingResponse(
        events(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.get("/chat/memory", response_model_exclude_none=True)
def chat_memory(
    user: UserRecord = Depends(get_current_user),
    chat_session_id: SafeId = Query("default"),
    limit: int = Query(20, ge=1, le=100),
    before_seq: int | None = Query(None, ge=1),
) -> ChatMemoryResponse:
    try:
        page = store.load_chat_page(
            user.user_id, chat_session_id, limit=limit, before_seq=before_seq
        )
    except Exception as e:
        raise AgentError("会话记忆加载失败") from e
    return ChatMemoryResponse(
        user_id=user.user_id,
        chat_session_id=chat_session_id,
        messages=[ChatMemoryMessage.model_validate(msg) for msg in page.messages],
        has_more=page.has_more,
        next_before_seq=page.next_before_seq,
    )
