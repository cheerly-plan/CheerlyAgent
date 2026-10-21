import json
from collections.abc import Iterator

from ai_agent.agents.loop import run_stream
from ai_agent.api.schemas import ChatMemoryMessage, ChatMemoryResponse, ChatRequest
from ai_agent.core.config import settings
from ai_agent.core.exceptions import AgentError
from ai_agent.memory.store import store
from ai_agent.tasks.title import start_title_job, wait_title_event
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/v1", tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _pack(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def sse_events(user_id: str, session_id: str, content: str) -> Iterator[str]:
    answer_parts: list[str] = []
    for event in run_stream(content, user_id=user_id, chat_session_id=session_id):
        if event.get("type") == "text":
            answer_parts.append(event.get("content") or "")
        yield _pack(event)

    future = start_title_job(user_id, session_id, content, "".join(answer_parts))
    event = wait_title_event(
        future, session_id, timeout=settings.title_timeout_seconds + 2
    )
    if event is not None:
        yield _pack(event)


def _stream(user_id: str, session_id: str, content: str) -> StreamingResponse:
    return StreamingResponse(
        sse_events(user_id, session_id, content),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return _stream(req.user_id, req.chat_session_id, req.message)


@router.get("/chat/stream")
def chat_stream(
    content: str = Query(..., min_length=1, max_length=8000),
    user_id: str = Query("default"),
    session_id: str | None = Query(None, min_length=1),
    sessionId: str | None = Query(None, min_length=1),
) -> StreamingResponse:
    sid = sessionId or session_id
    if not sid:
        raise HTTPException(status_code=422, detail="缺少 sessionId")
    try:
        session = store.get_session(user_id, sid)
    except Exception as e:
        raise AgentError("会话查询失败") from e
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _stream(user_id, sid, content)


@router.get("/chat/memory", response_model_exclude_none=True)
def chat_memory(
    user_id: str = "default",
    chat_session_id: str = "default",
    limit: int = Query(20, ge=1, le=100),
    before_seq: int | None = Query(None, ge=1),
) -> ChatMemoryResponse:
    try:
        page = store.load_chat_page(
            user_id, chat_session_id, limit=limit, before_seq=before_seq
        )
    except Exception as e:
        raise AgentError("会话记忆加载失败") from e
    return ChatMemoryResponse(
        user_id=user_id,
        chat_session_id=chat_session_id,
        messages=[ChatMemoryMessage.model_validate(msg) for msg in page.messages],
        has_more=page.has_more,
        next_before_seq=page.next_before_seq,
    )
