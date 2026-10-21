from fastapi import APIRouter, HTTPException, Query

from ai_agent.api.schemas import (
    SessionCreateRequest,
    SessionDeleteRequest,
    SessionInfo,
    SessionListResponse,
    SessionRenameRequest,
)
from ai_agent.core.exceptions import AgentError
from ai_agent.memory.store import SessionRecord, store

router = APIRouter(prefix="/v1/session", tags=["session"])


@router.post("/create", response_model=SessionInfo)
def create_session(req: SessionCreateRequest) -> SessionInfo:
    try:
        session = store.create_session(req.user_id, req.idempotency_key)
    except Exception as e:
        raise AgentError("创建会话失败") from e
    return _to_info(session)


@router.post("/rename", response_model=SessionInfo)
def rename_session(req: SessionRenameRequest) -> SessionInfo:
    try:
        title = store.update_session_title(
            req.user_id, req.session_id, req.title, only_if_default=False
        )
        session = store.get_session(req.user_id, req.session_id)
    except Exception as e:
        raise AgentError("重命名会话失败") from e
    if title is None or session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _to_info(session)


@router.post("/delete")
def delete_session(req: SessionDeleteRequest) -> dict[str, bool]:
    try:
        deleted = store.delete_session(req.user_id, req.session_id)
    except Exception as e:
        raise AgentError("删除会话失败") from e
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}


@router.get("/list", response_model=SessionListResponse)
def list_sessions(
    user_id: str = "default",
    limit: int = Query(100, ge=1, le=200),
) -> SessionListResponse:
    try:
        rows = store.list_sessions(user_id, limit=limit)
    except Exception as e:
        raise AgentError("会话列表加载失败") from e
    return SessionListResponse(
        user_id=user_id,
        sessions=[_to_info(row) for row in rows],
    )


def _to_info(session: SessionRecord) -> SessionInfo:
    return SessionInfo(
        session_id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )
