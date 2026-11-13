"""管理员查看任意用户的会话，以及每一问后面的 agent 过程。"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ai_agent.api.dependencies import require_admin
from ai_agent.api.inspect_turns import (
    input_tokens,
    sum_tokens,
    turns_from_messages,
)
from ai_agent.api.schemas import (
    InspectSession,
    InspectSessionList,
    InspectTurns,
    InspectUser,
    InspectUserList,
    SafeId,
)
from ai_agent.core.config import settings
from ai_agent.core.exceptions import AgentError
from ai_agent.storage.db import UserRecord
from ai_agent.storage.store import store

router = APIRouter(prefix="/v1/inspect", tags=["inspect"])


@router.get("/users", response_model=InspectUserList)
def list_users(
    _admin: UserRecord = Depends(require_admin),
    limit: int = Query(200, ge=1, le=500),
) -> InspectUserList:
    try:
        rows = store.list_users(limit)
    except Exception as e:
        raise AgentError("用户列表加载失败") from e
    users = [
        InspectUser(
            user_id=row.user_id,
            user_name=row.display_name or row.username,
            username=row.username,
            is_admin=settings.is_admin_username(row.username),
        )
        for row in rows
    ]
    # 管理员排在前面，打开页面时默认就能落在管理员身上
    users.sort(key=lambda item: (not item.is_admin, item.user_name))
    return InspectUserList(users=users)


@router.get("/sessions", response_model=InspectSessionList)
def list_user_sessions(
    user_id: SafeId = Query(...),
    _admin: UserRecord = Depends(require_admin),
    limit: int = Query(100, ge=1, le=200),
) -> InspectSessionList:
    try:
        rows = store.list_sessions(user_id, limit=limit)
    except Exception as e:
        raise AgentError("会话列表加载失败") from e
    return InspectSessionList(
        user_id=user_id,
        sessions=[
            InspectSession(
                session_id=row.id,
                title=row.title,
                updated_at=row.updated_at,
            )
            for row in rows
        ],
    )


@router.get("/report-config")
def report_config_lint(
    request: Request,
    _admin: UserRecord = Depends(require_admin),
) -> dict:
    """管理员查启动 lint 结果：不用翻日志就能看到哪几项配置没过。

    结果在 lifespan 里算一次并挂到 app.state，这里是只读快照；要重跑就重启服务
    或调用 ``lint_report_config()``。未执行过 lifespan（例如单测）返回 503。
    """
    report = getattr(request.app.state, "report_config_lint", None)
    if report is None:
        raise HTTPException(status_code=503, detail="启动配置 lint 尚未执行")
    return {
        "ok": report.ok,
        "strict": settings.report_config_strict,
        "checked": list(report.checked),
        "issues": list(report.issues),
    }


@router.get("/turns", response_model=InspectTurns)
def list_turns(
    user_id: SafeId = Query(...),
    session_id: SafeId = Query(...),
    _admin: UserRecord = Depends(require_admin),
) -> InspectTurns:
    try:
        session = store.get_session(user_id, session_id)
        messages = store.load_chat(user_id, session_id) if session else None
    except Exception as e:
        raise AgentError("对话过程加载失败") from e
    if session is None or messages is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    turns = turns_from_messages(messages)
    # prompt_tokens 每次请求都包含完整历史，按轮相加会成倍放大；但会话总量
    # 的口径就是「网关实际计费的输入 token 之和」，与计费单一致，所以照加。
    return InspectTurns(
        user_id=user_id,
        session_id=session.id,
        title=session.title,
        turns=turns,
        # 会话有多满：所有消息自身 token 之和（本地现算，不依赖网关字段）
        context_tokens=sum(input_tokens(m) for m in messages),
        # 会话花了多少：网关累计扣费，prompt 含历史重传
        prompt_tokens=sum_tokens([t.prompt_tokens for t in turns]),
        completion_tokens=sum_tokens([t.completion_tokens for t in turns]),
    )
