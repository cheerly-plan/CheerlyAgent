from fastapi import APIRouter, Depends, HTTPException, Request

from ai_agent.api.dependencies import extract_token, get_current_user
from ai_agent.api.schemas import (
    AuthResponse,
    AuthUser,
    LoginRequest,
    RegisterRequest,
)
from ai_agent.core.exceptions import AgentError
from ai_agent.memory.db import UserRecord
from ai_agent.memory.store import store

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def _to_auth_response(user: UserRecord) -> AuthResponse:
    try:
        token, expires_at = store.issue_token(user.user_id)
    except Exception as e:
        raise AgentError("签发会话令牌失败") from e
    return AuthResponse(
        token=token,
        expires_at=expires_at.isoformat(),
        user=AuthUser(user_id=user.user_id, user_name=user.display_name),
    )


@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest) -> AuthResponse:
    try:
        user = store.create_user(req.username, req.password, req.display_name)
    except Exception as e:
        raise AgentError("注册失败") from e
    if user is None:
        raise HTTPException(status_code=409, detail="用户名已被占用")
    return _to_auth_response(user)


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest) -> AuthResponse:
    try:
        user = store.authenticate(req.username, req.password)
    except Exception as e:
        raise AgentError("登录失败") from e
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return _to_auth_response(user)


@router.post("/logout")
def logout(request: Request, _user: UserRecord = Depends(get_current_user)) -> dict[str, bool]:
    store.revoke_token(extract_token(request))
    return {"ok": True}


@router.get("/me", response_model=AuthUser)
def me(user: UserRecord = Depends(get_current_user)) -> AuthUser:
    return AuthUser(user_id=user.user_id, user_name=user.display_name)
