import threading
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, Request

from ai_agent.api.dependencies import extract_token, get_current_user
from ai_agent.api.schemas import (
    AuthResponse,
    AuthUser,
    LoginRequest,
    RegisterRequest,
)
from ai_agent.core.config import settings
from ai_agent.core.exceptions import AgentError
from ai_agent.memory.db import UserRecord
from ai_agent.memory.store import store

router = APIRouter(prefix="/v1/auth", tags=["auth"])

# 注册/登录限流：单 IP 滑动窗口，无验证码场景下抬高撞库成本
_AUTH_RATE_LIMIT = 10
_AUTH_RATE_WINDOW = 60.0
_RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_RATE_LOCK = threading.Lock()


def _client_ip(request: Request) -> str:
    # 只在部署于可信反代之后（TRUST_FORWARDED_FOR=true）才读 X-Forwarded-For，
    # 且取最右侧一跳——它由自己的反代追加，客户端伪造的头只会出现在左侧。
    # 默认直接取 socket 对端地址，伪造 XFF 无法绕过限流。
    # 注意：uvicorn 自带的 ProxyHeadersMiddleware 也会改写 request.client，
    # 必须同时在 __main__.py 关闭（仅 TRUST_FORWARDED_FOR=true 时开启），
    # 直接用裸 uvicorn 启动会绕过这里的防护。
    if settings.trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            hops = [h.strip() for h in forwarded.split(",") if h.strip()]
            if hops:
                return hops[-1]
    return request.client.host if request.client else "unknown"


def _rate_limit(request: Request) -> None:
    key = _client_ip(request)
    now = time.monotonic()
    with _RATE_LOCK:
        # 防止字典无限增长（恶意伪造 XFF 的场景）
        if len(_RATE_BUCKETS) > 10000:
            stale = [
                k
                for k, v in _RATE_BUCKETS.items()
                if not v or now - v[-1] > _AUTH_RATE_WINDOW
            ]
            for k in stale:
                _RATE_BUCKETS.pop(k, None)
        bucket = _RATE_BUCKETS[key]
        while bucket and now - bucket[0] > _AUTH_RATE_WINDOW:
            bucket.popleft()
        if len(bucket) >= _AUTH_RATE_LIMIT:
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
        bucket.append(now)


def _to_auth_response(user: UserRecord) -> AuthResponse:
    try:
        token, expires_at = store.issue_token(user.user_id)
    except Exception as e:
        raise AgentError("签发会话令牌失败") from e
    return AuthResponse(
        token=token,
        expires_at=expires_at.isoformat(),
        user=AuthUser(
            user_id=user.user_id,
            user_name=user.display_name,
            is_admin=settings.is_admin_username(user.username),
        ),
    )


@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest, request: Request) -> AuthResponse:
    # 对外暴露前可关闭注册（REGISTRATION_ENABLED=false），避免任意建号烧模型额度
    if not settings.registration_enabled:
        raise HTTPException(status_code=403, detail="注册已关闭，请联系管理员开通账号")
    _rate_limit(request)
    try:
        user = store.create_user(req.username, req.password, req.display_name)
    except Exception as e:
        raise AgentError("注册失败") from e
    if user is None:
        raise HTTPException(status_code=409, detail="用户名已被占用")
    return _to_auth_response(user)


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest, request: Request) -> AuthResponse:
    _rate_limit(request)
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
    # 与登录响应、/user/info 保持同构：三个入口都必须带 is_admin，
    # 否则前端换一个取用户信息的来源就会丢失知识库入口
    return AuthUser(
        user_id=user.user_id,
        user_name=user.display_name,
        is_admin=settings.is_admin_username(user.username),
    )
