from fastapi import Depends, HTTPException, Request

from ai_agent.core.config import settings
from ai_agent.memory.db import UserRecord
from ai_agent.memory.store import store

_BEARER_PREFIX = "bearer "


def extract_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth[: len(_BEARER_PREFIX)].lower() != _BEARER_PREFIX:
        return ""
    return auth[len(_BEARER_PREFIX):].strip()


def get_current_user(request: Request) -> UserRecord:
    """从 Authorization: Bearer <token> 解析当前登录用户，未登录返回 401。"""
    user = store.resolve_token(extract_token(request))
    if user is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return user


def require_admin(user: UserRecord = Depends(get_current_user)) -> UserRecord:
    """管理员接口守卫：非白名单用户 403（知识库上传/删除用）。"""
    if not settings.is_admin_username(user.username):
        raise HTTPException(status_code=403, detail="只有管理员可以管理知识库文件")
    return user
