from fastapi import Depends

from ai_agent.api.dependencies import get_current_user
from ai_agent.api.schemas import UserInfo
from ai_agent.memory.db import UserRecord
from fastapi import APIRouter

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/info")
def user_info(user: UserRecord = Depends(get_current_user)) -> UserInfo:
    # chat_session_id 已由会话接口管理，仅为兼容旧 UserInfo 结构保留
    return UserInfo(
        user_id=user.user_id,
        chat_session_id="default",
        user_name=user.display_name,
    )
