from fastapi import APIRouter

from ai_agent.api.schemas import UserInfo

router = APIRouter(prefix="/user", tags=["user"])

_DEMO_USER = UserInfo(
    user_id="user_001",
    chat_session_id="session_001",
    user_name="ppl",
)


@router.get("/info")
def user_info() -> UserInfo:
    return _DEMO_USER
