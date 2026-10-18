from ai_agent.api.routers.user import user_info
from ai_agent.api.schemas import ChatRequest


def fake_stream(user_input: str, user_id: str = "default", session_id: str = "default"):
    yield {"type": "text", "content": "你"}
    yield {"type": "text", "content": "好"}
    yield {"type": "done"}


def test_user_info_returns_required_fields():
    info = user_info()
    assert info.user_id
    assert info.sessionsid
    assert info.user_name


def test_chat_request_prefers_sessionsid():
    req = ChatRequest(message="你好", user_id="user_001", sessionsid="session_001")
    assert req.user_id == "user_001"
    assert req.resolved_session_id() == "session_001"