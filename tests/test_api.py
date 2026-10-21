import pytest

from ai_agent.api.routers.chat import chat_memory
from ai_agent.api.routers.user import user_info
from ai_agent.api.schemas import ChatRequest
from ai_agent.core.exceptions import AgentError
from ai_agent.memory.store import ChatHistoryPage


def fake_stream(
    user_input: str, user_id: str = "default", chat_session_id: str = "default"
):
    yield {"type": "text", "content": "你"}
    yield {"type": "text", "content": "好"}
    yield {"type": "done"}


def test_user_info_returns_required_fields():
    info = user_info()
    assert info.user_id
    assert info.chat_session_id
    assert info.user_name


def test_chat_request_uses_chat_session_id():
    req = ChatRequest(message="你好", user_id="user_001", chat_session_id="session_001")
    assert req.user_id == "user_001"
    assert req.chat_session_id == "session_001"


def test_session_create_request_requires_idempotency_key():
    from pydantic import ValidationError

    from ai_agent.api.schemas import SessionCreateRequest

    with pytest.raises(ValidationError):
        SessionCreateRequest(user_id="user_001", idempotency_key="short")
    req = SessionCreateRequest(user_id="user_001", idempotency_key="idem-key-1")
    assert req.idempotency_key == "idem-key-1"


def test_chat_memory_returns_session_messages(monkeypatch):
    def fake_load(user_id, chat_session_id, limit=20, before_seq=None):
        assert user_id == "user_001"
        assert chat_session_id == "session_001"
        assert limit == 20
        assert before_seq is None
        return ChatHistoryPage(
            messages=[
                {"role": "user", "content": "北京天气"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "get_weather",
                            "arguments": {"city": "北京"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "name": "get_weather",
                    "content": "20度",
                },
                {"role": "assistant", "content": "20度"},
            ],
            has_more=False,
            next_before_seq=None,
        )

    monkeypatch.setattr("ai_agent.api.routers.chat.store.load_chat_page", fake_load)
    result = chat_memory(
        user_id="user_001",
        chat_session_id="session_001",
        limit=20,
        before_seq=None,
    )
    assert result.user_id == "user_001"
    assert result.chat_session_id == "session_001"
    assert result.has_more is False
    assert result.next_before_seq is None
    assert [m.role for m in result.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.messages[0].content == "北京天气"
    assert result.messages[1].tool_calls[0]["name"] == "get_weather"
    assert result.messages[2].tool_call_id == "c1"
    assert result.messages[3].content == "20度"


def test_chat_memory_paginates_older_page(monkeypatch):
    def fake_load(user_id, chat_session_id, limit=20, before_seq=None):
        assert user_id == "user_001"
        assert chat_session_id == "session_001"
        assert limit == 2
        assert before_seq == 5
        return ChatHistoryPage(
            messages=[
                {"role": "user", "content": "1"},
                {"role": "assistant", "content": "a1"},
            ],
            has_more=True,
            next_before_seq=1,
        )

    monkeypatch.setattr("ai_agent.api.routers.chat.store.load_chat_page", fake_load)
    result = chat_memory(
        user_id="user_001",
        chat_session_id="session_001",
        limit=2,
        before_seq=5,
    )
    assert [m.content for m in result.messages] == ["1", "a1"]
    assert result.has_more is True
    assert result.next_before_seq == 1


def test_chat_memory_empty(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.api.routers.chat.store.load_chat_page",
        lambda user_id, chat_session_id, limit=20, before_seq=None: ChatHistoryPage(
            [], False, None
        ),
    )
    result = chat_memory(
        user_id="user_001",
        chat_session_id="session_001",
        limit=20,
        before_seq=None,
    )
    assert result.chat_session_id == "session_001"
    assert result.messages == []
    assert result.has_more is False


def test_chat_memory_raises_when_store_fails(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr("ai_agent.api.routers.chat.store.load_chat_page", fail)
    with pytest.raises(AgentError, match="会话记忆加载失败"):
        chat_memory(user_id="user_001", chat_session_id="session_001")
