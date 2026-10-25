import json
from concurrent.futures import Future

import pytest

from ai_agent.api.routers.chat import _CHAT_SLOTS, chat, chat_sse_events
from ai_agent.api.routers.session import (
    create_session,
    delete_session,
    get_session_title,
    list_sessions,
    rename_session,
)
from ai_agent.api.schemas import (
    ChatRequest,
    SessionCreateRequest,
    SessionDeleteRequest,
    SessionRenameRequest,
)
from ai_agent.core.exceptions import AgentError
from ai_agent.memory.db import UserRecord
from ai_agent.memory.store import DEFAULT_SESSION_TITLE, SessionRecord
from ai_agent.tasks.title import (
    generate_session_title,
    normalize_title,
    start_title_job,
    take_title_event,
    wait_title_event,
)
from fastapi import HTTPException

# 各接口的 user 现在从登录令牌解析，直接调用时显式传入
_USER = UserRecord(
    user_id="user_001", username="user_001", display_name="ppl"
)


def _session(session_id="s1", title=DEFAULT_SESSION_TITLE) -> SessionRecord:
    return SessionRecord(
        id=session_id,
        user_id="user_001",
        title=title,
        created_at="t1",
        updated_at="t1",
    )


def test_normalize_title_strips_and_truncates():
    assert normalize_title('  “北京今天天气”  ', "fallback") == "北京今天天气"
    assert normalize_title("一二三四五六七八九十一二三四五六", "") == "一二三四五六七八九十一二三四五六"[:16]
    assert normalize_title("   ", "帮我写周报") == "帮我写周报"
    assert normalize_title("", "") == DEFAULT_SESSION_TITLE


def test_normalize_title_rejects_copied_question():
    question = "你能去豆包的水印吗"
    assert normalize_title(question, question=question) == "去豆包的水印"
    assert normalize_title("", question=question) == "去豆包的水印"
    assert normalize_title("豆包水印去除方法", question=question) == "豆包水印去除方法"


def test_normalize_title_never_returns_default_with_question():
    # LLM 对无主题问候可能输出默认标题，应回退为压缩后的问题
    assert normalize_title("新会话", question="hello") == "hello"
    assert normalize_title("新会话", question="你好呀！") == "你好"
    assert normalize_title("新会话", question="") == DEFAULT_SESSION_TITLE


def test_create_session_returns_default_title(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.api.routers.session.store.create_session",
        lambda user_id, key: _session("sid-1"),
    )
    result = create_session(
        SessionCreateRequest(idempotency_key="idem-key-1"), user=_USER
    )
    assert result.session_id == "sid-1"
    assert result.title == DEFAULT_SESSION_TITLE


def test_create_session_is_idempotent_at_api(monkeypatch):
    calls = {"n": 0}

    def fake_create(user_id, key):
        calls["n"] += 1
        assert key == "same-key-xx"
        return _session("sid-same")

    monkeypatch.setattr("ai_agent.api.routers.session.store.create_session", fake_create)
    first = create_session(
        SessionCreateRequest(idempotency_key="same-key-xx"), user=_USER
    )
    second = create_session(
        SessionCreateRequest(idempotency_key="same-key-xx"), user=_USER
    )
    assert first.session_id == second.session_id == "sid-same"
    assert calls["n"] == 2


def test_create_session_raises_when_store_fails(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr("ai_agent.api.routers.session.store.create_session", fail)
    with pytest.raises(AgentError, match="创建会话失败"):
        create_session(
            SessionCreateRequest(idempotency_key="idem-key-1"), user=_USER
        )


def test_list_sessions_returns_user_sessions(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.api.routers.session.store.list_sessions",
        lambda user_id, limit=100: [
            _session("s1", "天气咨询"),
            _session("s2", DEFAULT_SESSION_TITLE),
        ],
    )
    result = list_sessions(user=_USER, limit=100)
    assert result.user_id == "user_001"
    assert [item.title for item in result.sessions] == ["天气咨询", DEFAULT_SESSION_TITLE]


def test_rename_session_api(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.api.routers.session.store.update_session_title",
        lambda *_args, **_kwargs: "豆包去水印",
    )
    monkeypatch.setattr(
        "ai_agent.api.routers.session.store.get_session",
        lambda *_args, **_kwargs: _session("s1", "豆包去水印"),
    )
    result = rename_session(
        SessionRenameRequest(session_id="s1", title="豆包去水印"), user=_USER
    )
    assert result.title == "豆包去水印"


def test_delete_session_api(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.api.routers.session.store.delete_session",
        lambda *_args, **_kwargs: True,
    )
    assert delete_session(SessionDeleteRequest(session_id="s1"), user=_USER) == {
        "ok": True
    }


def test_chat_404_when_session_missing(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.api.routers.chat.store.get_session", lambda *_args, **_kwargs: None
    )
    with pytest.raises(HTTPException) as exc:
        chat(
            ChatRequest(message="你好", chat_session_id="missing"), user=_USER
        )
    assert exc.value.status_code == 404


def test_chat_skips_session_check_for_default_session(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("default 会话不应查询数据库")

    monkeypatch.setattr("ai_agent.api.routers.chat.store.get_session", boom)
    response = chat(ChatRequest(message="你好"), user=_USER)
    assert response.status_code == 200


def test_sse_ends_after_done_and_starts_title_job(monkeypatch):
    future: Future[str | None] = Future()
    future.set_result("天气咨询")
    calls: list[tuple[str, str]] = []

    def fake_start(user_id, session_id, question, answer=""):
        calls.append((question, answer))
        return future

    def fake_stream(user_input, user_id="default", chat_session_id="default", **_kwargs):
        yield {"type": "text", "content": "今天晴"}
        yield {"type": "text", "content": "，20度"}
        yield {"type": "done"}

    monkeypatch.setattr("ai_agent.api.routers.chat.start_title_job", fake_start)
    monkeypatch.setattr("ai_agent.api.routers.chat.run_stream", fake_stream)

    # chat_sse_events 结束时会释放并发槽（正常由路由层先 acquire），直连调用
    # 必须先自己拿一个，否则 BoundedSemaphore 会因超额释放报 ValueError
    assert _CHAT_SLOTS.acquire(blocking=False)
    events = [
        json.loads(chunk[6:].strip())
        for chunk in chat_sse_events("user_001", "s1", "北京天气")
    ]
    types = [event["type"] for event in events]
    assert calls == [("北京天气", "今天晴，20度")]
    assert types == ["text", "text", "done"]
    assert not any(event["type"] == "title_updated" for event in events)


def test_get_session_title(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.api.routers.session.store.get_session",
        lambda *_args, **_kwargs: _session("s1", "天气咨询"),
    )
    result = get_session_title(user=_USER, session_id="s1")
    assert result.session_id == "s1"
    assert result.title == "天气咨询"


def test_get_session_title_default_session(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("default 会话不应查询数据库")

    monkeypatch.setattr("ai_agent.api.routers.session.store.get_session", boom)
    result = get_session_title(user=_USER, session_id="default")
    assert result.title == DEFAULT_SESSION_TITLE


def test_get_session_title_404(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.api.routers.session.store.get_session",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(HTTPException) as exc:
        get_session_title(user=_USER, session_id="missing")
    assert exc.value.status_code == 404


def test_generate_session_title_uses_question_and_answer(monkeypatch):
    captured: dict = {}

    def fake_complete(messages, **kwargs):
        captured["content"] = messages[-1]["content"]
        captured.update(kwargs)
        return "  “天气咨询” "

    monkeypatch.setattr("ai_agent.tasks.title.complete_chat", fake_complete)
    assert generate_session_title("北京天气怎么样", "今天晴，20度") == "天气咨询"
    assert "北京天气怎么样" in captured["content"]
    assert "今天晴，20度" in captured["content"]
    assert captured["extra_body"]["enable_thinking"] is False
    assert captured["max_tokens"] == 48


def test_generate_session_title_does_not_echo_question(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.tasks.title.complete_chat",
        lambda *args, **kwargs: "你能去豆包的水印吗",
    )
    assert generate_session_title("你能去豆包的水印吗", "可以尝试去水印") == "去豆包的水印"


def test_start_title_job_skips_named_session(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.tasks.title.store.get_session",
        lambda *_args, **_kwargs: _session("s1", "已有标题"),
    )
    assert start_title_job("user_001", "s1", "你好") is None


def test_take_title_event_waits_until_done():
    future: Future[str | None] = Future()
    assert take_title_event(future, "s1") is None
    future.set_result("周报提纲")
    event = wait_title_event(future, "s1", timeout=0.1)
    assert event == {"type": "title_updated", "session_id": "s1", "title": "周报提纲"}


def test_wait_title_event_returns_none_on_timeout():
    future: Future[str | None] = Future()
    assert wait_title_event(future, "s1", timeout=0.01) is None
