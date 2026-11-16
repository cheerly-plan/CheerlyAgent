"""企业微信推送客户端与端点测试。

覆盖：凭据缺失、token 缓存、上传→发送链路、token 失效重试、业务错误分类、
端点鉴权/404/503/超限降级。全部走 mock，不产生真实网络调用；端点直接调函数
（项目未装 httpx，不引 TestClient）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from ai_agent.storage.db import UserRecord
from ai_agent.wecom import client
from ai_agent.wecom import (
    WecomError,
    WecomNotConfigured,
    send_file_to_user,
    send_text_to_user,
)


def _is_gettoken(url: str) -> bool:
    return url.split("?")[0].endswith("/gettoken")


def _is_send(url: str) -> bool:
    return url.split("?")[0].endswith("/message/send")


@pytest.fixture
def auth_user() -> UserRecord:
    return UserRecord(user_id="u-1", username="tester", display_name="tester")


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch):
    """默认未配凭据，避免真实网络调用；token 缓存清零防跨测试污染。"""
    monkeypatch.setattr(client.settings, "wecom_corp_id", "")
    monkeypatch.setattr(client.settings, "wecom_agent_secret", "")
    monkeypatch.setattr(client.settings, "wecom_agent_id", "")
    client._token_cache.invalidate()


def _configure(monkeypatch):
    monkeypatch.setattr(client.settings, "wecom_corp_id", "ww-test")
    monkeypatch.setattr(client.settings, "wecom_agent_secret", "secret-test")
    monkeypatch.setattr(client.settings, "wecom_agent_id", "1000002")


# ——— 客户端 ———


def test_send_file_requires_credentials():
    with pytest.raises(WecomNotConfigured):
        send_file_to_user(
            filename="a.pdf", content=b"%PDF", content_type="application/pdf", touser="u1"
        )


def test_send_file_uploads_then_sends(monkeypatch):
    _configure(monkeypatch)
    calls: list[tuple[str, dict | None]] = []

    def fake_request_json(url, body=None, timeout=10, **kw):
        calls.append((url, body))
        if _is_gettoken(url):
            return {"access_token": "tok-1", "expires_in": 7200}
        if _is_send(url):
            return {"errcode": 0, "errmsg": "ok", "invaliduser": ""}
        raise AssertionError(f"unexpected url {url}")

    def fake_request_upload(url, *, field_name, filename, content, content_type, **kw):
        assert field_name == "media"
        assert filename == "a.pdf"
        assert content == b"%PDF"
        return {"type": "file", "media_id": "media-1"}

    with (
        patch("ai_agent.core.http.request_json", side_effect=fake_request_json),
        patch("ai_agent.core.http.request_upload", side_effect=fake_request_upload),
    ):
        result = send_file_to_user(
            filename="a.pdf", content=b"%PDF", content_type="application/pdf", touser="u1"
        )

    assert result["errcode"] == 0
    # 顺序：gettoken → media/upload → message/send
    assert [c[0].split("?")[0] for c in calls] == [
        "https://qyapi.weixin.qq.com/cgi-bin/gettoken",
        "https://qyapi.weixin.qq.com/cgi-bin/message/send",
    ]
    send_body = calls[1][1]
    assert send_body["touser"] == "u1"
    assert send_body["msgtype"] == "file"
    assert send_body["agentid"] == 1000002
    assert send_body["file"]["media_id"] == "media-1"


def test_send_file_token_is_cached_across_calls(monkeypatch):
    """同一进程内 token 只取一次，第二次发文件不重复 gettoken。"""
    _configure(monkeypatch)
    gettoken_calls = 0

    def fake_request_json(url, body=None, timeout=10, **kw):
        nonlocal gettoken_calls
        if _is_gettoken(url):
            gettoken_calls += 1
            return {"access_token": "tok-1", "expires_in": 7200}
        if _is_send(url):
            return {"errcode": 0, "errmsg": "ok"}
        raise AssertionError(url)

    with (
        patch("ai_agent.core.http.request_json", side_effect=fake_request_json),
        patch("ai_agent.core.http.request_upload", return_value={"media_id": "m"}),
    ):
        send_file_to_user(filename="a.pdf", content=b"1", content_type="application/pdf", touser="u1")
        send_file_to_user(filename="b.pdf", content=b"2", content_type="application/pdf", touser="u2")

    assert gettoken_calls == 1


def test_send_file_retries_on_token_expired(monkeypatch):
    """token 失效（40014）时刷 token 重试一次，最终成功。"""
    _configure(monkeypatch)
    gettoken_calls = 0
    send_calls = 0

    def fake_request_json(url, body=None, timeout=10, **kw):
        nonlocal gettoken_calls, send_calls
        if _is_gettoken(url):
            gettoken_calls += 1
            return {"access_token": f"tok-{gettoken_calls}", "expires_in": 7200}
        if _is_send(url):
            send_calls += 1
            if send_calls == 1:
                raise WecomError(40014, "invalid access_token")
            return {"errcode": 0, "errmsg": "ok"}
        raise AssertionError(url)

    with (
        patch("ai_agent.core.http.request_json", side_effect=fake_request_json),
        patch("ai_agent.core.http.request_upload", return_value={"media_id": "m"}),
    ):
        result = send_file_to_user(
            filename="a.pdf", content=b"1", content_type="application/pdf", touser="u1"
        )

    assert result["errcode"] == 0
    assert gettoken_calls == 2
    assert send_calls == 2


def test_send_file_business_error_not_retried(monkeypatch):
    """非 token 类业务错误（如 48002 无发送权限）不重试，直接抛 WecomError。"""
    _configure(monkeypatch)

    def fake_request_json(url, body=None, timeout=10, **kw):
        if _is_gettoken(url):
            return {"access_token": "tok-1", "expires_in": 7200}
        raise WecomError(48002, "api forbidden")

    with (
        patch("ai_agent.core.http.request_json", side_effect=fake_request_json),
        patch("ai_agent.core.http.request_upload", return_value={"media_id": "m"}),
    ):
        with pytest.raises(WecomError) as exc_info:
            send_file_to_user(
                filename="a.pdf", content=b"1", content_type="application/pdf", touser="u1"
            )
    assert exc_info.value.errcode == 48002


# ——— 文本消息 ———


def test_send_text_requires_credentials():
    with pytest.raises(WecomNotConfigured):
        send_text_to_user(touser="u1", text="hello")


def test_send_text_uses_text_msgtype(monkeypatch):
    """文本消息走 msgtype=text，不触文件上传，token 只取一次。"""
    _configure(monkeypatch)
    calls: list[tuple[str, dict | None]] = []

    def fake_request_json(url, body=None, timeout=10, **kw):
        calls.append((url, body))
        if _is_gettoken(url):
            return {"access_token": "tok-1", "expires_in": 7200}
        if _is_send(url):
            return {"errcode": 0, "errmsg": "ok"}
        raise AssertionError(url)

    with patch("ai_agent.core.http.request_json", side_effect=fake_request_json):
        result = send_text_to_user(touser="u1", text="你好")

    assert result["errcode"] == 0
    send_body = calls[1][1]
    assert send_body["msgtype"] == "text"
    assert send_body["text"]["content"] == "你好"
    assert send_body["agentid"] == 1000002
    assert send_body["touser"] == "u1"


def test_send_text_retries_on_token_expired(monkeypatch):
    """文本消息 token 失效同样刷 token 重试一次。"""
    _configure(monkeypatch)
    gettoken_calls = 0
    send_calls = 0

    def fake_request_json(url, body=None, timeout=10, **kw):
        nonlocal gettoken_calls, send_calls
        if _is_gettoken(url):
            gettoken_calls += 1
            return {"access_token": f"tok-{gettoken_calls}", "expires_in": 7200}
        if _is_send(url):
            send_calls += 1
            if send_calls == 1:
                raise WecomError(42001, "access_token expired")
            return {"errcode": 0, "errmsg": "ok"}
        raise AssertionError(url)

    with patch("ai_agent.core.http.request_json", side_effect=fake_request_json):
        result = send_text_to_user(touser="u1", text="hi")

    assert result["errcode"] == 0
    assert gettoken_calls == 2
    assert send_calls == 2


# ——— 端点（直接调函数，不依赖 httpx/TestClient）———


@pytest.fixture
def report_dir(monkeypatch):
    """把 report_output_dir 指到临时目录，避免污染真实 data/reports。"""
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(client.settings, "report_output_dir", Path(td))
        yield Path(td)


def test_push_endpoint_missing_file(auth_user, report_dir):
    from ai_agent.api.routers.wecom import WecomPushRequest, push_report

    with pytest.raises(HTTPException) as exc_info:
        push_report(WecomPushRequest(filename="不存在.pdf", touser="u1"), user=auth_user)
    assert exc_info.value.status_code == 404


def test_push_endpoint_not_configured(monkeypatch, auth_user, report_dir):
    """有文件但未配凭据：503。文件放 report_output_dir 下被 artifact_path 认可。"""
    from ai_agent.api.routers.wecom import WecomPushRequest, push_report

    f = report_dir / "x.md"
    f.write_text("# 标题", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        push_report(WecomPushRequest(filename="x.md", touser="u1"), user=auth_user)
    assert exc_info.value.status_code == 503


def test_push_endpoint_oversize_degrades(monkeypatch, auth_user, report_dir):
    """超 wecom_max_file_bytes 时返回 413，带下载链接，不调用企业微信。"""
    from ai_agent.api.routers.wecom import WecomPushRequest, push_report

    monkeypatch.setattr(client.settings, "wecom_max_file_bytes", 10)
    f = report_dir / "big.md"
    f.write_text("x" * 13, encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        push_report(WecomPushRequest(filename="big.md", touser="u1"), user=auth_user)
    assert exc_info.value.status_code == 413
    assert "/v1/report/files/big.md" in exc_info.value.detail


def test_push_endpoint_success(monkeypatch, auth_user, report_dir):
    """完整成功链路：读文件 → 上传 → 发送，返回 invaliduser 透传。"""
    from ai_agent.api.routers.wecom import WecomPushRequest, push_report

    _configure(monkeypatch)
    f = report_dir / "ok.md"
    f.write_text("# 研报", encoding="utf-8")

    def fake_request_json(url, body=None, timeout=10, **kw):
        if _is_gettoken(url):
            return {"access_token": "tok-1", "expires_in": 7200}
        if _is_send(url):
            return {"errcode": 0, "errmsg": "ok", "invaliduser": "ghost"}
        raise AssertionError(url)

    with (
        patch("ai_agent.core.http.request_json", side_effect=fake_request_json),
        patch("ai_agent.core.http.request_upload", return_value={"media_id": "m"}),
    ):
        result = push_report(WecomPushRequest(filename="ok.md", touser="u1"), user=auth_user)

    assert result["errcode"] == 0
    assert result["invaliduser"] == "ghost"
    assert "未送达" in result["hint"]


def test_message_endpoint_success(monkeypatch, auth_user):
    """文本消息端点：发 text 消息成功，透传 invaliduser。"""
    from ai_agent.api.routers.wecom import WecomMessageRequest, push_message

    _configure(monkeypatch)

    def fake_request_json(url, body=None, timeout=10, **kw):
        if _is_gettoken(url):
            return {"access_token": "tok-1", "expires_in": 7200}
        if _is_send(url):
            return {"errcode": 0, "errmsg": "ok", "invaliduser": "ghost"}
        raise AssertionError(url)

    with patch("ai_agent.core.http.request_json", side_effect=fake_request_json):
        result = push_message(
            WecomMessageRequest(touser="u1", text="hello"), user=auth_user
        )

    assert result["errcode"] == 0
    assert result["invaliduser"] == "ghost"
    assert "未送达" in result["hint"]


def test_message_endpoint_not_configured(monkeypatch, auth_user):
    """未配凭据：503。"""
    from ai_agent.api.routers.wecom import WecomMessageRequest, push_message

    with pytest.raises(HTTPException) as exc_info:
        push_message(WecomMessageRequest(touser="u1", text="hi"), user=auth_user)
    assert exc_info.value.status_code == 503


def test_message_endpoint_too_long(monkeypatch, auth_user):
    """正文超 2048 字节：400，不触企业微信。"""
    from ai_agent.api.routers.wecom import WecomMessageRequest, push_message

    _configure(monkeypatch)
    with pytest.raises(HTTPException) as exc_info:
        push_message(
            WecomMessageRequest(touser="u1", text="x" * 2049), user=auth_user
        )
    assert exc_info.value.status_code == 400


# ——— 研报完成通知 ———


def test_notify_disabled_returns_without_thread(monkeypatch):
    """开关关闭：notify_report_done 直接返回，不启动线程。"""
    from ai_agent.wecom import notify

    monkeypatch.setattr(notify.settings, "wecom_report_notify_enabled", False)
    with patch("ai_agent.wecom.notify.threading.Thread") as mock_thread:
        notify.notify_report_done("uuid-1")
    mock_thread.assert_not_called()


def test_notify_worker_uses_username_as_touser(monkeypatch):
    """通知 worker：由 user_id 查 username 当 touser，发 text 消息。"""
    from ai_agent.wecom import notify
    from ai_agent.storage.db import UserRecord

    sent: list[dict] = []

    class FakeStore:
        def get_user_by_id(self, user_id):
            assert user_id == "uuid-1"
            return UserRecord(user_id="uuid-1", username="WeiJiaNan", display_name="韦佳男")

    def fake_send_text_to_user(*, touser, text):
        sent.append({"touser": touser, "text": text})
        return {"errcode": 0, "errmsg": "ok"}

    with (
        patch("ai_agent.storage.store.store", FakeStore()),
        patch("ai_agent.wecom.notify.send_text_to_user", side_effect=fake_send_text_to_user),
    ):
        notify._notify_worker("uuid-1", "")

    assert len(sent) == 1
    assert sent[0]["touser"] == "WeiJiaNan"
    assert "研报" in sent[0]["text"]


def test_notify_worker_includes_title_when_present(monkeypatch):
    """有标题时，通知正文带上「研报《标题》」，让用户知道是哪份。"""
    from ai_agent.wecom import notify
    from ai_agent.storage.db import UserRecord

    sent: list[str] = []

    class FakeStore:
        def get_user_by_id(self, user_id):
            return UserRecord(user_id="u", username="WeiJiaNan", display_name="")

    def fake_send_text_to_user(*, touser, text):
        sent.append(text)
        return {"errcode": 0, "errmsg": "ok"}

    with (
        patch("ai_agent.storage.store.store", FakeStore()),
        patch("ai_agent.wecom.notify.send_text_to_user", side_effect=fake_send_text_to_user),
    ):
        notify._notify_worker("u", "贵州茅台 2024 年报深度解析")

    assert len(sent) == 1
    assert "贵州茅台 2024 年报深度解析" in sent[0]
    assert sent[0].startswith("研报「")


def test_notify_worker_no_title_uses_default(monkeypatch):
    """无标题时退回默认文案，不出现空「」括号。"""
    from ai_agent.wecom import notify
    from ai_agent.storage.db import UserRecord

    sent: list[str] = []

    class FakeStore:
        def get_user_by_id(self, user_id):
            return UserRecord(user_id="u", username="WeiJiaNan", display_name="")

    def fake_send_text_to_user(*, touser, text):
        sent.append(text)
        return {"errcode": 0, "errmsg": "ok"}

    with (
        patch("ai_agent.storage.store.store", FakeStore()),
        patch("ai_agent.wecom.notify.send_text_to_user", side_effect=fake_send_text_to_user),
    ):
        notify._notify_worker("u", "")

    assert sent[0] == "研报已生成完成，请回系统查看并下载。"


def test_notify_worker_user_not_found_swallows(monkeypatch):
    """查不到用户：静默跳过，不抛异常。"""
    from ai_agent.wecom import notify

    class FakeStore:
        def get_user_by_id(self, user_id):
            return None

    with (
        patch("ai_agent.storage.store.store", FakeStore()),
        patch("ai_agent.wecom.notify.send_text_to_user") as mock_send,
    ):
        notify._notify_worker("ghost", "")

    mock_send.assert_not_called()


def test_notify_worker_send_error_swallows(monkeypatch):
    """发送抛异常：吞掉，不向上抛。"""
    from ai_agent.wecom import notify
    from ai_agent.storage.db import UserRecord

    class FakeStore:
        def get_user_by_id(self, user_id):
            return UserRecord(user_id="u", username="WeiJiaNan", display_name="")

    def fake_send_text_to_user(*, touser, text):
        raise WecomError(48002, "api forbidden")

    with (
        patch("ai_agent.storage.store.store", FakeStore()),
        patch("ai_agent.wecom.notify.send_text_to_user", side_effect=fake_send_text_to_user),
    ):
        # 不应抛异常
        notify._notify_worker("u", "")
