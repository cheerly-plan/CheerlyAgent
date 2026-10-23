from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from ai_agent.api.routers.auth import _client_ip, register
from ai_agent.api.schemas import RegisterRequest
from ai_agent.core.config import settings


def _request(headers: dict[str, str] | None = None, client_host: str = "10.0.0.1"):
    return SimpleNamespace(
        headers=headers or {}, client=SimpleNamespace(host=client_host)
    )


def test_client_ip_ignores_xff_by_default():
    # 默认不信任反代头：伪造 X-Forwarded-For 不能绕过限流
    req = _request({"x-forwarded-for": "1.2.3.4, 5.6.7.8"})
    assert _client_ip(req) == "10.0.0.1"


def test_client_ip_uses_rightmost_hop_when_trusted(monkeypatch):
    monkeypatch.setattr(settings, "trust_forwarded_for", True)
    req = _request({"x-forwarded-for": "1.2.3.4, 5.6.7.8"})
    # 最右侧一跳由自己的可信反代追加，客户端伪造的头只出现在左侧
    assert _client_ip(req) == "5.6.7.8"


def test_client_ip_falls_back_to_socket_when_trusted_but_no_header(monkeypatch):
    monkeypatch.setattr(settings, "trust_forwarded_for", True)
    assert _client_ip(_request()) == "10.0.0.1"


def test_register_rejects_blank_header_noise(monkeypatch):
    monkeypatch.setattr(settings, "trust_forwarded_for", True)
    assert _client_ip(_request({"x-forwarded-for": " , "} )) == "10.0.0.1"


def test_register_disabled_returns_403(monkeypatch):
    monkeypatch.setattr(settings, "registration_enabled", False)

    def boom(*_args, **_kwargs):
        raise AssertionError("注册关闭时不应触达数据库")

    monkeypatch.setattr("ai_agent.api.routers.auth.store.create_user", boom)
    with pytest.raises(HTTPException) as exc:
        register(
            RegisterRequest(username="mike", password="long-enough-8"),
            _request(),
        )
    assert exc.value.status_code == 403
