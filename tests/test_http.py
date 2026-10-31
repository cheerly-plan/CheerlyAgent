"""core.http 的请求透传与异常传播。"""

import urllib.error
from unittest.mock import patch

import pytest

from ai_agent.core import http


def test_request_json_passes_through_when_fetch_fast():
    """正常路径：_fetch 迅速返回时，结果原样透传。"""
    with patch("ai_agent.core.http._fetch", return_value={"ok": 1}):
        assert http.request_json("http://example/", timeout=1) == {"ok": 1}


def test_request_json_propagates_fetch_exception():
    """_fetch 抛出的网络异常应透传到调用方，由上层（如 gate/web_search）决定降级。"""
    with patch(
        "ai_agent.core.http._fetch",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with pytest.raises(urllib.error.URLError):
            http.request_json("http://example/", timeout=1)
