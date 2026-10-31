import json
import urllib.error
from unittest.mock import patch

import pytest

from ai_agent.memory import gate
from ai_agent.memory.gate import jev_noul, jev_noul_batch


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    """默认未配 key，避免真实网络调用。"""
    monkeypatch.setattr(gate.settings, "jev_api_key", "")


def test_jev_noul_returns_none_without_key():
    with patch("ai_agent.core.http._fetch") as urlopen:
        assert jev_noul("这条消息是否紧急？", "state") is None
    urlopen.assert_not_called()


def test_jev_noul_parses_probability(monkeypatch):
    monkeypatch.setattr(gate.settings, "jev_api_key", "k-test")
    captured: dict = {}

    def fake_fetch(req, read_timeout):
        captured["url"] = req.full_url
        captured["auth"] = req.get_header("Authorization")
        captured["body"] = json.loads(req.data.decode())
        return {"model": "jev-1.13.0", "answers": {"gate": {"type": "noul", "noul": 0.87}}}

    with patch("ai_agent.core.http._fetch", side_effect=fake_fetch):
        assert jev_noul("这条消息是否紧急？", "客户被重复扣款") == pytest.approx(0.87)

    assert captured["url"].endswith("/v1/systemone")
    assert captured["auth"] == "Bearer k-test"
    assert captured["body"]["questions"]["gate"]["type"] == "noul"
    assert captured["body"]["state"] == "客户被重复扣款"


def test_jev_noul_missing_field_returns_none(monkeypatch):
    monkeypatch.setattr(gate.settings, "jev_api_key", "k")
    with patch(
        "ai_agent.core.http._fetch",
        return_value={"answers": {}},
    ):
        assert jev_noul("q", "state") is None


def test_jev_noul_non_numeric_returns_none(monkeypatch):
    monkeypatch.setattr(gate.settings, "jev_api_key", "k")
    # bool 是 int 的子类，必须排除
    with patch(
        "ai_agent.core.http._fetch",
        return_value={"answers": {"gate": {"type": "noul", "noul": True}}},
    ):
        assert jev_noul("q", "state") is None


def test_jev_noul_network_error_returns_none(monkeypatch):
    monkeypatch.setattr(gate.settings, "jev_api_key", "k")
    with patch(
        "ai_agent.core.http._fetch",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        assert jev_noul("q", "state") is None


def test_jev_batch_parses_multiple_answers(monkeypatch):
    monkeypatch.setattr(gate.settings, "jev_api_key", "k")
    captured: dict = {}

    def fake_fetch(req, read_timeout):
        captured["body"] = json.loads(req.data.decode())
        return {
            "model": "jev-1.13.0",
            "answers": {
                "fabrication": {"type": "noul", "noul": 0.92},
                "omission": {"type": "noul", "noul": 0.31},
            },
        }

    with patch("ai_agent.core.http._fetch", side_effect=fake_fetch):
        result = jev_noul_batch(
            {"fabrication": "是否编造", "omission": "是否遗漏"}, "state"
        )
    assert result == {"fabrication": pytest.approx(0.92), "omission": pytest.approx(0.31)}
    assert set(captured["body"]["questions"]) == {"fabrication", "omission"}


def test_jev_batch_any_missing_answer_returns_none(monkeypatch):
    """半套概率不可用：整体返回 None，避免调用方拿不完整结果决策。"""
    monkeypatch.setattr(gate.settings, "jev_api_key", "k")
    with patch(
        "ai_agent.core.http._fetch",
        return_value={"answers": {"fabrication": {"type": "noul", "noul": 0.5}}},
    ):
        assert jev_noul_batch({"fabrication": "a", "omission": "b"}, "state") is None
