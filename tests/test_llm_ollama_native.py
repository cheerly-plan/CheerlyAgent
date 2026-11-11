"""本地 Ollama 原生路由：JSON 任务关思考，自由文本不关（关了会把推理写进正文）。"""

import json

from ai_agent.llm import ollama


class _FakeResp:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()
        self.status = 200

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _capture(monkeypatch, *, content="ok", thinking=""):
    calls: list[dict] = []

    def fake_urlopen(request, timeout=None):
        calls.append(json.loads(request.data))
        return _FakeResp({"message": {"content": content, "thinking": thinking}})

    monkeypatch.setattr(ollama.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_json_task_disables_thinking(monkeypatch):
    calls = _capture(monkeypatch)
    out = ollama.chat(
        [{"role": "user", "content": "hi"}],
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3:4b",
        json_mode=True,
        max_tokens=384,
    )
    assert out == "ok"
    body = calls[0]
    assert body["think"] is False          # 关思考只对结构化任务有效
    assert body["format"] == "json"
    assert body["options"]["num_predict"] == 384
    # 一点点重复惩罚：小模型在长 JSON 里复读会撞上 Ollama 的 repeat limit
    assert body["options"]["repeat_penalty"] == 1.1


def test_freetext_keeps_thinking_and_enlarges_budget(monkeypatch):
    """自由文本不能关思考：关了推理会写进 content，摘要变成「首先，用户的问题…」。"""
    calls = _capture(monkeypatch)
    ollama.chat(
        [{"role": "user", "content": "总结一下"}],
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3:4b",
        json_mode=False,
        max_tokens=512,
    )
    body = calls[0]
    assert "think" not in body
    assert "format" not in body
    assert body["options"]["num_predict"] >= ollama._FREETEXT_MIN_PREDICT


def test_messages_are_flattened_to_role_content(monkeypatch):
    calls = _capture(monkeypatch)
    ollama.chat(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        base_url="http://127.0.0.1:11434/v1",
        model="m",
        json_mode=True,
    )
    assert calls[0]["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


def test_empty_content_with_thinking_is_reported(monkeypatch, caplog):
    _capture(monkeypatch, content="", thinking="想了很久")
    with caplog.at_level("WARNING"):
        out = ollama.chat(
            [{"role": "user", "content": "hi"}],
            base_url="http://127.0.0.1:11434/v1",
            model="m",
            json_mode=False,
        )
    assert out == ""
    assert "thinking" in caplog.text
