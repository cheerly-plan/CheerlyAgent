"""图表识别（vision）的开关与熔断。

重点不是识别质量，而是失败时的行为：模型名写错（404 Model not exist）
不该每张图都刷一遍堆栈，也不该让一个 300 页的 PDF 白渲染 20 页。
"""

import httpx2  # openai SDK 自带的 httpx 分支，构造异常对象要用它
import pytest
from openai import APIConnectionError, APITimeoutError, NotFoundError

from ai_agent.core.config import settings
from ai_agent.kb import vision


@pytest.fixture(autouse=True)
def _reset_vision_state(monkeypatch):
    """每个用例都从「已启用、未熔断」开始，结束后自动还原。"""
    monkeypatch.setattr(vision, "_disabled_reason", None)
    monkeypatch.setattr(vision, "_consecutive_failures", 0)
    monkeypatch.setattr(settings, "kb_vision_model", "test-vl")


class _FakeClient:
    """只实现 describe_image 用到的那条链：chat.completions.create。"""

    def __init__(self, *, error=None, content="图里有三条柱状数据"):
        self.error = error
        self.content = content
        self.calls = 0

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        message = type("Message", (), {"content": self.content})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice]})()


def _not_found() -> NotFoundError:
    request = httpx2.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx2.Response(
        404, request=request, json={"error": {"message": "Model not exist."}}
    )
    return NotFoundError("Error code: 404 - Model not exist.", response=response, body=None)


def test_disabled_when_model_empty(monkeypatch):
    monkeypatch.setattr(settings, "kb_vision_model", "")
    fake = _FakeClient()
    monkeypatch.setattr(vision, "_client", fake)
    assert vision.available() is False
    assert vision.describe_image(b"png") is None
    assert fake.calls == 0  # 关闭时一次调用都不发


def test_success_returns_text(monkeypatch):
    fake = _FakeClient(content="折线图：Q3 120.5 亿")
    monkeypatch.setattr(vision, "_client", fake)
    assert vision.describe_image(b"png") == "折线图：Q3 120.5 亿"
    assert vision.available() is True


def test_model_not_found_disables_further_calls(monkeypatch):
    fake = _FakeClient(error=_not_found())
    monkeypatch.setattr(vision, "_client", fake)

    assert vision.describe_image(b"png") is None
    # 配置错误 → 熔断，理由里带上模型名，方便照日志改 .env
    assert vision.available() is False
    assert "test-vl" in (vision.disabled_reason() or "")

    # 后续图片不再发请求（parse 层看到 available() 为假会直接跳过渲染）
    assert vision.describe_image(b"png2") is None
    assert fake.calls == 1


def test_transient_error_keeps_vision_enabled(monkeypatch):
    fake = _FakeClient(error=APIConnectionError(request=httpx2.Request("POST", "https://x")))
    monkeypatch.setattr(vision, "_client", fake)

    assert vision.describe_image(b"png") is None
    assert vision.available() is True  # 网络抖动不该熔断
    assert vision.disabled_reason() is None


def test_unsupported_mime_skipped_without_call(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(vision, "_client", fake)
    assert vision.describe_image(b"emf", "image/x-emf") is None
    assert fake.calls == 0


def _timeout() -> APITimeoutError:
    return APITimeoutError(request=httpx2.Request("POST", "https://x"))


def test_consecutive_timeouts_disable_vision(monkeypatch):
    """本地视觉模型卡死时，不能把 20 页 × 超时上限的等待都挨一遍。

    SDK 默认超时 600s，一张图就能堵十分钟；连续两次超时说明这台机器
    跑不动这个模型，熔断掉让剩余页连渲染都不做。
    """
    monkeypatch.setattr(settings, "kb_vision_max_failures", 2)
    fake = _FakeClient(error=_timeout())
    monkeypatch.setattr(vision, "_client", fake)

    assert vision.describe_image(b"png") is None
    assert vision.available() is True  # 第一次只是警告，可能是偶发
    assert vision.describe_image(b"png") is None
    # 第二次 → 熔断，后续直接跳过
    assert vision.available() is False
    assert "超时" in (vision.disabled_reason() or "")
    assert vision.describe_image(b"png") is None
    assert fake.calls == 2


def test_success_resets_timeout_counter(monkeypatch):
    """中间成功一次就把连击清零：偶发超时不累积成熔断。"""
    monkeypatch.setattr(settings, "kb_vision_max_failures", 2)
    fake = _FakeClient(error=_timeout())
    monkeypatch.setattr(vision, "_client", fake)
    assert vision.describe_image(b"png") is None

    monkeypatch.setattr(vision, "_client", _FakeClient(content="柱状图"))
    assert vision.describe_image(b"png") == "柱状图"
    assert vision.available() is True

    monkeypatch.setattr(vision, "_client", _FakeClient(error=_timeout()))
    assert vision.describe_image(b"png") is None
    assert vision.available() is True  # 计数已清零，这次只是第 1 次
