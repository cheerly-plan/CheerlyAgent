"""整页 OCR：开关、解码参数、熔断，以及它优先于文字层。"""

import httpx2
import pymupdf
import pytest
from openai import APITimeoutError, NotFoundError

from ai_agent.core.config import settings
from ai_agent.kb import ocr
from ai_agent.kb.parsing import parse_segments
from ai_agent.prompts.kb import OCR_DOCUMENT_PROMPT


@pytest.fixture(autouse=True)
def _reset_ocr_state(monkeypatch):
    monkeypatch.setattr(ocr, "_disabled_reason", None)
    monkeypatch.setattr(ocr, "_consecutive_failures", 0)
    monkeypatch.setattr(settings, "kb_ocr_model", "")
    monkeypatch.setattr(settings, "kb_ocr_max_pages", 0)
    monkeypatch.setattr(settings, "kb_ocr_concurrency", 1)
    monkeypatch.setattr(settings, "kb_ocr_max_failures", 2)


class _FakeClient:
    def __init__(self, *, error=None, content="## 正文\n\n第一段"):
        self.error = error
        self.content = content
        self.calls = 0
        self.kwargs: list[dict] = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        if self.error:
            raise self.error
        message = type("Message", (), {"content": self.content})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice]})()


def _pdf(tmp_path, pages: int = 1):
    path = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {i + 1}")
    doc.save(path)
    doc.close()
    return path


def _not_found() -> NotFoundError:
    request = httpx2.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx2.Response(
        404, request=request, json={"error": {"message": "Model not exist."}}
    )
    return NotFoundError("Error code: 404 - Model not exist.", response=response, body=None)


def test_disabled_when_model_empty(tmp_path, monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(ocr, "_client", fake)
    assert ocr.ocr_pdf(_pdf(tmp_path)) is None
    assert fake.calls == 0


def test_page_uses_xiaomi_prompt_and_greedy_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "kb_ocr_model", "longwayxu/xiaomi-ocr-0:bf16")
    fake = _FakeClient(content="营业收入 12.3 亿")
    monkeypatch.setattr(ocr, "_client", fake)

    segments = ocr.ocr_pdf(_pdf(tmp_path, pages=2))

    assert segments is not None
    assert [s.page for s in segments] == [1, 2]
    assert segments[0].text == "营业收入 12.3 亿"
    assert fake.calls == 2
    call = fake.kwargs[0]
    assert call["model"] == "longwayxu/xiaomi-ocr-0:bf16"
    assert call["temperature"] == 0
    assert call["top_p"] == 0.8
    assert call["max_tokens"] == settings.kb_ocr_max_tokens
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    parts = call["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[1]["text"] == OCR_DOCUMENT_PROMPT


def test_model_not_found_stops_later_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "kb_ocr_model", "missing-ocr")
    fake = _FakeClient(error=_not_found())
    monkeypatch.setattr(ocr, "_client", fake)

    assert ocr.ocr_pdf(_pdf(tmp_path, pages=3)) is None
    assert fake.calls == 1
    assert ocr.available() is False
    assert "missing-ocr" in (ocr.disabled_reason() or "")


def test_timeouts_trip_breaker(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "kb_ocr_model", "slow-ocr")
    monkeypatch.setattr(settings, "kb_ocr_max_failures", 2)
    fake = _FakeClient(error=APITimeoutError(request=httpx2.Request("POST", "https://x")))
    monkeypatch.setattr(ocr, "_client", fake)

    assert ocr.ocr_pdf(_pdf(tmp_path, pages=4)) is None
    assert fake.calls == 2
    assert ocr.available() is False


def test_parse_pdf_skips_text_layer_when_ocr_hits(tmp_path, monkeypatch):
    from ai_agent.kb.parsing import Segment

    path = _pdf(tmp_path)
    monkeypatch.setattr(
        "ai_agent.kb.ocr.ocr_pdf", lambda _path: [Segment("识别正文", 1)]
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("文字层不该再跑")

    monkeypatch.setattr("ai_agent.kb.parsing._pdf_text", boom)
    segments = parse_segments(path)
    assert [(s.text, s.page) for s in segments] == [("识别正文", 1)]
