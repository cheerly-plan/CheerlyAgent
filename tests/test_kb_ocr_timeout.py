"""整页 OCR 的超时语义：单页超时只跳过该页，不熔断后续页。"""

from ai_agent.kb.ingest import ocr


class _TimeoutClient:
    def __init__(self, calls, fail_pages=None):
        self.calls = calls
        self.fail_pages = fail_pages or set()

        class _Completions:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                from openai import APITimeoutError

                self._outer.calls.append(kwargs)
                # 前两页慢、后面的页正常
                if len(self._outer.calls) <= len(self._outer.fail_pages):
                    raise APITimeoutError(request=kwargs)  # type: ignore[arg-type]
                raise AssertionError("不应该再被调用")

        class _Chat:
            def __init__(self, outer):
                self.completions = _Completions(outer)

        self.chat = _Chat(self)


def test_timeout_skips_page_without_disabling_ocr(monkeypatch):
    calls: list[dict] = []
    client = _TimeoutClient(calls, fail_pages={1, 2})
    monkeypatch.setattr(ocr, "_client", client)
    monkeypatch.setattr(ocr, "_disabled_reason", None)
    monkeypatch.setattr(ocr, "_consecutive_failures", 0)
    monkeypatch.setattr(ocr.settings, "kb_ocr_model", "fake-ocr")

    from openai import APITimeoutError

    # 两页都超时：两次都只返回 None，OCR 不能被停用
    try:
        assert ocr._recognize(b"png") is None
    except APITimeoutError:
        pass
    assert ocr.disabled_reason() is None


def test_timeout_does_not_raise_or_disable_after_threshold(monkeypatch):
    """超过旧的 kb_ocr_max_failures 阈值也不再熔断。"""
    calls: list[dict] = []
    client = _TimeoutClient(calls, fail_pages={999})
    monkeypatch.setattr(ocr, "_client", client)
    monkeypatch.setattr(ocr, "_disabled_reason", None)
    monkeypatch.setattr(ocr, "_consecutive_failures", 0)
    monkeypatch.setattr(ocr.settings, "kb_ocr_model", "fake-ocr")
    monkeypatch.setattr(ocr.settings, "kb_ocr_max_failures", 1)

    assert ocr._recognize(b"png") is None
    assert ocr.disabled_reason() is None


def test_fatal_config_error_still_disables(monkeypatch):
    """模型不存在这类配置错误才停用——重试只会把同样的 404 刷一遍。"""
    from types import SimpleNamespace

    from openai import NotFoundError

    class _Bad:
        def __init__(self):
            response = SimpleNamespace(
                request=SimpleNamespace(), status_code=404, headers={}, json=lambda: {}
            )

            def _fail(**kwargs):
                raise NotFoundError("Model not exist", response=response, body=None)

            class _Chat:
                def __init__(self):
                    self.completions = type("C", (), {"create": staticmethod(_fail)})()

            self.chat = _Chat()

    monkeypatch.setattr(ocr, "_client", _Bad())
    monkeypatch.setattr(ocr, "_disabled_reason", None)
    monkeypatch.setattr(ocr.settings, "kb_ocr_model", "fake-ocr")

    assert ocr._recognize(b"png") is None
    assert ocr.disabled_reason() is not None
    ocr._disabled_reason = None  # 不污染其它用例


def test_page_timeout_prefers_specific_setting(monkeypatch):
    monkeypatch.setattr(ocr.settings, "kb_ocr_page_timeout_seconds", 42.0)
    monkeypatch.setattr(ocr.settings, "kb_ocr_timeout_seconds", 180.0)
    assert ocr._page_timeout() == 42.0

    monkeypatch.setattr(ocr.settings, "kb_ocr_page_timeout_seconds", 0.0)
    assert ocr._page_timeout() == 180.0
