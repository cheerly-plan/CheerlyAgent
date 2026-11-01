"""整页 OCR：开关、解码参数、熔断，以及正文以 OCR 为准、章节来自文本层。"""

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


def test_parse_pdf_uses_ocr_body_and_text_layer_section(tmp_path, monkeypatch):
    from ai_agent.kb.parsing import TABLE, TEXT, Segment

    path = _pdf(tmp_path)
    monkeypatch.setattr(
        "ai_agent.kb.parsing._pdf_text",
        lambda _path: ([Segment("文本层旧正文", 2, TEXT, "第十节 财务报告")], []),
    )
    monkeypatch.setattr(
        "ai_agent.kb.ocr.ocr_pdf",
        lambda _path: [
            Segment(
                "说明\n\n母公司利润表\n<table><tr><td>2,721,444</td></tr></table>\n附注",
                2,
                TEXT,
            )
        ],
    )
    segments = parse_segments(path)
    assert [s.kind for s in segments] == [TEXT, TABLE, TEXT]
    assert all(s.section == "第十节 财务报告" for s in segments)
    assert segments[1].scope == "parent"
    assert segments[1].text.startswith("母公司利润表")
    assert "文本层旧正文" not in "".join(s.text for s in segments)
    assert "2,721,444" in segments[1].text


def test_cover_fields_dropped_by_ocr_are_restored_from_text_layer(tmp_path, monkeypatch):
    from ai_agent.kb.parsing import TEXT, Segment

    monkeypatch.setattr(
        "ai_agent.kb.parsing._pdf_text",
        lambda _path: (
            [
                Segment(
                    "证券代码：601988\n证券简称：中国银行\n中国银行股份有限公司",
                    1,
                    TEXT,
                )
            ],
            [],
        ),
    )
    monkeypatch.setattr(
        "ai_agent.kb.ocr.ocr_pdf",
        lambda _path: [Segment("# 中国银行股份有限公司\n2026年第三季度报告", 1, TEXT)],
    )
    segments = parse_segments(_pdf(tmp_path))
    assert segments[0].text.startswith("证券代码：601988\n证券简称：中国银行\n")
    assert segments[0].text.count("证券代码") == 1


def test_cover_fields_already_in_ocr_are_not_duplicated(tmp_path, monkeypatch):
    from ai_agent.kb.parsing import TEXT, Segment

    monkeypatch.setattr(
        "ai_agent.kb.parsing._pdf_text",
        lambda _path: (
            [
                Segment(
                    "证券代码：000625（200625）    证券简称：长安汽车    公告编号：2026-92",
                    1,
                    TEXT,
                )
            ],
            [],
        ),
    )
    monkeypatch.setattr(
        "ai_agent.kb.ocr.ocr_pdf",
        lambda _path: [
            Segment(
                "证券代码：000625（200625） 证券简称：长安汽车 公告编号：2026-92\n# 重庆长安汽车",
                1,
                TEXT,
            )
        ],
    )
    segments = parse_segments(_pdf(tmp_path))
    assert segments[0].text.count("证券代码") == 1
    assert segments[0].text.count("公告编号") == 1


def test_intro_sentence_before_unit_joins_the_table():
    from ai_agent.kb.parsing import TABLE, TEXT, _split_ocr_markdown

    text = (
        "主营业务：从事与可再生能源相关的产品和服务分销。\n"
        "股东及持股情况（本次交易前）：Yuma Holding GmbH 持有 100%股权。\n"
        "主要财务数据：标的公司按照德国会计准则编制财务报表，"
        "其最近一年又一期的主要财务数据（未经审计）如下：\n"
        "\n"
        "单位：万欧元\n"
        "<table><tr><td>资产总额</td><td>426.58</td></tr></table>\n"
        "标的公司产权清晰，不存在抵押。\n"
    )
    segments, _carried, pending = _split_ocr_markdown(text, 2, None, None)
    assert [s.kind for s in segments] == [TEXT, TABLE, TEXT]
    assert "主营业务" in segments[0].text
    assert "股东及持股情况" in segments[0].text
    assert "如下" not in segments[0].text
    assert segments[1].text.startswith("主要财务数据：")
    assert "单位：万欧元" in segments[1].text
    assert "<table>" in segments[1].text
    assert segments[2].text.startswith("标的公司产权清晰")
    assert pending == ""


def test_title_not_flush_against_table_still_joins_it():
    from ai_agent.kb.parsing import TABLE, TEXT, _split_ocr_markdown

    text = (
        "##一、主要财务数据\n\n"
        "##（一）主要会计数据和财务指标\n\n"
        "公司是否需追溯调整或重述以前年度会计数据\n"
        "□ 是 √ 否\n"
        "<table><tr><td>营业收入</td><td>1</td></tr></table>"
    )
    segments, _carried, pending = _split_ocr_markdown(text, 1, "一、主要财务数据", None)
    assert [s.kind for s in segments] == [TABLE]
    assert segments[0].text.startswith("##一、主要财务数据\n")
    assert "公司是否需追溯调整" in segments[0].text
    assert "□ 是 √ 否" in segments[0].text
    assert pending == ""


def test_title_at_end_of_previous_page_joins_next_table():
    from ai_agent.kb.parsing import TABLE, _split_ocr_markdown

    page1 = (
        "<table><tr><td>总资产</td><td>1</td></tr></table>\n"
        "##（二）非经常性损益项目和金额\n\n"
        "√ 适用 □ 不适用\n\n"
        "单位：元\n"
    )
    _first, carried, pending = _split_ocr_markdown(page1, 1, "一、主要财务数据", None)
    assert pending.startswith("##（二）非经常性损益项目和金额")
    assert "单位：元" in pending
    page2 = "<table><tr><td>项目</td><td>1</td></tr></table>"
    second, _carried, _pending = _split_ocr_markdown(
        page2, 2, "一、主要财务数据", carried, pending
    )
    assert [s.kind for s in second] == [TABLE]
    assert second[0].text.startswith("##（二）非经常性损益项目和金额")
    assert "单位：元" in second[0].text
    assert "<table>" in second[0].text


def test_headings_above_table_stay_on_the_table():
    from ai_agent.kb.parsing import TABLE, TEXT, _split_ocr_markdown

    text = (
        "前文说明了口径变化，这段要留在正文。\n\n"
        "##一、主要财务数据\n"
        "##1.1 基于中国企业会计准则编制的主要会计数据和财务指标\n"
        "单位：百万元人民币（另有说明者除外）\n"
        "<table><tr><td>营业收入</td><td>169,462</td></tr></table>"
    )
    segments, _carried, _pending = _split_ocr_markdown(text, 2, "一、主要财务数据", None)
    assert [s.kind for s in segments] == [TEXT, TABLE]
    assert segments[0].text == "前文说明了口径变化，这段要留在正文。"
    assert segments[1].text.startswith("##一、主要财务数据\n##1.1 ")
    assert "单位：百万元人民币" in segments[1].text
    assert "<table>" in segments[1].text


def test_statement_scope_skips_unit_line_and_carries_to_next_page():
    from ai_agent.kb.parsing import TEXT, _split_ocr_markdown

    page1 = (
        "## （二）财务报表\n## 合并资产负债表\n2026年9月30日\n"
        "编制单位：中国黄金\n单位：元 币种：人民币\n"
        "<table><tr><td>货币资金</td><td>1</td></tr></table>\n"
        "公司负责人：刘科军\n## 合并利润表\n"
    )
    first, carried, _pending = _split_ocr_markdown(page1, 8, "四、季度财务报表", None)
    tables = [s for s in first if s.kind != TEXT]
    assert tables[0].scope == "consolidated"
    assert carried == "consolidated"
    page2 = "<table><tr><td>营业收入</td><td>2</td></tr></table>"
    second, carried, _pending = _split_ocr_markdown(
        page2, 11, "四、季度财务报表", carried
    )
    assert second[0].scope == "consolidated"
    page3 = (
        "1、母公司资产负债表\n单位：元\n<table><tr><td>货币资金</td><td>3</td></tr></table>"
    )
    third, carried, _pending = _split_ocr_markdown(page3, 16, "四、季度财务报表", carried)
    assert third[-1].scope == "parent"
    assert carried == "parent"


def test_sentence_mentioning_merger_does_not_set_scope():
    from ai_agent.kb.parsing import _split_ocr_markdown

    text = (
        "本次重组构成同一控制下企业合并，合并报表范围发生变化。\n"
        "<table><tr><td>营业收入</td><td>1</td></tr></table>"
    )
    segments, carried, _pending = _split_ocr_markdown(text, 3, None, None)
    assert segments[-1].scope is None
    assert carried is None


def test_ocr_page_mark_is_dropped_and_not_used_as_caption():
    from ai_agent.kb.parsing import TABLE, TEXT, _split_ocr_markdown

    segments, _carried, _pending = _split_ocr_markdown(
        "1 / 16\n证券代码：600916\n\n合并利润表\n<table><tr><td>1</td></tr></table>\n第 2 页 / 共 16 页",
        1,
        None,
    )
    assert [s.kind for s in segments] == [TEXT, TABLE]
    assert segments[0].text.startswith("证券代码：600916")
    assert "1 / 16" not in segments[0].text
    assert segments[1].text.startswith("合并利润表")
    assert segments[1].scope == "consolidated"
    assert "第 2 页" not in "".join(s.text for s in segments)


def test_parse_pdf_falls_back_to_text_layer_when_ocr_missing(tmp_path, monkeypatch):
    from ai_agent.kb.parsing import TEXT, Segment

    monkeypatch.setattr("ai_agent.kb.ocr.ocr_pdf", lambda _path: None)
    monkeypatch.setattr(
        "ai_agent.kb.parsing._pdf_text",
        lambda _path: ([Segment("文本层正文", 1, TEXT, "第一节")], [4]),
    )
    monkeypatch.setattr("ai_agent.kb.figures._pdf_figures", lambda *_a, **_k: [])
    segments = parse_segments(_pdf(tmp_path))
    assert [(s.text, s.section) for s in segments] == [("文本层正文", "第一节")]


def test_section_by_page_carries_forward():
    from ai_agent.kb.parsing import TEXT, Segment, _section_by_page

    pages = _section_by_page(
        [
            Segment("a", 1, TEXT, "第一节"),
            Segment("b", 3, TEXT, "第二节"),
        ]
    )
    assert pages == {1: "第一节", 2: "第一节", 3: "第二节"}
