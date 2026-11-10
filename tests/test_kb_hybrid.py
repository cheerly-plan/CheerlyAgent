"""按页混合解析：密页用文字层，只有稀字页才送整页 OCR。

回归的是「长年报 OCR 熔断后只剩前 30 页」：旧逻辑只要 OCR 非空就整本采用，
半截 OCR 把后面一百多页的电子文本全盖掉了。混合后 OCR 没产出的页必须
继续保留文字层。
"""

import shutil
import tempfile
from pathlib import Path

import pymupdf
import pytest

from ai_agent.core.config import settings
from ai_agent.kb.ingest import figures
from ai_agent.kb.ingest.parsing import (
    TABLE,
    TEXT,
    Segment,
    _sparse_pages,
    parse_segments,
)

# 模块导入时保存真函数：autouse fixture 会把 _pdf_figures 打桩，本文件里
# 唯一要测它的用例得先用它把桩换回来。
_REAL_PDF_FIGURES = figures._pdf_figures


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(settings, "kb_ocr_model", "longwayxu/xiaomi-ocr-0:bf16")
    monkeypatch.setattr(settings, "kb_ocr_min_text_chars", 80)
    # 图表页视觉识别与本用例无关，统一短路，避免真去渲染页面。
    monkeypatch.setattr(
        "ai_agent.kb.ingest.figures._pdf_figures", lambda *_a, **_k: []
    )


@pytest.fixture
def kb_tmp():
    """不用 pytest 的 tmp_path：沙箱里它和固定根目录抢路径会抛 EEXIST。"""
    base = Path(__file__).resolve().parent.parent / ".tmp-kb-test"
    if not base.exists():
        base.mkdir()
    path = Path(tempfile.mkdtemp(dir=base))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _pdf(directory, pages: int = 4):
    path = directory / "doc.pdf"
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), "x")
    doc.save(path)
    doc.close()
    return path


def _text_layer():
    """第 1、3 页密；第 2、4 页稀。第 3 页带一张文字层表格。"""
    return (
        [
            Segment("第1页密页正文", 1, TEXT, "第一节"),
            Segment("第2页稀疏文字", 2, TEXT, "第一节"),
            Segment("| 营业收入 | 1 |", 3, TABLE, "第十节 财务报告"),
            Segment("第3页密页正文", 3, TEXT, "第十节 财务报告"),
            Segment("第4页稀疏文字", 4, TEXT, "第十节 财务报告"),
        ],
        [500, 10, 900, 5],
    )


def _patch_text_layer(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.kb.ingest.parsing._pdf_text", lambda _p, **_k: _text_layer()
    )


def _texts_by_page(segments):
    out: dict[int, str] = {}
    for seg in segments:
        if seg.page is not None:
            out[seg.page] = out.get(seg.page, "") + seg.text
    return out


def test_only_sparse_pages_sent_to_ocr(kb_tmp, monkeypatch):
    _patch_text_layer(monkeypatch)
    captured: dict = {}

    def fake_ocr(_path, **kwargs):
        captured.update(kwargs)
        return [Segment("第2页 OCR 正文", 2, TEXT), Segment("第4页 OCR 正文", 4, TEXT)]

    monkeypatch.setattr("ai_agent.kb.ingest.ocr.ocr_pdf", fake_ocr)
    parse_segments(_pdf(kb_tmp))
    assert captured.get("pages") == [2, 4]


def test_dense_pages_keep_text_layer_and_sparse_use_ocr(kb_tmp, monkeypatch):
    _patch_text_layer(monkeypatch)
    monkeypatch.setattr(
        "ai_agent.kb.ingest.ocr.ocr_pdf",
        lambda _p, **_k: [
            Segment("第2页 OCR 正文", 2, TEXT),
            Segment("第4页 OCR 正文", 4, TEXT),
        ],
    )
    segments = parse_segments(_pdf(kb_tmp))
    by_page = _texts_by_page(segments)

    assert "第1页密页正文" in by_page[1]
    assert "第2页稀疏文字" not in by_page[2]
    assert "第2页 OCR 正文" in by_page[2]
    # 密页的表格段不能被 OCR 挤掉
    assert "营业收入" in by_page[3]
    assert "第3页密页正文" in by_page[3]
    assert "第4页稀疏文字" not in by_page[4]
    assert "第4页 OCR 正文" in by_page[4]
    assert [s.page for s in segments] == sorted(s.page for s in segments)


def test_ocr_breaker_does_not_drop_text_layer(kb_tmp, monkeypatch):
    """OCR 熔断一页没出：整本文字层照旧，重演「只剩前 30 页」的反例。"""
    _patch_text_layer(monkeypatch)
    monkeypatch.setattr("ai_agent.kb.ingest.ocr.ocr_pdf", lambda _p, **_k: None)
    segments = parse_segments(_pdf(kb_tmp))
    joined = "".join(s.text for s in segments)
    for expected in (
        "第1页密页正文",
        "第2页稀疏文字",
        "营业收入",
        "第3页密页正文",
        "第4页稀疏文字",
    ):
        assert expected in joined


def test_partial_ocr_keeps_uncovered_pages(kb_tmp, monkeypatch):
    """只识别出第 2 页时，第 4 页仍用文字层，不被半截 OCR 清空。"""
    _patch_text_layer(monkeypatch)
    monkeypatch.setattr(
        "ai_agent.kb.ingest.ocr.ocr_pdf",
        lambda _p, **_k: [Segment("第2页 OCR 正文", 2, TEXT)],
    )
    by_page = _texts_by_page(parse_segments(_pdf(kb_tmp)))
    assert "第2页 OCR 正文" in by_page[2]
    assert "第2页稀疏文字" not in by_page[2]
    assert "第4页稀疏文字" in by_page[4]


def test_all_dense_pages_skip_ocr_entirely(kb_tmp, monkeypatch):
    monkeypatch.setattr(
        "ai_agent.kb.ingest.parsing._pdf_text",
        lambda _p, **_k: ([Segment("整本电子文本", 1, TEXT)], [500, 900, 800, 700]),
    )
    calls: list = []

    def fake_ocr(_path, **_kwargs):
        calls.append(1)
        return [Segment("不该被调用", 1, TEXT)]

    monkeypatch.setattr("ai_agent.kb.ingest.ocr.ocr_pdf", fake_ocr)
    segments = parse_segments(_pdf(kb_tmp))
    assert calls == []
    assert "".join(s.text for s in segments) == "整本电子文本"


def test_min_chars_zero_keeps_legacy_full_ocr(kb_tmp, monkeypatch):
    """0 = 关闭混合：无 pages 关键字、OCR 有产出就整本采用（旧行为）。"""
    monkeypatch.setattr(settings, "kb_ocr_min_text_chars", 0)
    _patch_text_layer(monkeypatch)
    captured: dict = {}

    def fake_ocr(_path, **kwargs):
        captured.update(kwargs)
        return [Segment("OCR 整本正文", 1, TEXT)]

    monkeypatch.setattr("ai_agent.kb.ingest.ocr.ocr_pdf", fake_ocr)
    segments = parse_segments(_pdf(kb_tmp))
    assert captured == {}
    assert "".join(s.text for s in segments) == "OCR 整本正文"


def test_sparse_pages_pads_missing_tail(kb_tmp):
    """text_lens 只统计有文字行的页，末尾纯图片页靠总页数补零。"""
    assert _sparse_pages(_pdf(kb_tmp, pages=4), [500, 10, 900], 80) == [2, 4]


def test_figures_skip_pages_covered_by_ocr(kb_tmp, monkeypatch):
    """已被 OCR 覆盖的页不再重复送视觉模型；默认 None 行为不变。"""
    monkeypatch.setattr(
        "ai_agent.kb.ingest.figures._pdf_figures", _REAL_PDF_FIGURES
    )
    monkeypatch.setattr("ai_agent.kb.ingest.figures.vision.available", lambda: True)
    monkeypatch.setattr(
        "ai_agent.kb.ingest.figures._page_has_chart", lambda _page, _tl: True
    )
    monkeypatch.setattr(
        "ai_agent.kb.ingest.figures._describe_parallel",
        lambda jobs: ["图示"] * len(jobs),
    )
    monkeypatch.setattr(settings, "kb_vision_concurrency", 1)
    monkeypatch.setattr(settings, "kb_vision_max_pages", 20)
    path = _pdf(kb_tmp)
    assert [s.page for s in figures._pdf_figures(path, skip_pages={1, 3})] == [2, 4]
    assert len(figures._pdf_figures(path)) == 4
