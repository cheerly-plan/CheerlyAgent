"""整页 OCR：PDF 逐页渲染成图，交给 Xiaomi-OCR-0 抽成 Markdown。

官方 MCP（ocr_pdf 的 page 模式）做的就是这件事：150 DPI 渲页、长边压到
2048、固定英文抽取提示词、temperature=0、关掉 thinking。它的 stdio 服务
拒绝非 127.0.0.1 的推理地址，而这里的 Ollama 在局域网上，所以同一套
page 流程直接打 BASE_URL，不另起 MCP 进程。

模型名为空则整体关闭，PDF 仍走文字层。识别一页都没有时也退回文字层，
避免模型没拉下来时上传直接失败。
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import (
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
)

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.otsl import render_otsl
from ai_agent.kb.parsing import TEXT, Segment
from ai_agent.prompts.kb import OCR_DOCUMENT_PROMPT

logger = get_logger(__name__)

_client = OpenAI(
    base_url=settings.base_url,
    api_key=settings.api_key,
    timeout=settings.kb_ocr_timeout_seconds,
    max_retries=0,
)

_disabled_reason: str | None = None
_consecutive_failures = 0


def enabled() -> bool:
    return bool(settings.kb_ocr_model.strip())


def available() -> bool:
    return enabled() and _disabled_reason is None


def disabled_reason() -> str | None:
    return _disabled_reason


def _disable(reason: str) -> None:
    global _disabled_reason
    if _disabled_reason is not None:
        return
    _disabled_reason = reason
    logger.error(
        "kb 整页 OCR 已停用：%s。修改 KB_OCR_MODEL / API_KEY 后重启即可恢复，"
        "本次解析其余页直接跳过",
        reason,
    )


def _error_detail(err: Exception) -> str:
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        message = (inner or {}).get("message") if isinstance(inner, dict) else None
        if message:
            return str(message)[:120]
    return str(err)[:120]


def _render_page(page, dpi: int, max_edge: int) -> bytes:
    """按 DPI 渲染，长边超过上限就等比缩小。对齐官方 fit_image 的 2048。"""
    import pymupdf

    scale = dpi / 72
    rect = page.rect
    long_px = max(rect.width, rect.height) * scale
    if max_edge > 0 and long_px > max_edge:
        scale *= max_edge / long_px
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return pix.tobytes("png")


def _recognize(png: bytes) -> str | None:
    """一页图 → Markdown。失败返回 None，不抛给解析管道。"""
    global _consecutive_failures
    if not available() or not png:
        return None
    try:
        b64 = base64.b64encode(png).decode()
        resp = _client.chat.completions.create(
            model=settings.kb_ocr_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": OCR_DOCUMENT_PROMPT},
                    ],
                }
            ],
            max_tokens=settings.kb_ocr_max_tokens,
            temperature=0,
            top_p=0.8,
            presence_penalty=0,
            timeout=settings.kb_ocr_timeout_seconds,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = render_otsl((resp.choices[0].message.content or "").strip())
        _consecutive_failures = 0
    except (NotFoundError, AuthenticationError, PermissionDeniedError) as e:
        _disable(
            f"{type(e).__name__}: {_error_detail(e)}"
            f"（当前模型 {settings.kb_ocr_model!r}）"
        )
        return None
    except APITimeoutError:
        _consecutive_failures += 1
        logger.warning(
            "kb 整页 OCR 超时（%.0fs，连续第 %d 次）",
            settings.kb_ocr_timeout_seconds,
            _consecutive_failures,
        )
        if _consecutive_failures >= max(1, settings.kb_ocr_max_failures):
            _disable(
                f"连续 {_consecutive_failures} 次超时（{settings.kb_ocr_timeout_seconds:.0f}s），"
                f"当前模型 {settings.kb_ocr_model!r} 在本机跑不动"
            )
        return None
    except Exception:
        logger.warning("kb 整页 OCR 失败，跳过该页", exc_info=True)
        return None
    return text or None


def _recognize_parallel(pngs: list[bytes]) -> list[str | None]:
    workers = max(1, min(settings.kb_ocr_concurrency, len(pngs)))
    if workers <= 1:
        return [_recognize(png) for png in pngs]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kb-ocr") as pool:
        return list(pool.map(_recognize, pngs))


def ocr_pdf(path: Path) -> list[Segment] | None:
    """逐页识别。未开启、打不开或一页都没有时返回 None，调用方走文字层。"""
    if not available():
        return None
    try:
        import pymupdf
    except ImportError:
        logger.warning("未安装 pymupdf，跳过整页 OCR")
        return None
    try:
        doc = pymupdf.open(path)
    except Exception:
        logger.warning("PDF 打不开，跳过整页 OCR：%s", path.name, exc_info=True)
        return None
    try:
        total = doc.page_count
        limit = settings.kb_ocr_max_pages
        if limit > 0:
            total = min(total, limit)
        if total <= 0:
            return None
        batch = max(1, settings.kb_ocr_concurrency)
        out: list[Segment] = []
        for start in range(0, total, batch):
            if not available():
                break
            pngs: list[tuple[int, bytes]] = []
            for i in range(start, min(start + batch, total)):
                try:
                    pngs.append(
                        (
                            i,
                            _render_page(
                                doc[i], settings.kb_ocr_dpi, settings.kb_ocr_max_edge
                            ),
                        )
                    )
                except Exception:
                    logger.warning("PDF 第 %d 页渲染失败，跳过", i + 1, exc_info=True)
            if not pngs:
                continue
            texts = _recognize_parallel([png for _i, png in pngs])
            for (i, _png), text in zip(pngs, texts):
                if text:
                    out.append(Segment(text, i + 1, TEXT))
        if out:
            logger.info("kb 整页 OCR 完成：%d/%d 页", len(out), total)
        return out or None
    finally:
        doc.close()
