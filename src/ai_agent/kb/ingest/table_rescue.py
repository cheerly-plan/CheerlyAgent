"""表格页 OCR 兜底：文字层搞不定的表，交给整页 OCR 重出一次。

为什么要这一层：`find_tables` 只认表格线，遇到两种版式就交白卷——

* **表头丢了**：列名退化成占位「列2/列3/列4」，值全对但不知道是哪一列；
* **原生崩溃**：某些页直接 SIGSEGV，整页表格凭空消失。

这两种页的**像素是好的**，只是文字层拿不到结构。整页 OCR（Xiaomi-OCR-0）
直接输出带 rowspan/colspan 的 HTML 表，实测在这类页上能补回列名。

但它不能无脑用：宽表（所有者权益变动表那种竖线密的）OCR 会切得更碎，
而且视觉模型有可能"读"出原文没有的数字。所以这里设三道闸：

1. **只在文字层已经坏掉的页上跑**（出现占位列头，或 find_tables 崩溃）；
2. **数字校验**：OCR 读出的每个 ≥6 位数字，其数字串必须能在该页文字层的
   数字串里找到（文字层里数字常被列宽切成碎片，按顺序拼起来才对得上，
   例如 ``819,8`` + ``71,17`` + ``0.19`` → ``81987117019``）；命中率不足
   九成就整页弃用，保留文字层原样；
3. **列数合理性**：OCR 切出的列数超过文字层 2 倍时判为切碎，弃用。

任何一道不过，就保持原样——宁可少补一页，也不能往库里塞错数。
"""

from __future__ import annotations

import re
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.ingest.parsing import TABLE, Segment

logger = get_logger(__name__)

_PLACEHOLDER = re.compile(r"\|\s*列\d+\s*\|")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# 校验只看足够长的数字：短数字（0、5.00、100）在原文里遍地都是，没有区分度
_MIN_CHECK_DIGITS = 6
_MIN_VERIFY_RATE = 0.9


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def _numbers(text: str, min_digits: int = _MIN_CHECK_DIGITS) -> list[str]:
    found = []
    for raw in _NUMBER.findall(text):
        digits = _digits(raw)
        if len(digits) >= min_digits:
            found.append(digits)
    return found


def _page_digit_stream(path: Path, page: int) -> str:
    """该页文字层的全部数字（按阅读顺序拼接，去掉分隔符）。"""
    import pymupdf

    try:
        doc = pymupdf.open(str(path))
    except Exception:
        return ""
    try:
        if page < 1 or page > doc.page_count:
            return ""
        return _digits(doc[page - 1].get_text("text"))
    except Exception:
        return ""
    finally:
        doc.close()


def numbers_verified(ocr_text: str, stream: str) -> bool:
    """OCR 的数字能不能在文字层数字串里找到。"""
    numbers = _numbers(ocr_text)
    if not numbers:
        return True  # 没读到大额数字，谈不上"编数字"
    hit = sum(1 for n in numbers if n in stream)
    return hit / len(numbers) >= _MIN_VERIFY_RATE


_TABLE_BLOCK = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)


def _table_blocks(text: str) -> list[str]:
    """OCR 结果里的 HTML 表格块（丢掉它顺手输出的正文）。"""
    return [match.strip() for match in _TABLE_BLOCK.findall(text or "") if match.strip()]


def _column_count(text: str) -> int:
    """表格段的最大列数（HTML 按 <td> 数，Markdown 按 | 数）。"""
    if "<td" in text or "<th" in text:
        widths = [row.count("<td") + row.count("<th") for row in text.split("<tr>")]
        return max(widths) if widths else 0
    widths = [line.count("|") - 1 for line in text.splitlines() if line.strip().startswith("|")]
    return max(widths) if widths else 0


# 整个单元格只由数字组成、且空格分隔出 ≥3 个数：列被压扁了。
# 在全库 12147 个表格切片上标定：这条命中 38 片（0.31%），抽查全是真故障；
# 放宽成「单格 ≥3 个数字」会命中 1920 片，因为 2026年9月30日 这种日期里
# 也带着三个数字，所以必须要求整格只有数字。
_PURE_NUMBERS = re.compile(r"^-?\d[\d,.]*(?:\s+-?\d[\d,.]*){2,}$")

# 表头出现同一个字符连续 4 次以上（`年年年年年年年年2025 …`）：find_tables 把
# 「年」「月」各当成一列的结果。样本少（全库 8 片），只作为辅助信号。
_REPEAT_CHARS = re.compile(r"(.)\1{3,}")


def _table_cells(text: str) -> list[list[str]]:
    """切成二维单元格；HTML 表先转 Markdown。"""
    from ai_agent.kb.ingest.chunking import html_tables_to_markdown

    raw = html_tables_to_markdown(text) if "<table" in (text or "").lower() else (text or "")
    grid: list[list[str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(re.fullmatch(r"-{2,}", cell or "-") for cell in cells):
            continue  # 分隔行
        grid.append(cells)
    return grid


def collapsed_cells(text: str) -> int:
    """「多个数值被塞进同一格」的单元格数。"""
    count = 0
    for cells in _table_cells(text):
        count += sum(1 for cell in cells if _PURE_NUMBERS.match(cell))
    return count


def has_repeated_header(text: str) -> bool:
    """表头是否出现连续重复字符（表头被按字切碎的特征）。"""
    for cells in _table_cells(text)[:4]:
        for cell in cells:
            if len(cell) <= 30 and _REPEAT_CHARS.search(cell):
                return True
    return False


def _candidates(segments: list[Segment], crashed_pages: set[int]) -> dict[int, list[int]]:
    """页 → 该页上「坏掉的表格段」下标。"""
    by_page: dict[int, list[int]] = {}
    for index, seg in enumerate(segments):
        page = seg.page or 0
        if page <= 0:
            continue
        text = seg.text or ""
        # 三类坏表：表头变占位「列2」、列被压扁（多个数值挤一格）、表头被切碎
        broken = seg.kind == TABLE and (
            _PLACEHOLDER.search(text)
            or collapsed_cells(text) >= 1
            or has_repeated_header(text)
        )
        if broken or page in crashed_pages:
            if seg.kind == TABLE or page in crashed_pages:
                by_page.setdefault(page, []).append(index)
    return by_page


def rescue_table_pages(
    path: Path,
    segments: list[Segment],
    crashed_pages: set[int] | None = None,
    max_pages: int = 400,
) -> tuple[list[Segment], dict[str, int]]:
    """对坏掉的表格页跑 OCR，校验通过就整页替换。返回新列表与统计。"""
    from ai_agent.kb.ingest import ocr

    stats = {"候选页": 0, "OCR补回": 0, "校验未过": 0, "无产出": 0}
    if not segments or not ocr.available():
        return segments, stats

    by_page = _candidates(segments, crashed_pages or set())
    if not by_page:
        return segments, stats
    if max_pages > 0 and len(by_page) > max_pages:
        logger.info(
            "表格页 OCR 兜底命中 %d 页，超过上限 %d，只处理前 %d 页",
            len(by_page),
            max_pages,
            max_pages,
        )
        by_page = dict(sorted(by_page.items())[:max_pages])
    stats["候选页"] = len(by_page)

    replaced: dict[int, Segment] = {}
    # 一次把所有候选页交给 ocr_pdf：它内部按 kb_ocr_concurrency 分批并发，
    # 逐页调用等于并发退化成 1（实测 30 页要 18 分钟）
    try:
        found_all = ocr.ocr_pdf(path, pages=sorted(by_page)) or []
    except Exception:
        logger.warning("表格页 OCR 失败，保留文字层", exc_info=True)
        found_all = []
    ocr_text_by_page: dict[int, str] = {}
    for seg in found_all:
        if seg.page and (seg.text or "").strip():
            ocr_text_by_page[seg.page] = ocr_text_by_page.get(seg.page, "") + "\n" + seg.text

    for page, indexes in sorted(by_page.items()):
        old_text = "\n".join(segments[i].text or "" for i in indexes)
        text = ocr_text_by_page.get(page, "").strip()
        if not text:
            stats["无产出"] += 1
            continue
        # 只取 OCR 结果里的表格：整页 OCR 同时会吐出这一页的正文，原样插进来
        # 会和文字层已有的正文重复一遍（实测汇川 p7 的段落、联系人表各出现两次）
        tables = _table_blocks(text)
        if not tables:
            stats["校验未过"] += 1
            continue
        payload = "\n\n".join(tables)
        if _column_count(payload) > max(_column_count(old_text), 1) * 2:
            logger.info("表格页 OCR 切得过碎 page=%d，保留文字层", page)
            stats["校验未过"] += 1
            continue
        if not numbers_verified(payload, _page_digit_stream(path, page)):
            logger.warning("表格页 OCR 数字对不上 page=%d，保留文字层", page)
            stats["校验未过"] += 1
            continue
        replaced[page] = Segment(payload, page, TABLE, segments[indexes[0]].section)
        stats["OCR补回"] += 1

    if not replaced:
        return segments, stats

    out: list[Segment] = []
    dropped: set[int] = set()
    for page in replaced:
        # 整页 OCR 会把页内每张表都读出来，所以该页**所有**表格段都要让位，
        # 只摘坏掉的那几段会让没坏的表重复出现
        for index, seg in enumerate(segments):
            if (seg.page or 0) == page and seg.kind == TABLE:
                dropped.add(index)
    for index, seg in enumerate(segments):
        if index in dropped:
            page = seg.page or 0
            if page in replaced:
                out.append(replaced.pop(page))
            continue
        out.append(seg)
    logger.info(
        "kb 表格页 OCR 兜底：候选 %d 页，补回 %d 页，校验未过 %d 页，无产出 %d 页",
        stats["候选页"],
        stats["OCR补回"],
        stats["校验未过"],
        stats["无产出"],
    )
    return out, stats


def enabled() -> bool:
    return bool(getattr(settings, "kb_table_rescue_enabled", True))