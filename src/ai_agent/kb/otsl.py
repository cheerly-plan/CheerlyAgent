"""Xiaomi-OCR-0 整页后处理。

来自 https://github.com/xiaomi-research/Xiaomi-OCR-0 的 postprocess/document.py
与 postprocess/otsl.py（Apache-2.0）。整页结果先截掉尾部重复、再把公式
分隔符收成 Markdown，最后把 OTSL 收成带 rowspan/colspan 的 HTML 表。
单元格文字会转义，避免原文里的尖括号被当成标签。
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

ROW_BREAK = "<nl>"
CELL = "<fcel>"
EMPTY = "<ecel>"
LEFT = "<lcel>"
UP = "<ucel>"
BOTH = "<xcel>"

ALL_TAGS = (CELL, EMPTY, LEFT, UP, BOTH, ROW_BREAK)
TAG_RE = re.compile("(" + "|".join(re.escape(tag) for tag in ALL_TAGS) + ")")
OTSL_TAG_RE = re.compile(r"<(?:fcel|ecel|nl|lcel|ucel|xcel)>")

ORIGIN_TAGS = frozenset({CELL, EMPTY})
RIGHT_CONTINUATION = frozenset({LEFT, BOTH})
DOWN_CONTINUATION = frozenset({UP, BOTH})


@dataclass
class Cell:
    """A resolved cell placed on the rectangular grid."""

    text: str
    row: int
    column: int
    rowspan: int = 1
    colspan: int = 1


def _tag_of(chunk: str) -> str:
    match = TAG_RE.match(chunk)
    return match.group(0) if match else ""


def _split_chunks(line: str) -> list[str]:
    """Split a row into chunks, each holding one tag plus the text after it."""
    starts = [match.start() for match in TAG_RE.finditer(line)]
    ends = starts[1:] + [len(line)]
    return [line[start:end] for start, end in zip(starts, ends)]


def _required_width(chunks: list[str]) -> int:
    """Number of leading cells a row must keep to preserve its last content."""
    width = 0
    for position, chunk in enumerate(chunks, start=1):
        if _tag_of(chunk) == CELL:
            width = position
    return width


def _target_width(rows: list[list[str]]) -> int:
    """Pick one cell count every row is padded or truncated to."""
    lower = max(_required_width(row) for row in rows)
    upper = max(len(row) for row in rows)
    best_width = upper
    best_cost = None
    for width in range(lower, upper + 1):
        cost = sum(abs(len(row) - width) for row in rows)
        if best_cost is None or cost < best_cost:
            best_width = width
            best_cost = cost
    return best_width


def pad_rows(otsl: str) -> str:
    """Give every row the same cell count so the grid can be indexed."""
    text = otsl.strip()
    if ROW_BREAK not in text:
        return text + ROW_BREAK

    rows = []
    for line in text.split(ROW_BREAK):
        chunks = _split_chunks(line)
        if chunks:
            rows.append(chunks)
    if not rows:
        return ROW_BREAK

    width = _target_width(rows)
    padded = []
    for chunks in rows:
        fixed = chunks[:width]
        fixed += [EMPTY] * (width - len(fixed))
        padded.append("".join(fixed))
    return ROW_BREAK.join(padded) + ROW_BREAK


def parse_grid(otsl: str) -> tuple[list[list[str]], list[list[str]]]:
    """Return the padded table as parallel tag and text grids."""
    tag_rows: list[list[str]] = []
    text_rows: list[list[str]] = []

    for line in otsl.split(ROW_BREAK):
        chunks = _split_chunks(line)
        if not chunks:
            continue
        tags = []
        texts = []
        for chunk in chunks:
            tag = _tag_of(chunk)
            tags.append(tag)
            texts.append(chunk[len(tag) :] if tag == CELL else "")
        tag_rows.append(tags)
        text_rows.append(texts)

    width = max((len(row) for row in tag_rows), default=0)
    for tags, texts in zip(tag_rows, text_rows):
        tags += [EMPTY] * (width - len(tags))
        texts += [""] * (width - len(texts))
    return tag_rows, text_rows


def resolve_cells(
    tag_rows: list[list[str]], text_rows: list[list[str]]
) -> list[Cell]:
    """Turn origin tags into cells, measuring spans from the continuation tags."""
    height = len(tag_rows)
    cells: list[Cell] = []

    for row_index, tags in enumerate(tag_rows):
        for column, tag in enumerate(tags):
            if tag not in ORIGIN_TAGS:
                continue

            colspan = 1
            while (
                column + colspan < len(tags)
                and tags[column + colspan] in RIGHT_CONTINUATION
            ):
                colspan += 1

            rowspan = 1
            while (
                row_index + rowspan < height
                and column < len(tag_rows[row_index + rowspan])
                and tag_rows[row_index + rowspan][column] in DOWN_CONTINUATION
            ):
                rowspan += 1

            text = text_rows[row_index][column] if tag == CELL else ""
            cells.append(
                Cell(
                    text=text.strip(),
                    row=row_index,
                    column=column,
                    rowspan=rowspan,
                    colspan=colspan,
                )
            )
    return cells


def render_html(cells: list[Cell], num_rows: int, num_cols: int) -> str:
    """Render resolved cells as an HTML table element."""
    if not cells:
        return ""

    grid: list[list[Cell | None]] = [[None] * num_cols for _ in range(num_rows)]
    for cell in cells:
        for row in range(cell.row, min(cell.row + cell.rowspan, num_rows)):
            for column in range(
                cell.column, min(cell.column + cell.colspan, num_cols)
            ):
                if row >= 0 and column >= 0:
                    grid[row][column] = cell

    covered_until: dict[int, int] = {}
    parts = []
    for row in range(num_rows):
        parts.append("<tr>")
        column = 0
        while column < num_cols:
            if covered_until.get(column, 0) > row:
                column += 1
                continue

            cell = grid[row][column]
            if cell is None or cell.row != row or cell.column != column:
                parts.append("<td></td>")
                column += 1
                continue

            colspan = 1
            while (
                column + colspan < num_cols and grid[row][column + colspan] is cell
            ):
                colspan += 1

            rowspan = 1
            while row + rowspan < num_rows and all(
                grid[row + rowspan][offset] is cell
                for offset in range(column, column + colspan)
            ):
                rowspan += 1

            attributes = ""
            if rowspan > 1:
                attributes += f' rowspan="{rowspan}"'
            if colspan > 1:
                attributes += f' colspan="{colspan}"'
            parts.append(f"<td{attributes}>{html.escape(cell.text)}</td>")

            for offset in range(column, column + colspan):
                covered_until[offset] = max(
                    covered_until.get(offset, 0), row + rowspan
                )
            column += colspan
        parts.append("</tr>")
    return "<table>" + "".join(parts) + "</table>"


def convert_otsl_to_html(otsl_content: str) -> str:
    """Convert an OTSL table into an HTML table element."""
    if not isinstance(otsl_content, str):
        raise TypeError("otsl_content must be a string")

    tag_rows, text_rows = parse_grid(pad_rows(otsl_content))
    cells = resolve_cells(tag_rows, text_rows)
    num_rows = len(tag_rows)
    num_cols = max((len(row) for row in tag_rows), default=0)
    return render_html(cells, num_rows, num_cols)


def _split_markdown_blocks(markdown: str):
    parts = re.split(r"(\n\s*\n)", markdown)
    blocks = []
    for i in range(0, len(parts), 2):
        text = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        blocks.append([text, sep])
    return blocks


def postprocess_otsl_tables(markdown: str) -> str:
    """Convert OTSL blocks to HTML tables. A failed block keeps its original text."""
    if not markdown or not OTSL_TAG_RE.search(markdown):
        return markdown

    blocks = _split_markdown_blocks(markdown)
    in_fence = False
    converted = []
    for text, sep in blocks:
        stripped = text.strip()
        fence_count = stripped.count("```")
        should_skip = in_fence
        if fence_count % 2 == 1:
            in_fence = not in_fence
            should_skip = True

        if not should_skip and OTSL_TAG_RE.search(stripped):
            try:
                table = convert_otsl_to_html(stripped)
                if table:
                    text = table
            except Exception:
                pass
        converted.append(text + sep)
    return "".join(converted)


def postprocess_math_delimiters(markdown: str) -> str:
    """Convert LaTeX delimiters in whole-page markdown to markdown math delimiters."""
    if not markdown:
        return markdown
    markdown = markdown.replace("\\[\\[", "\\[").replace("\\]\\]", "\\]")
    markdown = re.sub(
        r"\$\$?\s*\\\[\s*(.*?)\s*\\\]\s*\$\$?",
        r" $$ \1 $$ ",
        markdown,
        flags=re.DOTALL,
    )
    markdown = re.sub(
        r"\$\s*\\\(\s*(.*?)\s*\\\)\s*\$",
        r" $ \1 $",
        markdown,
        flags=re.DOTALL,
    )
    markdown = re.sub(
        r"\\\[\s*(.*?)\s*\\\]",
        r" $$ \1 $$ ",
        markdown,
        flags=re.DOTALL,
    )
    markdown = re.sub(
        r"\\\(\s*(.*?)\s*\\\)",
        r" $ \1 $",
        markdown,
        flags=re.DOTALL,
    )
    return markdown


def _clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """截断尾部周期性重复，只对足够长的输出生效。"""
    n = len(text)
    if n < min_text_len:
        return text

    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue

        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1

        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len

        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len :]

    return text


def finalize_document(text: str) -> str:
    """整页 OCR 原文 → 可入库的 Markdown。无 OTSL 时只做公式和重复清理。"""
    if not text:
        return text
    cleaned = postprocess_math_delimiters(_clean_truncated_repeats(text))
    return postprocess_otsl_tables(cleaned)


def render_otsl(text: str) -> str:
    """读出时把残留的 OTSL 收成 HTML。没有标记则原样返回。"""
    return postprocess_otsl_tables(text)
