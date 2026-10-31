"""把 Xiaomi-OCR-0 的 OTSL 表格标记收成 Markdown 表格。

模型按训练输出 `<fcel>` `<ecel>` `<nl>` 这类格子标记。原文里的数字还在，
但标签把一行拆碎之后，对话模型和重排模型对不齐列。这里在送给模型、
以及切片预览之前换成项目里统一的 Markdown 表格；库里的原文不动，
已经入库的文件不用重新解析。
"""

from __future__ import annotations

import re

_TAG_RE = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|nl)>")
# 同一行里、第一个标签前面的短片段，多半是漏了开头 <fcel> 的单元格
_LEADING_CELL = re.compile(r"^[^。！？!?\n]{1,40}$")
_CONTINUATION = frozenset({"lcel", "ucel", "xcel"})


def render_otsl(text: str) -> str:
    """没有 OTSL 标记时原样返回。转换失败也退回原文，避免把正文弄丢。"""
    if not text or not _TAG_RE.search(text):
        return text
    try:
        rendered = _render_blocks(text)
    except Exception:
        return text
    return rendered or text


def _render_blocks(text: str) -> str:
    parts = re.split(r"(\n\s*\n)", text)
    out: list[str] = []
    for i in range(0, len(parts), 2):
        block = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        out.append(_render_block(block) if _TAG_RE.search(block) else block)
        out.append(sep)
    return "".join(out)


def _render_block(block: str) -> str:
    first = _TAG_RE.search(block)
    if first is None:
        return block
    last = None
    for last in _TAG_RE.finditer(block):
        pass
    assert last is not None
    line_start = block.rfind("\n", 0, first.start()) + 1
    inline = block[line_start:first.start()]
    if _LEADING_CELL.match(inline.strip()):
        prose_before = block[:line_start]
        table_src = "<fcel>" + inline + block[first.start():last.end()]
    else:
        prose_before = block[:first.start()]
        table_src = block[first.start():last.end()]
    prose_after = block[last.end():]
    rows = _grid(table_src)
    if not rows:
        return block
    from ai_agent.kb.parsing import _escape_cell, _rows_to_markdown

    escaped = [[_escape_cell(cell) for cell in row] for row in rows]
    table = _rows_to_markdown(escaped)
    pieces = [part.strip("\n") for part in (prose_before, table, prose_after)]
    return "\n\n".join(part for part in pieces if part.strip())


def _grid(src: str) -> list[list[str]]:
    """按 `<nl>` 分行，带文字的标记都算一格；无文字的跨列标记留空以对齐。"""
    tokens: list[tuple[str, str]] = []
    matches = list(_TAG_RE.finditer(src))
    for i, match in enumerate(matches):
        nxt = matches[i + 1].start() if i + 1 < len(matches) else len(src)
        tokens.append((match.group(1), src[match.end():nxt].strip()))

    raw_rows: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    for tag, cell in tokens:
        if tag == "nl":
            if current:
                raw_rows.append(current)
                current = []
            continue
        if cell or tag not in _CONTINUATION:
            tag = "fcel" if cell else tag
        current.append((tag, cell if tag == "fcel" else ""))
    if current:
        raw_rows.append(current)
    if not raw_rows:
        return []

    width = max(len(row) for row in raw_rows)
    grid: list[list[str]] = []
    for row in raw_rows:
        cells = [cell if tag == "fcel" else "" for tag, cell in row]
        cells += [""] * (width - len(cells))
        grid.append(cells[:width])
    return grid
