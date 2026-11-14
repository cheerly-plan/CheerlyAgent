"""表格切片：一个行项目一片，每片重复表头；「其中：」与合计跟随科目。

由原 chunking.py 拆分而来，逻辑未改动。
"""

import re

from ai_agent.kb.ingest.chunking.common import (
    _HTML_CELL,
    _HTML_TABLE,
    _MD_SEPARATOR,
    _NUM_ONLY_CELL,
    _cell_plain,
    _is_sub_row,
    _is_total_row,
)

from ai_agent.kb.ingest.chunking.html import (
    _html_header_row_count,
    _is_html_section_row,
    _split_html_table,
    _table_rows,
)

from ai_agent.kb.ingest.chunking.text import (
    chunk_text,
)


def extract_table_indicators(text: str) -> list[str]:
    """表格第一列的行项目名。单位、纯数字、表头和分组标题丢掉，不存数值。"""
    cells = _table_label_cells(text or "")
    names: list[str] = []
    seen: set[str] = set()
    for cell in cells:
        name = _clean_indicator(cell)
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= _INDICATOR_LIMIT:
            break
    return names

def _table_label_cells(text: str) -> list[str]:
    tables = list(_HTML_TABLE.finditer(text))
    if tables:
        cells: list[str] = []
        for match in tables:
            rows = _table_rows(match.group(0))
            head_n = _html_header_row_count(rows)
            for row in rows[head_n:]:
                if _is_html_section_row(row):
                    continue
                found = next(_HTML_CELL.finditer(row), None)
                if found:
                    cells.append(_cell_plain(found.group(2)))
        return cells
    lines = [line for line in text.splitlines() if line.strip()]
    while lines and "|" not in lines[0]:
        lines.pop(0)
    if not lines:
        return []
    body = [
        line
        for line in lines[1:]
        if not _MD_SEPARATOR.match(line.strip())
    ]
    return [_md_first_cell(line) for line in body]

def _clean_indicator(cell: str) -> str:
    name = (cell or "").strip().strip("*").strip()
    if not (2 <= len(name) <= 40):
        return ""
    if (
        _INDICATOR_NUM.match(name)
        or _INDICATOR_HEADER.match(name)
        or _INDICATOR_SECTION.match(name)
        or _INDICATOR_TOTAL.match(name)
    ):
        return ""
    return name

def _chunk_table(text: str, *, max_chars: int, group_mode: str = "row") -> list[str]:
    """表格切片：短表整片保留；多行表按行项目切，每片重复表头和表题。

    表格正文开头可能有若干不含竖线的行：表题（「母公司利润表」）和
    单位 / 审计状态。这些必须在每片里重复——否则后半张表只剩
    「项目 | 本期 | 上期 + 一堆数字」，看不出这张表属于谁、单位是什么。

    「其中：」子项、金融工具三个计量类别、合计 / 总计都跟所属科目走。

    ``group_mode``：``row`` 每个科目一片；``section`` 按 max_chars 在科目
    边界累积（与 HTML 表同一策略）。文字层产出的是 Markdown 表，以前漏接
    了 section，年报会切出上千片几乎等长的单行表。
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    # HTML 表没有竖线。不走下面的 Markdown 切法，否则表题会被单独算成一张表。
    # 多行科目即使整表不到 max_chars 也切开，避免一张利润表合成一个向量。
    html_parts = _chunk_html_table(text, max_chars=max_chars, group_mode=group_mode)
    if html_parts is not None:
        return html_parts
    # row 模式短表整片留；section 模式要进打包器，才能在「流动资产：」这类
    # 分组标题处强制起新片（与 HTML section 行为一致）。
    if group_mode != "section" and len(text) <= max_chars:
        return [text]

    caption: list[str] = []
    while lines and "|" not in lines[0]:
        caption.append(lines.pop(0))

    sep_index = next(
        (i for i, line in enumerate(lines[:5]) if _MD_SEPARATOR.match(line.strip())),
        None,
    )
    if sep_index is not None:
        header, data = _demote_fake_md_header(
            caption, lines[:sep_index], lines[sep_index], lines[sep_index + 1 :]
        )
    else:
        header, data = [*caption, *lines[:1]], lines[1:]
    if not data:
        # 不是表格形态（单行超长），退回普通切分
        return chunk_text(text, max_chars=max_chars, overlap=0)

    units = _table_row_units(data)
    if group_mode == "section":
        return _pack_table_units_section(header, units, max_chars=max_chars) or [text]
    return _pack_table_units(header, units) or [text]

def _md_row_cells(line: str) -> list[str]:
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    return [part.strip() for part in re.split(r"(?<!\\)\|", raw)]

def _demote_fake_md_header(
    caption: list[str],
    header_rows: list[str],
    separator: str,
    data: list[str],
) -> tuple[list[str], list[str]]:
    """科目行误当 Markdown 表头时降回数据行，换成中性列名再切。"""
    from ai_agent.kb.ingest.table_cells import _is_data_row, _is_item_row

    if not header_rows:
        return [*caption, separator], data
    cells = _md_row_cells(header_rows[-1])
    if not (_is_item_row(cells) or _is_data_row(cells)):
        return [*caption, *header_rows, separator], data
    width = max(1, len(cells))
    synth = "| " + " | ".join(["项目"] + [f"列{i}" for i in range(2, width + 1)]) + " |"
    return [*caption, synth, separator], [*header_rows, *data]

def _md_first_cell(line: str) -> str:
    """Markdown 表行的第一列（科目名）。"""
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    parts = re.split(r"(?<!\\)\|", raw)
    return parts[0].strip() if parts else ""

def _is_section_label(line: str) -> bool:
    """「流动资产：」这种只有科目、没有数字的分组标题。"""
    cell = _md_first_cell(line)
    if not cell.endswith(("：", ":")):
        return False
    return not any(
        _NUM_ONLY_CELL.match(part.strip())
        for part in re.split(r"(?<!\\)\|", line.strip().strip("|"))
        if part.strip()
    )

def _table_row_units(data: list[str]) -> list[list[str]]:
    """把数据行收成不可拆的行组：父行 + 其中/子项，合计贴到上一组。"""
    units: list[list[str]] = []
    for line in data:
        cell = _md_first_cell(line)
        if units and (_is_sub_row(cell) or _is_total_row(cell)):
            units[-1].append(line)
            continue
        units.append([line])
    return units

def _pack_table_units(header: list[str], units: list[list[str]]) -> list[str]:
    """每个行项目一片。表头和当前分组标题每片都带，合计绝不单独开片。"""
    out: list[str] = []
    buf: list[str] = list(header)
    section: list[str] = []

    def has_account() -> bool:
        return any(line not in header and not _is_section_label(line) for line in buf)

    def emit() -> None:
        if has_account():
            out.append("\n".join(buf))

    def restart() -> None:
        nonlocal buf
        buf = [*header, *section]

    for unit in units:
        if len(unit) == 1 and _is_section_label(unit[0]):
            if has_account():
                emit()
            section = list(unit)
            restart()
            continue
        if has_account():
            emit()
            restart()
        buf.extend(unit)
    emit()
    return _merge_orphan_total_chunks(header, out)

def _lines_len(lines: list[str]) -> int:
    return sum(len(line) + 1 for line in lines)

def _pack_table_units_section(
    header: list[str], units: list[list[str]], *, max_chars: int
) -> list[str]:
    """Markdown 表的 section 模式：按 max_chars 在科目边界累积。

    与 ``_split_html_table_grouped`` 对齐——分组标题（「流动资产：」）强制
    起新片；「其中：」/合计已收进 unit，不会被窗口切断。
    """
    out: list[str] = []
    section: list[str] = []
    buf: list[str] = list(header)
    cur_len = _lines_len(buf)

    def has_account() -> bool:
        return any(line not in header and not _is_section_label(line) for line in buf)

    def emit() -> None:
        if has_account():
            out.append("\n".join(buf))

    def restart() -> None:
        nonlocal buf, cur_len
        buf = [*header, *section]
        cur_len = _lines_len(buf)

    for unit in units:
        if len(unit) == 1 and _is_section_label(unit[0]):
            if has_account():
                emit()
            section = list(unit)
            restart()
            continue
        unit_len = _lines_len(unit)
        if has_account() and cur_len + unit_len > max_chars:
            emit()
            restart()
        buf.extend(unit)
        cur_len += unit_len
    emit()
    return _merge_orphan_total_chunks(header, out)

def _merge_orphan_total_chunks(header: list[str], chunks: list[str]) -> list[str]:
    """最后一片如果只剩合计/总计，并回上一片。"""
    if len(chunks) < 2:
        return chunks
    last_lines = chunks[-1].splitlines()
    body = last_lines[len(header) :]
    if body and all(_is_total_row(_md_first_cell(line)) for line in body):
        prev = chunks[-2].splitlines()
        return [*chunks[:-2], "\n".join(prev + body)]
    return chunks

def _chunk_html_table(
    text: str, *, max_chars: int = 500, group_mode: str = "row"
) -> list[str] | None:
    """表题加一张 HTML 表时按行切开，每片重复表题。不是这种形态就返回 None。"""
    matches = list(_HTML_TABLE.finditer(text))
    if len(matches) != 1 or text[matches[0].end() :].strip():
        return None
    match = matches[0]
    caption = text[: match.start()].strip()
    parts = _split_html_table(
        match.group(0), max_chars=max_chars, group_mode=group_mode
    )
    if not caption:
        return parts
    return [f"{caption}\n{part}" for part in parts]

# 第一列里不是行项目的字：表头、单位、纯数字、分组标题（「流动资产：」）
_INDICATOR_NUM = re.compile(r"^[\d.,，%％+\-－—()（）/\s]+$")

_INDICATOR_HEADER = re.compile(
    r"^(项目|科目|附注|注释|行次|序号|金额|本期金额|上期金额|期末余额|期初余额)$"
    r"|^(单位|币种)[:：]"
)

_INDICATOR_SECTION = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9]{1,12}[：:]$")

_INDICATOR_TOTAL = re.compile(r"^(合计|小计|总计)$")

_INDICATOR_LIMIT = 40
