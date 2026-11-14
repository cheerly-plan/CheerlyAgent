"""HTML 表格：行/列解析、rowspan 展开、与 Markdown 互转、超长表切分。

由原 chunking.py 拆分而来，逻辑未改动。
"""

import re

from ai_agent.kb.ingest.parsing import TABLE

from ai_agent.kb.ingest.chunking.common import (
    Chunk,
    _HTML_CELL,
    _HTML_COLSPAN,
    _HTML_ROW,
    _HTML_ROWSPAN,
    _HTML_TABLE,
    _NOT_HEADER_CELL,
    _NUM_ONLY_CELL,
    _cell_plain,
    _is_sub_row,
    _is_total_row,
    _markdown_table_plain,
    _md_cell,
    _row_plain_line,
)


def _html_chunk_kind(seg_kind: str, text: str) -> str:
    """整段就是一张 HTML 表时，切片类型记成 table。"""
    if _HTML_TABLE.fullmatch(text.strip()):
        return TABLE
    return seg_kind

def _table_rows(table: str) -> list[str]:
    return _HTML_ROW.findall(table)

def _row_width(row: str) -> int:
    width = 0
    for match in _HTML_CELL.finditer(row):
        span = _HTML_COLSPAN.search(match.group(1) or "")
        width += int(span.group(1)) if span else 1
    return width

def _row_texts(row: str) -> list[str]:
    return [
        re.sub(r"\s+", "", match.group(2))
        for match in _HTML_CELL.finditer(row)
    ]

def _html_row_is_data(row: str) -> bool:
    """至少一个格子是纯数字。列名行里的「2026年1-9月」不算。"""
    return any(cell and _NUM_ONLY_CELL.match(cell) for cell in _row_texts(row))

def _html_row_is_item(row: str) -> bool:
    """科目行：第一列是名称，其余列是空的或数字。

    资产负债表跨页时，页首常常是「其他债权投资」这种金额整格为空的行，
    它没有数字，但仍然是上一张表的续表，不是新的列名。
    """
    cells = _row_texts(row)
    if not cells or not cells[0] or _html_row_is_header(row):
        return False
    return all(not cell or _NUM_ONLY_CELL.match(cell) for cell in cells[1:])

def _html_row_is_header(row: str) -> bool:
    if _html_row_is_clipped_header(row):
        return False
    cells = [cell for cell in _row_texts(row) if cell]
    if len(cells) < 2 or _html_row_is_data(row):
        return False
    if any(_NOT_HEADER_CELL.match(cell) for cell in cells):
        return False
    return any(re.search(r"[\u4e00-\u9fff]", cell) for cell in cells)

def _html_row_is_clipped_header(row: str) -> bool:
    """跨页被页眉切掉的表头：第一列以及紧挨着的格子都空，剩下「(%)」「份数量」。

    新的一行第一列不会空。只空左上角、第二格已经是「期末余额」的，是正常表头。
    """
    texts = _row_texts(row)
    if len(texts) < 2 or texts[0] or texts[1] or _html_row_is_data(row):
        return False
    return any(texts)

def _rowspan_continuations(rows: list[str]) -> set[int]:
    """被上面某一行的 rowspan 盖住的行号。这些行必须和起始行留在同一片。"""
    covered: set[int] = set()
    for index, row in enumerate(rows):
        for match in _HTML_CELL.finditer(row):
            span = _span_attr(_HTML_ROWSPAN, match.group(1) or "")
            for down in range(1, span):
                covered.add(index + down)
    return covered

def _clamp_rowspans(rows: list[str]) -> list[str]:
    """rowspan 不能超出这一片实际剩下的行，否则展示会补出空行。"""
    total = len(rows)
    clamped: list[str] = []
    for index, row in enumerate(rows):
        room = total - index

        def repl(match: re.Match[str], room: int = room) -> str:
            attrs = match.group(1) or ""
            span = _span_attr(_HTML_ROWSPAN, attrs)
            if span <= room:
                return match.group(0)
            if room <= 1:
                attrs = _HTML_ROWSPAN.sub("", attrs)
            else:
                attrs = _HTML_ROWSPAN.sub(f' rowspan="{room}"', attrs)
            open_tag = re.match(r"<\s*(t[dh])", match.group(0), re.IGNORECASE)
            name = open_tag.group(1) if open_tag else "td"
            return f"<{name}{attrs}>{match.group(2)}</{name}>"

        clamped.append(_HTML_CELL.sub(repl, row))
    return clamped

def _html_tables_from_groups(groups: list[list[str]]) -> list[str]:
    return [
        "<table>" + "".join(_clamp_rowspans(group)) + "</table>" for group in groups
    ]

def _html_header_to_carry(rows: list[str]) -> list[str] | None:
    """续表要继承的列名。

    页中途换过列名（「合并资产负债表项目」下面又印「合并利润表项目」）时，
    带过去的是最后一组列名，不是表头第一行。开头连续的列名行一起带上。
    """
    last: int | None = None
    first_data: int | None = None
    for index, row in enumerate(rows):
        if _html_row_is_header(row):
            last = index
        elif first_data is None and _html_row_is_data(row):
            first_data = index
    if last is None:
        return None
    if first_data is None or last < first_data:
        end = 0
        while end < len(rows) and _html_row_is_header(rows[end]):
            end += 1
        return rows[:end] or None
    return [rows[last]]

def _can_merge_html_tables(prev: Chunk, nxt: Chunk) -> bool:
    if prev.kind != TABLE or nxt.kind != TABLE:
        return False
    if prev.page_end is None or nxt.page is None or nxt.page != prev.page_end + 1:
        return False
    rows_a = _table_rows(prev.text)
    rows_b = _table_rows(nxt.text)
    if not rows_a or not rows_b:
        return False
    width = _row_width(rows_a[0])
    return width > 0 and width == _row_width(rows_b[0])

def _merge_html_tables(first: str, second: str) -> str:
    """接上续页的数据行。续页若把表头又印了一遍，这几行丢掉。"""
    rows_a = _table_rows(first)
    rows_b = _table_rows(second)
    copied = 0
    limit = min(3, len(rows_a), len(rows_b))
    while copied < limit:
        left = _row_texts(rows_a[copied])
        if not any(left) or left != _row_texts(rows_b[copied]):
            break
        copied += 1
    return "<table>" + "".join(rows_a + rows_b[copied:]) + "</table>"

def _html_pieces(para: str) -> list[tuple[str, str]]:
    """把一段里的 HTML 表抽出来。表不参与后面的 500 字聚合。"""
    if not _HTML_TABLE.search(para):
        return [("text", para)]
    pieces: list[tuple[str, str]] = []
    last = 0
    for match in _HTML_TABLE.finditer(para):
        before = para[last : match.start()].strip()
        if before:
            pieces.append(("text", before))
        pieces.append(("table", match.group(0)))
        last = match.end()
    after = para[last:].strip()
    if after:
        pieces.append(("text", after))
    return pieces or [("text", para)]

def html_table_plain(text: str) -> str:
    """表格换成「单元格 | 单元格」的行。标签和 Markdown 分隔行不进向量，也不进重排。"""
    if not text:
        return text
    if _HTML_TABLE.search(text):

        def repl(match: re.Match[str]) -> str:
            lines = [_row_plain_line(row) for row in _table_rows(match.group(0))]
            return "\n".join(line for line in lines if line.strip())

        text = _HTML_TABLE.sub(repl, text).strip()
    return _markdown_table_plain(text)

def html_tables_to_markdown(text: str) -> str:
    """HTML 表换成 Markdown 表。只用于喂给模型的正文，库存和引用仍是 HTML。

    colspan / rowspan 展开成矩形，合并格的文字写进它盖住的每一列，
    「期末余额」横跨「合并 | 母公司」时两列都还带着这个列名。
    整行只有一个分组标题（「流动资产：」）时不横着复制，避免把它当成每列的值。
    """
    if not text or not _HTML_TABLE.search(text):
        return text
    return _HTML_TABLE.sub(lambda match: _one_html_table_to_markdown(match.group(0)), text)

def _one_html_table_to_markdown(table: str) -> str:
    grid = _expand_html_grid(_table_rows(table))
    if not grid or not any(grid):
        plain = html_table_plain(table)
        return plain or table
    width = max(len(row) for row in grid)
    lines = [
        "| " + " | ".join(_md_cell(cell) for cell in row + [""] * (width - len(row))) + " |"
        for row in grid
    ]
    separator = "| " + " | ".join("---" for _ in range(width)) + " |"
    return "\n".join([lines[0], separator, *lines[1:]])

def _expand_html_grid(rows: list[str]) -> list[list[str]]:
    """按 colspan / rowspan 把表铺成矩形。被上一行占住的格子跳过。"""
    grid: list[list[str | None]] = []
    for row_index, row_html in enumerate(rows):
        while len(grid) <= row_index:
            grid.append([])
        column = 0
        # 「流动资产：」这种分组行横跨整表。复制到每一列，模型会把它当成该列的数值。
        section_row = _is_html_section_row(row_html)
        for match in _HTML_CELL.finditer(row_html):
            attrs = match.group(1) or ""
            colspan = _span_attr(_HTML_COLSPAN, attrs)
            rowspan = _span_attr(_HTML_ROWSPAN, attrs)
            while column < len(grid[row_index]) and grid[row_index][column] is not None:
                column += 1
            text = _cell_plain(match.group(2))
            section = section_row and colspan > 1
            for down in range(rowspan):
                # 切片把合并格切断后，rowspan 可能比这一片剩下的行多。
                # 只铺已经存在的行，不能为了凑 rowspan 补出空行。
                if row_index + down >= len(rows):
                    break
                while len(grid) <= row_index + down:
                    grid.append([])
                target = grid[row_index + down]
                while len(target) < column + colspan:
                    target.append(None)
                for offset in range(colspan):
                    if target[column + offset] is None:
                        target[column + offset] = "" if section and offset else text
            column += colspan
    width = max((len(row) for row in grid), default=0)
    return [[cell or "" for cell in row] + [""] * (width - len(row)) for row in grid]

def _span_attr(pattern: re.Pattern[str], attrs: str) -> int:
    found = pattern.search(attrs)
    if not found:
        return 1
    try:
        return max(int(found.group(1)), 1)
    except ValueError:
        return 1

def _html_header_row_count(rows: list[str]) -> int:
    """表头占几行：第一行的 rowspan 盖住的都算表头，每片都要重复。"""
    if not rows:
        return 0
    span = 1
    for match in _HTML_CELL.finditer(rows[0]):
        found = _HTML_ROWSPAN.search(match.group(1) or "")
        if found:
            span = max(span, int(found.group(1)))
    return min(max(span, 1), len(rows))

def _is_html_section_row(row: str) -> bool:
    """「流动资产：」这种分组标题，跟在它后面的科目走，不留在上一组末尾。"""
    cells = [_cell_plain(match.group(2)) for match in _HTML_CELL.finditer(row)]
    filled = [cell for cell in cells if cell]
    return len(filled) == 1 and filled[0].endswith(("：", ":"))

def _buf_has_account(buf: list[str], head_n: int) -> bool:
    """表头和分组标题（「流动资产：」）之外，已经有一行科目。"""
    return any(not _is_html_section_row(row) for row in buf[head_n:])

def _split_html_table(
    table: str, *, max_chars: int = 500, group_mode: str = "row"
) -> list[str]:
    """HTML 表按行项目切开。每片重复表头和当前分组标题。

    「其中：」、加减行、合计跟所属科目留在同一片。单行再长也不从格子中间切。
    被 rowspan 盖住的续行跟起始行留在同一片，避免合并格的文字留在上一片、
    下面的项目名被当成新表头。

    group_mode="row"（默认）：每个行项目单独成片，自解释最强但切片数多。
    group_mode="section"：语义感知滑动窗口（见 _split_html_table_grouped），
    按 max_chars 累积、在科目行边界断、分组标题强制起新片，切片数大幅减少。
    """
    rows = _table_rows(table)
    if not rows:
        return [table]
    clipped = 0
    while clipped < len(rows) and _html_row_is_clipped_header(rows[clipped]):
        clipped += 1
    if clipped:
        rows = rows[clipped:]
        if not rows:
            return [table]
    head_n = _html_header_row_count(rows)
    # 残头后面直接是股东行时，不能把第一位股东再当成列名复制到每一片。
    if (
        clipped
        and _html_row_is_data(rows[0])
        and not _html_row_is_header(rows[0])
    ):
        head_n = 0
    header, data = rows[:head_n], rows[head_n:]
    if len(data) <= 1:
        return _html_tables_from_groups([rows]) if rows else [table]
    covered = _rowspan_continuations(rows)
    if group_mode == "section":
        return _split_html_table_grouped(header, data, max_chars, covered, head_n)
    groups: list[list[str]] = []
    section: list[str] = []
    buf = list(header)

    def restart(extra: list[str]) -> None:
        nonlocal buf
        if _buf_has_account(buf, len(header)):
            groups.append(buf)
        buf = [*header, *extra]

    for offset, row in enumerate(data):
        if head_n + offset in covered:
            buf.append(row)
            continue
        first = _row_plain_line(row).split(" | ", 1)[0].strip()
        # 同一张表中途换列名（资产负债表下面又印利润表）时，后面的科目跟新列名。
        if _html_row_is_header(row):
            if _buf_has_account(buf, len(header)):
                groups.append(buf)
                header = [row]
            elif buf == header:
                header = [*header, row]
            else:
                header = [row]
            section = []
            buf = list(header)
            continue
        if _is_html_section_row(row):
            section = [row]
            restart(section)
            continue
        sticky = _is_sub_row(first) or _is_total_row(first)
        if not sticky and _buf_has_account(buf, len(header)):
            restart(section)
        buf.append(row)
    if _buf_has_account(buf, len(header)):
        groups.append(buf)
    if not groups:
        return [table]
    return _html_tables_from_groups(groups)

def _rows_len(rows: list[str]) -> int:
    """一组表格行的字符长度（用向量化的单元格文字计，与 max_chars 同一量级）。"""
    return sum(len(_row_plain_line(row)) + 1 for row in rows)

def _split_html_table_grouped(
    header: list[str],
    data: list[str],
    max_chars: int,
    covered: set[int] | None = None,
    head_n: int = 0,
) -> list[str]:
    """section 模式：语义感知滑动窗口，替代固定行数切分。

    按 max_chars 累积行，到窗口临界时在下一个科目行边界（非「其中：」/合计、
    非分组标题）断开——绝不切断一个科目及其子项。分组标题（「流动资产：」）
    强制起新片，让同组科目聚在一片；「其中：」子项、合计/小计跟所属科目走。
    每片重复表头 + 当前分组标题，离开表题/单位也能自解释。无分组标题的扁平
    表同样适用：只是在科目边界按窗口断，不会无限聚成一片。
    """
    groups: list[list[str]] = []
    section: list[str] = []
    buf = list(header)
    cur_len = _rows_len(header)

    def flush() -> None:
        nonlocal buf, cur_len
        if _buf_has_account(buf, len(header)):
            groups.append(buf)
        buf = [*header, *section]
        cur_len = _rows_len(buf)

    for offset, row in enumerate(data):
        if head_n + offset in (covered or ()):
            buf.append(row)
            cur_len += len(_row_plain_line(row)) + 1
            continue
        first = _row_plain_line(row).split(" | ", 1)[0].strip()
        if _html_row_is_header(row):
            # 中途换列名：保留原分支语义
            if _buf_has_account(buf, len(header)):
                groups.append(buf)
                header = [row]
            elif buf == header:
                header = [*header, row]
            else:
                header = [row]
            section = []
            buf = list(header)
            cur_len = _rows_len(buf)
            continue
        if _is_html_section_row(row):
            # 分组标题强制起新片，同组科目聚在一片
            flush()
            section = [row]
            buf = [*header, *section]
            cur_len = _rows_len(buf)
            continue
        sticky = _is_sub_row(first) or _is_total_row(first)
        row_len = len(_row_plain_line(row)) + 1
        if (
            not sticky
            and _buf_has_account(buf, len(header))
            and cur_len + row_len > max_chars
        ):
            flush()
        buf.append(row)
        cur_len += row_len
    if _buf_has_account(buf, len(header)):
        groups.append(buf)
    if not groups:
        return _html_tables_from_groups([header + data])
    return _html_tables_from_groups(groups)
