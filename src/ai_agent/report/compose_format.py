"""compose Markdown / HTML 排版。"""

from __future__ import annotations

import html
import re

def markdown(
    title: str, sections: list[tuple[str, list[str]]], note: tuple[str, ...] = ()
) -> str:
    """段落之间空一行，但表格的连续行必须挨着，否则 Markdown 表格会散架。

    顶层 join 会把每个元素之间插入空行，所以表格行先合并成一块再交给它。
    HTML 路径（`html_block`）本来就是按连续 "|" 行拼 <table>，不受影响。
    """
    parts = [f"# {title}"]
    if note:
        parts.append("\n".join(f"> {line}" for line in note))
    for heading, lines in sections:
        parts.append(f"## {heading}")
        table: list[str] = []
        for line in lines:
            if line.startswith("|"):
                table.append(line)
                continue
            if table:
                parts.append("\n".join(table))
                table.clear()
            parts.append(line)
        if table:
            parts.append("\n".join(table))
    return "\n\n".join(parts).strip() + "\n"


def html_doc(
    title: str, sections: list[tuple[str, list[str]]], note: tuple[str, ...] = ()
) -> str:
    body = [f"<h1>{html.escape(title)}</h1>"]
    if note:
        inner = "".join(f"<p>{html.escape(line)}</p>" for line in note)
        body.append(f"<blockquote>{inner}</blockquote>")
    for heading, lines in sections:
        body.append("<section>")
        body.append(f"<h2>{html.escape(heading)}</h2>")
        body.append(html_block(lines))
        body.append("</section>")
    inner = "\n".join(body)
    return (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        f"<title>{html.escape(title)}</title>\n</head>\n<body>\n{inner}\n</body>\n</html>\n"
    )


# 千分位长数字：包进 .num 禁止 PDF 中途断行。
LONG_NUMBER = re.compile(r"[+\-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?")
NUMERIC_CELL = re.compile(r"^[+\-]?[\d,]+(?:\.\d+)?(?:%|元|次|亿元|万元|百万元)?$")


def html_block(lines: list[str]) -> str:
    chunks: list[str] = []
    buffer: list[str] = []

    def flush_table() -> None:
        if not buffer:
            return
        rows = [line for line in buffer if not is_rule(line)]
        head, *rest = rows
        rendered = ["<table>", "<thead><tr>", *th(cells(head)), "</tr></thead>", "<tbody>"]
        for row in rest:
            rendered.append("<tr>")
            rendered.extend(td(cells(row)))
            rendered.append("</tr>")
        rendered.append("</tbody></table>")
        chunks.append("\n".join(rendered))
        buffer.clear()

    for line in lines:
        if line.startswith("|"):
            buffer.append(line)
            continue
        flush_table()
        chunks.append(f"<p>{wrap_long_numbers(html.escape(line))}</p>")
    flush_table()
    return "\n".join(chunks)


def is_rule(line: str) -> bool:
    return set(line.replace("|", "").replace(" ", "")) <= {"-", ":"}


def cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def th(cells: list[str]) -> list[str]:
    return [f"<th>{html.escape(cell)}</th>" for cell in cells]


def td(cells: list[str]) -> list[str]:
    out: list[str] = []
    for cell in cells:
        escaped = html.escape(cell)
        if NUMERIC_CELL.match(cell.replace(" ", "")):
            out.append(f'<td><span class="num">{escaped}</span></td>')
        else:
            out.append(f"<td>{wrap_long_numbers(escaped)}</td>")
    return out


def wrap_long_numbers(escaped: str) -> str:
    """已 HTML escape 的文本里，给千分位数字加 nowrap 包裹。"""
    return LONG_NUMBER.sub(lambda match: f'<span class="num">{match.group(0)}</span>', escaped)
