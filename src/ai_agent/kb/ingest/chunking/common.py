"""共享叶子层：Chunk 数据类、HTML 表格正则、行项目判定与单元格转文本。

由原 chunking.py 拆分而来，逻辑未改动。
"""

import html
import re

from dataclasses import dataclass

from ai_agent.kb.ingest.parsing import TEXT


@dataclass
class Chunk:
    text: str
    page: int | None = None  # PDF 页码 / Excel sheet 序号；无页概念的文档为 None
    kind: str = TEXT  # text / table / figure
    page_end: int | None = None  # 同节内聚合导致的跨页切片：末页页码
    section: str | None = None  # 所属一级节，如「第十节 管理层讨论与分析」
    # 报表口径：consolidated / parent / None（非报表或无表题）
    scope: str | None = None
    # 入库后的切片序号。从库里读出来建图时必须带上：建图会先过滤掉低信息
    # 切片，过滤后的列表下标和库里的 seq 不再相等，用下标当 seq 写 mention
    # 会让"原文证据"整体错位（实测只有 5% 指对了片子）
    seq: int | None = None

_HTML_TABLE = re.compile(r"<table\b[^>]*>[\s\S]*?</table>", re.IGNORECASE)

_HTML_ROW = re.compile(r"<tr\b[^>]*>[\s\S]*?</tr>", re.IGNORECASE)

_HTML_CELL = re.compile(r"<t[dh]\b([^>]*)>([\s\S]*?)</t[dh]>", re.IGNORECASE)

_HTML_COLSPAN = re.compile(r"""colspan=["'](\d+)["']""", re.IGNORECASE)

_HTML_ROWSPAN = re.compile(r"""rowspan=["'](\d+)["']""", re.IGNORECASE)

_NUM_ONLY_CELL = re.compile(r"^[\d.,%+\-]+$")

# 「利润总额 | 不适用」「研发占比 | 减少5.75个百分点」是数据，不是列名。
# 当成列名会从这里把表切开，起始行上的 rowspan 还留在上一片，
# 后面的行被当成新表头，最后一行还会因为「没有科目」被丢掉。
_NOT_HEADER_CELL = re.compile(
    r"^不适用$|^(?:增加|减少)[\d.,]+个百分点$"
)

# 嵌套子行：紧跟上一行，不能单独切到下一片
_TABLE_SUB = re.compile(r"^(其中[:：]?|[（(]其中[）)]|[—–\-•·]\s*|[加减][:：])")

# 金融工具三个计量类别是同一组，即使没写「其中：」也不能拆
_TABLE_IFRS_SUB = re.compile(r"^以(公允价值|摊余成本)计量")

# 汇总行必须跟本片明细走，禁止单独成片
_TABLE_TOTAL = re.compile(r"(合计|小计|总计)$")

def _is_sub_row(cell: str) -> bool:
    return bool(_TABLE_SUB.match(cell) or _TABLE_IFRS_SUB.match(cell))

def _is_total_row(cell: str) -> bool:
    return bool(_TABLE_TOTAL.search(cell))

def _md_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2

def _md_cells(line: str) -> list[str]:
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    return [part.strip().replace("\\|", "|") for part in re.split(r"(?<!\\)\|", raw)]

def _md_cell(text: str) -> str:
    return (text or "").replace("|", "\\|")

# Markdown 表格的分隔行：|---|---|
_MD_SEPARATOR = re.compile(r"^\|[\s:\-|]+\|$")

def _markdown_table_plain(text: str) -> str:
    """人工修订存成 Markdown 表时，分隔行同样不进向量。"""
    if "|" not in text:
        return text
    lines = text.splitlines()
    out: list[str] = []
    changed = False
    index = 0
    while index < len(lines):
        nxt = lines[index + 1] if index + 1 < len(lines) else ""
        if _md_row(lines[index]) and _MD_SEPARATOR.match(nxt.strip()):
            changed = True
            while index < len(lines) and _md_row(lines[index]):
                if not _MD_SEPARATOR.match(lines[index].strip()):
                    joined = " | ".join(_md_cells(lines[index]))
                    if joined.strip():
                        out.append(joined)
                index += 1
            continue
        out.append(lines[index])
        index += 1
    return "\n".join(out).strip() if changed else text

def _cell_plain(raw: str) -> str:
    text = re.sub(r"<[^>]+>", "", raw or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()

def _row_plain_line(row: str) -> str:
    return " | ".join(_cell_plain(match.group(2)) for match in _HTML_CELL.finditer(row))
