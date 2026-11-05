"""检索并抽取财报指标。

数字只从表格格子里读出来，并且必须原样出现在切片正文里。对不上就不返回，
避免模型改写或口算之后当成事实。每条指标带文件、页码和那一行原文。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from ai_agent.core.logging import get_logger
from ai_agent.kb.chunking import (
    _HTML_CELL,
    _HTML_COLSPAN,
    _cell_plain,
    _html_header_row_count,
    _table_rows,
)
from ai_agent.kb.meta import name_in_text
from ai_agent.kb.tables import SCOPE_CONSOLIDATED, SCOPE_PARENT
from ai_agent.memory.store import store

logger = get_logger(__name__)

# 大纲里的简称落到报表行项目。检索用更长的子串，避免「扣非」对不上正式科目。
_INDICATORS: dict[str, tuple[str, tuple[str, ...]]] = {
    "营收": ("营业收入", ("营业收入",)),
    "营业收入": ("营业收入", ("营业收入",)),
    "扣非": ("扣除非经常性损益", ("扣除非经常性损益的净利润", "扣除非经常性损益后的净利润")),
    "扣非净利润": (
        "扣除非经常性损益",
        ("扣除非经常性损益的净利润", "扣除非经常性损益后的净利润"),
    ),
    "归母净利润": ("归属于上市公司股东的净利润", ("归属于上市公司股东的净利润",)),
    "净利润": ("净利润", ("净利润",)),
    "毛利率": ("毛利率", ("毛利率",)),
}

_LEADING = re.compile(
    r"^(?:其中[:：]?|[加减][:：]?|[-－—]|(?:[一二三四五六七八九十]|\d+)[、.．])\s*"
)
_TRAILING_PAREN = re.compile(r"[（(][^）)]*[）)]\s*$")
_PRIOR_COL = re.compile(r"上年|上期|期初|去年|同比|增减|变动|比上年|比上期")
_CURRENT_COL = re.compile(r"本报告期|本期|期末|本年|本期末")
_NUMBER = re.compile(r"^[（(]?\s*-?[\d,，]+(?:\.\d+)?\s*%?\s*[)）]?$")
_UNIT = re.compile(r"单位[:：]\s*([^\s，,。；;|<]+)")
_MD_SEPARATOR = re.compile(r"^\|[\s:\-|]+\|$")


@dataclass
class Evidence:
    file_id: str
    filename: str
    page: int | None
    page_end: int | None
    section: str
    quote: str


@dataclass
class IndicatorFact:
    subject: str
    period: str
    name: str
    value: str
    unit: str
    column: str
    scope: str
    evidence: Evidence


@dataclass
class ExtractResult:
    facts: list[IndicatorFact] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def extract_indicators(
    subjects: list[str],
    period: str,
    indicators: list[str],
) -> ExtractResult:
    """在指定公司、指定报告期的季报里抽取指标。

    subjects 是公司名或简称，period 形如 2025Q3。一家都没圈到、或表格里没有
    这一行时，写进 missing，不补一个看起来像的数字。
    """
    result = ExtractResult()
    names = [item.strip() for item in subjects if item and item.strip()]
    period = (period or "").strip()
    wanted = [item.strip() for item in indicators if item and item.strip()]
    if not names or not period or not wanted:
        result.missing.append("需要公司、报告期和指标")
        return result

    try:
        files = _files_for(names, period)
    except Exception:
        logger.exception("抽取指标时读取文件失败")
        result.missing.append("知识库文件列表读取失败")
        return result
    if not files:
        result.missing.append(f"没有找到 {period} 的对应季报")
        return result

    file_ids = [_as_uuid(row.get("file_id")) for row in files]
    file_ids = [item for item in file_ids if item is not None]
    by_id = {str(row.get("file_id")): row for row in files}

    for indicator in wanted:
        keyword, aliases = _indicator_spec(indicator)
        try:
            rows = store.search_kb_chunks_keyword(
                [keyword],
                file_ids=file_ids,
                top_k=20,
                kinds=["table"],
            )
        except Exception:
            logger.exception("抽取指标时检索失败 indicator=%s", indicator)
            result.missing.append(f"{indicator}：检索失败")
            continue
        picked = _pick_facts(rows, by_id, aliases, indicator, period)
        if not picked:
            result.missing.append(f"{indicator}：表格里没有对上的一行")
            continue
        result.facts.extend(picked)
    return result


def read_indicator(
    text: str,
    aliases: tuple[str, ...] | list[str],
    *,
    period: str = "",
) -> dict | None:
    """从一张表里读出指标的本期数字。读不到或数字不在原文里就返回 None。"""
    tables = _tables(text or "")
    best: dict | None = None
    best_rank = 0
    for headers, body in tables:
        for cells in body:
            if not cells:
                continue
            rank = _row_rank(cells[0], aliases)
            if rank <= best_rank:
                continue
            column, value = _current_value(headers, cells, period)
            if not value or not _value_in_source(value, text):
                continue
            best_rank = rank
            best = {
                "value": value,
                "column": column,
                "unit": _unit_of(text),
                "quote": _quote(headers, cells, text),
                "rank": rank,
            }
    return best


def _indicator_spec(name: str) -> tuple[str, tuple[str, ...]]:
    key = _label(name)
    if key in _INDICATORS:
        return _INDICATORS[key]
    return key, (key,)


def _files_for(subjects: list[str], period: str) -> list[dict]:
    rows = store.list_kb_files()
    chosen: list[dict] = []
    for row in rows:
        if row.get("status") != "ready":
            continue
        if (row.get("period") or "") != period:
            continue
        blob = " ".join(
            [
                str(row.get("subject") or ""),
                " ".join(str(item) for item in (row.get("subject_aliases") or [])),
                str(row.get("filename") or ""),
            ]
        )
        if any(name_in_text(blob, subject) for subject in subjects):
            chosen.append(row)
    return chosen


def _pick_facts(
    rows: list[dict],
    files: dict[str, dict],
    aliases: tuple[str, ...],
    indicator: str,
    period: str,
) -> list[IndicatorFact]:
    """同一份文件只留一条。科目本行优先于明细，合并口径优先于母公司。"""
    found: dict[str, tuple[tuple[int, int], IndicatorFact]] = {}
    for row in rows:
        parsed = read_indicator(str(row.get("text") or ""), aliases, period=period)
        if not parsed:
            continue
        file_id = str(row.get("file_id") or "")
        meta = files.get(file_id, {})
        scope = str(row.get("scope") or "")
        scope_rank = 2 if scope == SCOPE_CONSOLIDATED else 1 if scope != SCOPE_PARENT else 0
        # 科目本行优先于「其中」明细，合并口径优先于母公司口径
        rank = (parsed["rank"], scope_rank)
        fact = IndicatorFact(
            subject=str(meta.get("subject") or row.get("subject") or ""),
            period=period,
            name=indicator,
            value=parsed["value"],
            unit=parsed["unit"],
            column=parsed["column"],
            scope=scope,
            evidence=Evidence(
                file_id=file_id,
                filename=str(row.get("filename") or meta.get("filename") or ""),
                page=row.get("page"),
                page_end=row.get("page_end"),
                section=str(row.get("section") or ""),
                quote=parsed["quote"],
            ),
        )
        previous = found.get(file_id)
        if previous is None or rank > previous[0]:
            found[file_id] = (rank, fact)
    return [item[1] for item in found.values()]


def _tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    found = list(re.finditer(r"<table\b[^>]*>[\s\S]*?</table>", text, re.IGNORECASE))
    if found:
        tables: list[tuple[list[str], list[list[str]]]] = []
        for match in found:
            rows = [_expand_cells(row) for row in _table_rows(match.group(0))]
            if not rows:
                continue
            head_n = _html_header_row_count(_table_rows(match.group(0)))
            headers = _header_labels(rows[:head_n])
            tables.append((headers, rows[head_n:]))
        return tables
    return _markdown_tables(text)


def _expand_cells(row: str) -> list[str]:
    cells: list[str] = []
    for match in _HTML_CELL.finditer(row):
        span = _HTML_COLSPAN.search(match.group(1) or "")
        width = int(span.group(1)) if span else 1
        text = _cell_plain(match.group(2))
        cells.extend([text] * max(width, 1))
    return cells


def _header_labels(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    labels: list[str] = []
    for index in range(width):
        parts = [row[index] for row in rows if index < len(row) and row[index]]
        labels.append(" ".join(parts))
    return labels


def _markdown_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    lines = text.splitlines()
    tables: list[tuple[list[str], list[list[str]]]] = []
    index = 0
    while index < len(lines):
        nxt = lines[index + 1] if index + 1 < len(lines) else ""
        if "|" in lines[index] and _MD_SEPARATOR.match(nxt.strip()):
            body: list[list[str]] = []
            header = _split_md(lines[index])
            index += 2
            while index < len(lines) and "|" in lines[index]:
                body.append(_split_md(lines[index]))
                index += 1
            tables.append((header, body))
            continue
        index += 1
    return tables


def _split_md(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _label(name: str) -> str:
    text = re.sub(r"\s+", "", name or "")
    changed = True
    while changed:
        changed = False
        nxt = _TRAILING_PAREN.sub("", text).strip()
        if nxt != text:
            text = nxt
            changed = True
        nxt = _LEADING.sub("", text).strip()
        if nxt != text:
            text = nxt
            changed = True
    return text


def _row_rank(raw: str, aliases: tuple[str, ...] | list[str]) -> int:
    """科目本行高于「其中」明细。明细里的数字不能当成这个科目本身。"""
    base = _label_rank(_label(raw), aliases)
    if base == 0:
        return 0
    if (raw or "").strip().startswith("其中"):
        return base
    return base + 2


def _label_rank(label: str, aliases: tuple[str, ...] | list[str]) -> int:
    if not label:
        return 0
    rank = 0
    for alias in aliases:
        target = _label(alias)
        if not target:
            continue
        if label == target:
            rank = max(rank, 2)
        elif len(target) >= 8 and label.endswith(target):
            rank = max(rank, 1)
    return rank


def _current_value(
    headers: list[str], cells: list[str], period: str
) -> tuple[str, str]:
    year = period[:4] if len(period) >= 4 else ""
    current: tuple[str, str] | None = None
    unknown: tuple[str, str] | None = None
    for index, cell in enumerate(cells):
        if index == 0 or not _NUMBER.match(cell.strip()):
            continue
        header = headers[index] if index < len(headers) else ""
        if _PRIOR_COL.search(header):
            continue
        if (year and year in header) or _CURRENT_COL.search(header):
            current = (header, cell.strip())
            break
        if unknown is None:
            unknown = (header, cell.strip())
    chosen = current or unknown
    if not chosen:
        return "", ""
    return chosen


def _value_in_source(value: str, source: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    if not compact:
        return False
    return compact in re.sub(r"\s+", "", source)


def _unit_of(text: str) -> str:
    found = _UNIT.search(text or "")
    return found.group(1) if found else ""


def _quote(headers: list[str], cells: list[str], text: str) -> str:
    lines: list[str] = []
    unit = _unit_of(text)
    if unit:
        lines.append(f"单位：{unit}")
    if any(headers):
        lines.append(" | ".join(headers))
    lines.append(" | ".join(cells))
    quote = "\n".join(lines).strip()
    return quote if len(quote) <= 500 else quote[:500]


def _as_uuid(value: object) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None
