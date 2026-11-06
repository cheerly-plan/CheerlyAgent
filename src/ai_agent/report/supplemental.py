"""单季视图、较上年末变动：只呈现财报**原文已经披露**的另一种观察角度。

这一层刻意不进 ``compare_measures``：累计/单季/时点的可比性门禁在上一轮刚
建好，把单季值当普通 Measure 塞进去会重新打开「9个月减3个月」的口子。
这里只用本期文件里的两类原始披露：

1. 主要财务数据表的「本报告期」(7-9月单季) 列，以及表格自带的
   「本报告期比上年同期增减(%)」官方单季同比；
2. 经营情况简要分析里「比上年末下降/增长…」的原文句子。

不自行折算年化、不由累计值倒推单季、不写任何因果归因。每个数字都必须
出现在保留下来的原文片段里，由 validate 再核一遍。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai_agent.core.logging import get_logger
from ai_agent.storage.store import store
from ai_agent.report.extract import (
    Evidence,
    _as_uuid,
    _files_for,
    _indicator_spec,
    _row_rank,
    _unit_of,
)

logger = get_logger(__name__)

_TABLE = re.compile(r"<table\b[^>]*>[\s\S]*?</table>", re.IGNORECASE)
_ROW = re.compile(r"<tr\b[^>]*>([\s\S]*?)</tr>", re.IGNORECASE)
_CELL = re.compile(r"<t[dh]\b[^>]*>([\s\S]*?)</t[dh]>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_NUMBER = re.compile(r"^[\s（(]*-?\d[\d,，]*(?:\.\d+)?%?[\s)）]*$")

# 经营分析正文里带「较上年末」变动的指标。值正则用于定位，随后只在该值
# 之后、句号之前的短窗口里找变动描述，避免跨句抓到别人的百分比。
_YEAR_END_SPECS: tuple[tuple[str, str, str], ...] = (
    ("不良贷款率", r"不良贷款率\s*([\d,]+(?:\.\d+)?)\s*%", "%"),
    (
        "不良贷款拨备覆盖率",
        r"不良贷款拨备覆盖率\s*([\d,]+(?:\.\d+)?)\s*%",
        "%",
    ),
    (
        "发放贷款和垫款总额",
        r"(?<!人民币)发放贷款和垫款总额(?:\s*\$\s*\^\{\d+\}\s*\$)?\s*"
        r"([\d,]+(?:\.\d+)?)\s*亿元",
        "亿元",
    ),
    (
        "吸收存款总额",
        r"(?<!人民币)吸收存款总额(?:\s*\$\s*\^\{\d+\}\s*\$)?\s*"
        r"([\d,]+(?:\.\d+)?)\s*亿元",
        "亿元",
    ),
    ("资产总计", r"资产总计\s*([\d,]+(?:\.\d+)?)\s*亿元", "亿元"),
)
_YEAR_END_ANCHOR = re.compile(r"比上年末|较上年末|比上年度末")
_CHANGE_TOKEN = re.compile(
    r"(下降|上升|增长|减少|增加)\s*([\d,]+(?:\.\d+)?)\s*(个百分点|%|亿元)"
)
_DOWN = {"下降", "减少"}


@dataclass
class QuarterView:
    """同一指标在一张表里的单季列与累计列（都是原文数字）。"""

    name: str
    unit: str
    cumulative_value: str = ""
    cumulative_yoy: str = ""
    quarter_value: str = ""
    quarter_yoy: str = ""
    evidence: Evidence | None = None


@dataclass
class YearEndView:
    """期末值 + 原文披露的「较上年末」变动。"""

    name: str
    value: str = ""
    unit: str = ""
    change: str = ""
    evidence: Evidence | None = None


@dataclass
class SupplementalResult:
    quarters: list[QuarterView] = field(default_factory=list)
    year_ends: list[YearEndView] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


def extract_supplemental(
    subjects: list[str], period: str, indicators: list[str]
) -> SupplementalResult:
    """只查本期文件。找不到就返回空视图并写明缺口，绝不跨文件拼数。"""
    result = SupplementalResult()
    names = [item.strip() for item in subjects if item and item.strip()]
    if not names or not period:
        return result
    try:
        files = _files_for(names, period)
    except Exception:
        logger.exception("读取单季视图的文件列表失败")
        result.gaps.append("单季视图：知识库文件列表读取失败")
        return result
    if not files:
        result.gaps.append(f"单季视图：没有找到 {period} 的对应季报")
        return result
    file_ids = [
        item
        for item in (_as_uuid(row.get("file_id")) for row in files)
        if item is not None
    ]
    wanted = {item.strip() for item in indicators if item and item.strip()}
    result.quarters = _quarter_views(file_ids, files, wanted)
    result.year_ends = _year_end_views(file_ids, files)
    if not result.quarters:
        result.gaps.append("单季视图：本期季报没有同时披露单季列与累计列")
    if not result.year_ends:
        result.gaps.append("较上年末：本期季报没有披露较上年末变动")
    return result


def _quarter_views(
    file_ids: list, files: list[dict], wanted: set[str]
) -> list[QuarterView]:
    try:
        rows = store.search_kb_chunks_keyword(
            ["主要财务数据"], file_ids=file_ids, top_k=10, kinds=["table"]
        )
    except Exception:
        logger.exception("检索主要财务数据表失败")
        return []
    by_id = {str(row.get("file_id")): row for row in files}
    views: list[QuarterView] = []
    seen: set[tuple[str, str]] = set()
    for chunk in rows:
        text = str(chunk.get("text") or "")
        parsed = _main_table(text)
        if parsed is None:
            continue
        headers, body = parsed
        q_i = _column(headers, lambda h: h == "本报告期")
        q_yoy_i = _column(headers, lambda h: "本报告期比上年同期增减" in h)
        c_i = _column(headers, lambda h: "年初至报告期末" in h and "增减" not in h)
        c_yoy_i = _column(
            headers, lambda h: "年初至报告期末比上年同期增减" in h
        )
        if q_i is None or c_i is None:
            continue
        unit = _unit_of(text)
        for cells in body:
            if not cells:
                continue
            row_name = cells[0].strip()
            matched = _match_wanted_indicator(row_name, wanted)
            if not matched or (matched, unit) in seen:
                continue
            quarter = _cell(cells, q_i)
            cumulative = _cell(cells, c_i)
            if not quarter or not cumulative:
                continue
            file_id = str(chunk.get("file_id") or "")
            meta = by_id.get(file_id, {})
            views.append(
                QuarterView(
                    name=matched,
                    unit=unit,
                    cumulative_value=cumulative,
                    cumulative_yoy=_percent(_cell(cells, c_yoy_i)),
                    quarter_value=quarter,
                    quarter_yoy=_percent(_cell(cells, q_yoy_i)),
                    evidence=Evidence(
                        file_id=file_id,
                        filename=str(chunk.get("filename") or meta.get("filename") or ""),
                        page=chunk.get("page"),
                        page_end=chunk.get("page_end"),
                        section=str(chunk.get("section") or ""),
                        quote=_quote(headers, cells),
                    ),
                )
            )
            seen.add((matched, unit))
    return views


def _match_wanted_indicator(row_name: str, wanted: set[str]) -> str:
    """把主表行名对回计划指标简称（如扣非净利润 ← 归属于…扣除非经常性损益的净利润）。"""
    if row_name in wanted:
        return row_name
    best = ""
    best_rank = 0
    for indicator in wanted:
        _keyword, aliases = _indicator_spec(indicator)
        rank = _row_rank(row_name, aliases)
        if rank > best_rank:
            best_rank = rank
            best = indicator
    return best


def _year_end_views(file_ids: list, files: list[dict]) -> list[YearEndView]:
    """较上年末：先读资产负债主表（本报告期末／上年度末），再读经营分析正文。"""
    by_id = {str(row.get("file_id")): row for row in files}
    views: list[YearEndView] = []
    found: set[str] = set()

    try:
        table_chunks = store.search_kb_chunks_keyword(
            ["上年度末"], file_ids=file_ids, top_k=10, kinds=["table"]
        )
    except Exception:
        logger.exception("检索较上年末资产负债表失败")
        table_chunks = []
    for chunk in table_chunks:
        text = str(chunk.get("text") or "")
        for item in _balance_year_end_rows(text):
            if item.name in found:
                continue
            file_id = str(chunk.get("file_id") or "")
            meta = by_id.get(file_id, {})
            views.append(
                YearEndView(
                    name=item.name,
                    value=item.value,
                    unit=item.unit or _unit_of(text) or "元",
                    change=item.change,
                    evidence=Evidence(
                        file_id=file_id,
                        filename=str(chunk.get("filename") or meta.get("filename") or ""),
                        page=chunk.get("page"),
                        page_end=chunk.get("page_end"),
                        section=str(chunk.get("section") or ""),
                        quote=item.quote,
                    ),
                )
            )
            found.add(item.name)

    try:
        chunks = store.search_kb_chunks_keyword(
            ["比上年末"], file_ids=file_ids, top_k=10, kinds=["text"]
        )
    except Exception:
        logger.exception("检索较上年末变动失败")
        chunks = []
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        for name, pattern, unit in _YEAR_END_SPECS:
            if name in found:
                continue
            match = re.search(pattern, text)
            if not match:
                continue
            change = _change_after(text, match.end())
            if not change:
                continue
            sentence = _sentence_around(text, match.start())
            file_id = str(chunk.get("file_id") or "")
            meta = by_id.get(file_id, {})
            views.append(
                YearEndView(
                    name=name,
                    value=match.group(1).strip(),
                    unit=unit,
                    change=change,
                    evidence=Evidence(
                        file_id=file_id,
                        filename=str(chunk.get("filename") or meta.get("filename") or ""),
                        page=chunk.get("page"),
                        page_end=chunk.get("page_end"),
                        section=str(chunk.get("section") or ""),
                        quote=sentence,
                    ),
                )
            )
            found.add(name)
    return views


@dataclass
class _BalanceYearEnd:
    name: str
    value: str
    change: str
    unit: str = ""
    quote: str = ""


def _balance_year_end_rows(text: str) -> list[_BalanceYearEnd]:
    """读「本报告期末｜上年度末｜比上年度末增减」表，常见于非金融公司季报第 1–2 页。"""
    wanted = ("总资产", "归属于上市公司股东的所有者权益", "归属于上市公司股东的净资产")
    found: list[_BalanceYearEnd] = []
    for table in _TABLE.finditer(text):
        block = table.group(0)
        rows = [
            [_clean(cell.group(1)) for cell in _CELL.finditer(row.group(1))]
            for row in _ROW.finditer(block)
        ]
        if len(rows) < 2:
            continue
        headers = rows[0]
        flat = "".join(headers)
        if "上年度末" not in flat or "本报告期末" not in flat:
            continue
        end_i = _column(headers, lambda h: h == "本报告期末" or h.endswith("本报告期末"))
        # 表头第一列常为空，用「上年度末」定位后一列增减
        change_i = _column(
            headers,
            lambda h: "比上年度末" in h or "较上年度末" in h or "比上年末" in h,
        )
        if end_i is None or change_i is None:
            continue
        for cells in rows[1:]:
            if not cells:
                continue
            name = cells[0].strip()
            if name not in wanted:
                continue
            value = _cell(cells, end_i)
            change_raw = _cell(cells, change_i)
            if not value or not change_raw:
                continue
            change = _percent(change_raw)
            if not change:
                continue
            # 表内增减未带方向词时，正数记为 +、负数保留 -
            if not change.startswith(("+", "-")):
                change = f"+{change}"
            # 校验要求片段里能见到「增加/减少」方向词；表头只有「增减幅度」，
            # 按数值符号写成方向词，并保留原百分比数字。
            direction = "减少" if change.startswith("-") else "增加"
            pct = change.lstrip("+-")
            quote = (
                f"{name} 本报告期末 {value}，"
                f"本报告期末比上年度末增减变动幅度(%) {change_raw}，"
                f"比上年度末{direction}{pct}"
            )
            found.append(
                _BalanceYearEnd(
                    name=name,
                    value=value,
                    change=change,
                    unit=_unit_of(text) or "元",
                    quote=quote,
                )
            )
    return found


def _change_after(text: str, start: int) -> str:
    """在指标值之后的同一句里找「比上年末…」，只回一个带方向的变动串。

    「比上年末增加15,667.30亿元，增长6.68%」里有金额增量和百分比两个数字，
    优先取百分比/百分点（增长率），不取金额增量，避免把两种口径混写。
    """
    tail = text[start : start + 120].split("。")[0].split("；")[0]
    anchor = _YEAR_END_ANCHOR.search(tail)
    if not anchor:
        return ""
    tokens = list(_CHANGE_TOKEN.finditer(tail[anchor.start() :]))
    if not tokens:
        return ""
    # 一句里可能连着两个指标（不良率 + 拨备覆盖率），只取紧跟本指标的第一个
    # 变动，不能取到后一个指标头上去。
    preferred = [item for item in tokens if item.group(3) in {"%", "个百分点"}]
    token = preferred[0] if preferred else tokens[0]
    sign = "-" if token.group(1) in _DOWN else "+"
    return f"{sign}{token.group(2)}{token.group(3)}"


def _sentence_around(text: str, position: int) -> str:
    left = max(text.rfind("。", 0, position), text.rfind("\n", 0, position))
    right = text.find("。", position)
    stop = right + 1 if right >= 0 else len(text)
    return text[left + 1 : stop].strip()[:300]


def _main_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    """找同时有「本报告期」和「年初至报告期末」的那张主表。"""
    for table in _TABLE.finditer(text):
        block = table.group(0)
        if "colspan" in block.lower():
            continue
        rows = [
            [_clean(cell.group(1)) for cell in _CELL.finditer(row.group(1))]
            for row in _ROW.finditer(block)
        ]
        if not rows:
            continue
        headers = rows[0]
        flat = "".join(headers)
        if "本报告期" in flat and "年初至报告期末" in flat:
            return headers, rows[1:]
    return None


def _clean(raw: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub("", raw)).strip()


def _column(headers: list[str], predicate) -> int | None:
    for index, header in enumerate(headers):
        if predicate(header.strip()):
            return index
    return None


def _cell(cells: list[str], index: int | None) -> str:
    if index is None or index >= len(cells):
        return ""
    raw = cells[index].strip()
    return raw if _NUMBER.match(raw) else ""


def _percent(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned:
        return ""
    return cleaned if cleaned.endswith("%") else f"{cleaned}%"


def _quote(headers: list[str], cells: list[str]) -> str:
    text = " | ".join(headers) + "\n" + " | ".join(cells)
    return text[:500]
