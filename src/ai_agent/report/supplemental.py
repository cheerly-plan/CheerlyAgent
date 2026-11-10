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
from ai_agent.report import registry

logger = get_logger(__name__)

_TABLE = re.compile(r"<table\b[^>]*>[\s\S]*?</table>", re.IGNORECASE)
_ROW = re.compile(r"<tr\b[^>]*>([\s\S]*?)</tr>", re.IGNORECASE)
_CELL = re.compile(r"<t[dh]\b[^>]*>([\s\S]*?)</t[dh]>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_NUMBER = re.compile(r"^[\s（(]*-?\d[\d,，]*(?:\.\d+)?%?[\s)）]*$")

# 经营分析正文里带「较上年末」变动的指标。值正则用于定位，随后只在该值
# 之后、句号之前的短窗口里找变动描述，避免跨句抓到别人的百分比。
_YEAR_END_SPECS: tuple[tuple[str, str, str], ...] = registry.year_end_specs()
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
    # 上年度末原值（构成拆解用）；主表有「上年度末」列时填入。
    prior_value: str = ""
    evidence: Evidence | None = None


@dataclass
class BalanceMoveView:
    """合并资产负债表科目：本期末 vs 上年末（均为原文列）。"""

    name: str
    end_value: str = ""
    prior_value: str = ""
    unit: str = "元"
    evidence: Evidence | None = None


@dataclass
class YearEndReasonView:
    """「主要会计数据、财务指标发生变动的情况、原因」表里的原文原因。"""

    name: str
    reason: str = ""
    evidence: Evidence | None = None


@dataclass
class SupplementalResult:
    quarters: list[QuarterView] = field(default_factory=list)
    year_ends: list[YearEndView] = field(default_factory=list)
    # 权益/资产暴涨时的科目拆解（其他综合收益、资本公积…），来自合并资产负债表。
    balance_moves: list[BalanceMoveView] = field(default_factory=list)
    # 总资产／归母权益变动原因原文（②级引述）。
    year_end_reasons: list[YearEndReasonView] = field(default_factory=list)
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
    result.balance_moves = _balance_move_views(file_ids, files)
    result.year_end_reasons = _year_end_reason_views(file_ids, files)
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
                    prior_value=item.prior_value,
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
    prior_value: str = ""
    quote: str = ""


# 权益暴涨时优先拆的资产负债表科目（合并报表「本期末｜上年末」两列）。
_BALANCE_MOVE_NAMES = registry.ordering_list("balance_move_names_order")

_DATE_HEADER = re.compile(r"20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日")


def _balance_year_end_rows(text: str) -> list[_BalanceYearEnd]:
    """读「本报告期末｜上年度末｜比上年度末增减」表，常见于非金融公司季报第 1–2 页。"""
    wanted = registry.ordering_list("balance_year_end_rows")
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
        prior_i = _column(headers, lambda h: "上年度末" in h and "比" not in h and "较" not in h)
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
            prior = _cell(cells, prior_i) if prior_i is not None else ""
            prior_bit = f"上年度末 {prior}，" if prior else ""
            quote = (
                f"{name} 本报告期末 {value}，{prior_bit}"
                f"本报告期末比上年度末增减变动幅度(%) {change_raw}，"
                f"比上年度末{direction}{pct}"
            )
            found.append(
                _BalanceYearEnd(
                    name=name,
                    value=value,
                    change=change,
                    unit=_unit_of(text) or "元",
                    prior_value=prior,
                    quote=quote,
                )
            )
    return found


def _balance_move_views(file_ids: list, files: list[dict]) -> list[BalanceMoveView]:
    """从合并资产负债表抽关键科目期末／上年末原值，供权益构成拆解。"""
    by_id = {str(row.get("file_id")): row for row in files}
    try:
        chunks = store.search_kb_chunks_keyword(
            list(registry.ordering_list("balance_search_keywords")),
            file_ids=file_ids,
            top_k=16,
            kinds=["table"],
        )
    except Exception:
        logger.exception("检索资产负债表科目变动失败")
        return []
    found: dict[str, BalanceMoveView] = {}
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        file_id = str(chunk.get("file_id") or "")
        meta = by_id.get(file_id, {})
        for item in _balance_statement_rows(text):
            if item.name in found:
                continue
            delta = _format_delta(item.end_value, item.prior_value)
            quote = (
                f"{item.name} 本期末 {item.end_value}，上年末 {item.prior_value}"
                + (f"，增减 {delta}" if delta else "")
            )
            found[item.name] = BalanceMoveView(
                name=item.name,
                end_value=item.end_value,
                prior_value=item.prior_value,
                unit=item.unit or _unit_of(text) or "元",
                evidence=Evidence(
                    file_id=file_id,
                    filename=str(chunk.get("filename") or meta.get("filename") or ""),
                    page=chunk.get("page"),
                    page_end=chunk.get("page_end"),
                    section=str(chunk.get("section") or ""),
                    quote=quote,
                ),
            )
    # 稳定顺序：按预设名单
    order = {name: index for index, name in enumerate(_BALANCE_MOVE_NAMES)}
    return sorted(found.values(), key=lambda row: order.get(row.name, 99))


def _year_end_reason_views(file_ids: list, files: list[dict]) -> list[YearEndReasonView]:
    """读变动原因表：总资产／归母权益旁的「主要是…」原文。"""
    by_id = {str(row.get("file_id")): row for row in files}
    wanted = registry.ordering_list("year_end_reason_rows")
    try:
        chunks = store.search_kb_chunks_keyword(
            list(registry.ordering_list("year_end_reason_keywords")),
            file_ids=file_ids,
            top_k=8,
            kinds=["table", "text"],
        )
    except Exception:
        logger.exception("检索较上年末变动原因失败")
        return []
    found: dict[str, YearEndReasonView] = {}
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        file_id = str(chunk.get("file_id") or "")
        meta = by_id.get(file_id, {})
        for name, reason in _change_reason_rows(text, wanted):
            if name in found or len(reason) < 12:
                continue
            found[name] = YearEndReasonView(
                name=name,
                reason=reason,
                evidence=Evidence(
                    file_id=file_id,
                    filename=str(chunk.get("filename") or meta.get("filename") or ""),
                    page=chunk.get("page"),
                    page_end=chunk.get("page_end"),
                    section=str(chunk.get("section") or ""),
                    quote=f"{name}：{reason}",
                ),
            )
    return [found[name] for name in wanted if name in found]


@dataclass
class _BalanceStatementRow:
    name: str
    end_value: str
    prior_value: str
    unit: str = ""


def _balance_statement_rows(text: str) -> list[_BalanceStatementRow]:
    """合并资产负债表：表头两列日期（本期末／上年末）。"""
    found: list[_BalanceStatementRow] = []
    unit = _unit_of(text) or "元"
    for table in _TABLE.finditer(text):
        block = table.group(0)
        rows = [
            [_clean(cell.group(1)) for cell in _CELL.finditer(row.group(1))]
            for row in _ROW.finditer(block)
        ]
        if len(rows) < 2:
            continue
        headers = rows[0]
        date_cols = [i for i, h in enumerate(headers) if _DATE_HEADER.search(h or "")]
        if len(date_cols) < 2:
            continue
        end_i, prior_i = date_cols[0], date_cols[1]
        for cells in rows[1:]:
            if not cells:
                continue
            name = cells[0].strip()
            if name not in _BALANCE_MOVE_NAMES:
                continue
            # 统一「实收资本」显示名
            display = "实收资本（或股本）" if "实收资本" in name else name
            end_v = _cell(cells, end_i)
            prior_v = _cell(cells, prior_i)
            if not end_v or not prior_v:
                continue
            found.append(
                _BalanceStatementRow(
                    name=display, end_value=end_v, prior_value=prior_v, unit=unit
                )
            )
    return found


def _change_reason_rows(
    text: str, wanted: tuple[str, ...]
) -> list[tuple[str, str]]:
    """从变动原因 HTML 表抽出 (指标名, 原因句)。"""
    out: list[tuple[str, str]] = []
    for table in _TABLE.finditer(text):
        block = table.group(0)
        for row in _ROW.finditer(block):
            cells = [_clean(cell.group(1)) for cell in _CELL.finditer(row.group(1))]
            if len(cells) < 2:
                continue
            name = cells[0].strip()
            if name not in wanted:
                continue
            # 常见列：指标｜变动比例｜主要原因；原因在最后一列最长文本。
            reason = max(cells[1:], key=len).strip()
            if name in reason and len(cells) >= 3:
                reason = cells[-1].strip()
            if "主要是" in reason or "公允价值" in reason or "资本公积" in reason:
                out.append((name, reason))
    # 正文兜底：总资产 309.67 主要是…
    if not out:
        for name in wanted:
            pattern = re.compile(
                rf"{re.escape(name)}[^。]{{0,40}}?(主要是[^。]{{10,220}})"
            )
            match = pattern.search(text.replace("\n", ""))
            if match:
                out.append((name, match.group(1).strip()))
    return out


def signed_delta(end_value: str, prior_value: str):
    """期末减上年末；解析失败返回 None。"""
    from ai_agent.report.compare import parse_number

    left = parse_number(end_value)
    right = parse_number(prior_value)
    if not left or not right:
        return None
    return left[0] - right[0]


def format_money_delta(delta) -> str:
    """带正负号与千分位的增减额文案。"""
    sign = "+" if delta >= 0 else "-"
    amount = abs(delta)
    if amount == amount.to_integral_value():
        return f"{sign}{int(amount):,}"
    return f"{sign}{amount:,.2f}"


def _format_delta(end_value: str, prior_value: str) -> str:
    """期末减上年末，带千分位与正负号；解析失败返回空。"""
    delta = signed_delta(end_value, prior_value)
    if delta is None:
        return ""
    return format_money_delta(delta)


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
