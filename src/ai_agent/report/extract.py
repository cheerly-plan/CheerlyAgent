"""检索并抽取财报指标。

数字只从表格格子里读出来，并且必须原样出现在切片正文里。对不上就不返回，
避免模型改写或口算之后当成事实。每条指标带文件、页码和那一行原文。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from ai_agent.core.logging import get_logger
from ai_agent.kb.ingest.chunking import (
    _HTML_CELL,
    _HTML_COLSPAN,
    _cell_plain,
    _html_header_row_count,
    _table_rows,
)
from ai_agent.kb.ingest.meta import (
    extract_subject,
    is_plausible_subject,
    name_in_text,
)
from ai_agent.kb.ingest.tables import SCOPE_CONSOLIDATED, SCOPE_PARENT
from ai_agent.report.basis import basis_of, ratio_unit
from ai_agent.report.meta import (
    enrich_quote_for_conversion,
    infer_scope,
    normalize_amount,
)
from ai_agent.storage.store import store

logger = get_logger(__name__)

# 证据引文噪音：PDF 断字、勾选框、误入的 Markdown 章节符。
_QUOTE_COMMA_BREAK = re.compile(r"(\d),(\s+)(\d)")
_QUOTE_PCT_BREAK = re.compile(r"(\d)\.\s*%")
_QUOTE_CHECKBOX = re.compile(r"[□√☑☒]\s*(?:适用|不适用)")
_QUOTE_MD_HEADING = re.compile(
    r"#{1,6}\s*(?:（[一二三四五六七八九十]+）|[^\n#]{1,40})"
)

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
    "毛利率": ("毛利率", ("毛利率", "综合毛利率")),
    "研发费用率": ("研发费用率", ("研发费用率", "研发投入占比")),
    "经营活动产生的现金流量净额": (
        "经营活动产生的现金流量净额",
        ("经营活动产生的现金流量净额",),
    ),
    "合同负债": ("合同负债", ("合同负债",)),
    "存货": ("存货", ("存货",)),
    "在手订单": ("在手订单", ("在手订单", "新签订单", "订单金额")),
}

# 银行季报的资产质量、息差、资本充足率写在「经营情况简要分析」正文里，
# 不是表格行。这里用**逐指标的显式短语**抽取，要求数字紧跟指标名、单位明确，
# 并保留原文句子作为证据；匹配不上就写缺口，不用通用正则去猜。
# 值： (检索关键词, 正则, 单位)。金额类只在正文给出「亿元」。
_NARRATIVE_PATTERNS: dict[str, tuple[str, str, str]] = {
    # 半导体等制造业季报常把毛利率写在经营分析正文，主表未必有独立行。
    "毛利率": ("毛利率", r"毛利率为?\s*([\d,]+(?:\.\d+)?)\s*%", "%"),
    "研发费用率": (
        "研发",
        r"(?:研发费用率|研发投入占营业收入(?:的比例)?|研发费用占营业收入(?:的比例)?)"
        r"为?\s*([\d,]+(?:\.\d+)?)\s*%",
        "%",
    ),
    "存货周转率": (
        "存货周转",
        r"存货周转率(?:（次）)?为?\s*([\d,]+(?:\.\d+)?)",
        "次",
    ),
    "资产负债率": ("资产负债率", r"资产负债率为?\s*([\d,]+(?:\.\d+)?)\s*%", "%"),
    "产能利用率": (
        "产能利用",
        r"(?:产能利用率|稼动率)为?\s*([\d,]+(?:\.\d+)?)\s*%",
        "%",
    ),
    "大客户集中度": (
        "客户",
        r"(?:前五名客户合计(?:销售)?占比|客户集中度|向前五名客户销售(?:额)?占(?:年度)?销售(?:总额)?的比例)"
        r"为?\s*([\d,]+(?:\.\d+)?)\s*%",
        "%",
    ),
    "合同负债": ("合同负债", r"合同负债\s*([\d,]+(?:\.\d+)?)\s*亿元", "亿元"),
    "在手订单": (
        "在手订单",
        r"(?:在手订单|新签订单)(?:金额|总额|合计)?(?:约为|为|达)?\s*"
        r"([\d,]+(?:\.\d+)?)\s*亿元",
        "亿元",
    ),
    "净息差": ("净息差", r"净息差\s*([\d,]+(?:\.\d+)?)\s*%", "%"),
    "不良贷款率": ("不良贷款率", r"不良贷款率\s*([\d,]+(?:\.\d+)?)\s*%", "%"),
    "不良贷款拨备覆盖率": (
        "拨备覆盖率",
        r"不良贷款拨备覆盖率\s*([\d,]+(?:\.\d+)?)\s*%",
        "%",
    ),
    "加权平均净资产收益率": (
        "加权平均净资产收益率",
        r"加权平均净资产收益率（ROE）\s*([\d,]+(?:\.\d+)?)\s*%",
        "%",
    ),
    "核心一级资本充足率": (
        "核心一级资本充足率",
        r"核心一级资本充足率为?\s*([\d,]+(?:\.\d+)?)\s*%",
        "%",
    ),
    # 前面加负向断言，别把「一级资本充足率」「核心一级资本充足率」当成总资本充足率
    "资本充足率": (
        "资本充足率",
        r"(?<!一级)资本充足率为?\s*([\d,]+(?:\.\d+)?)\s*%",
        "%",
    ),
    "利息净收入": ("利息净收入", r"利息净收入\s*([\d,]+(?:\.\d+)?)\s*亿元", "亿元"),
    "非利息收入": ("非利息收入", r"非利息收入\s*([\d,]+(?:\.\d+)?)\s*亿元", "亿元"),
    "成本收入比": ("成本收入比", r"成本收入比\s*([\d,]+(?:\.\d+)?)\s*%", "%"),
    "平均总资产回报率": (
        "平均总资产回报率",
        r"平均总资产回报率（ROA）\s*([\d,]+(?:\.\d+)?)\s*%",
        "%",
    ),
    "非利息收入占比": (
        "在营业收入中占比",
        r"非利息收入在营业收入中占比为\s*([\d,]+(?:\.\d+)?)\s*%",
        "%",
    ),
    "手续费及佣金净收入": (
        "手续费及佣金净收入",
        r"手续费及佣金净收入\s*([\d,]+(?:\.\d+)?)\s*亿元",
        "亿元",
    ),
    "业务及管理费": (
        "业务及管理费",
        r"业务及管理费\s*([\d,]+(?:\.\d+)?)\s*亿元",
        "亿元",
    ),
    # 正文「集团资产减值损失2958.40亿元」里的前导 2 是上标脚注（见原文脚注
    # 「2 资产减值损失包括…」），真值是 958.40。只剥「一位脚注 + 三位整数」：
    # 2958.40→958.40；干净的 958.40 / 带逗号的 2,958.40 不动。
    "资产减值损失": (
        "资产减值损失",
        r"资产减值损失\s*(?:\$\s*\^\{\d+\}\s*\$|(?:[2-9])(?=\d{3}(?:\.\d+)?\s*亿))?\s*"
        r"([\d,]+(?:\.\d+)?)\s*亿元",
        "亿元",
    ),
    # 原文脚注「$ ^{3} $」或 PDF 抽成「总额3250,202.22」：前导 3 是脚注，
    # 真值 250,202.22。只剥「一位脚注 + 三位,三位」；干净的 250,202.22 不动。
    "发放贷款和垫款总额": (
        "发放贷款和垫款总额",
        r"(?<!人民币)发放贷款和垫款总额"
        r"(?:\s*\$\s*\^\{\d+\}\s*\$|(?:[2-9])(?=\d{3},\d{3}))?\s*"
        r"([\d,]+(?:\.\d+)?)\s*亿元",
        "亿元",
    ),
    "吸收存款总额": (
        "吸收存款总额",
        r"(?<!人民币)吸收存款总额"
        r"(?:\s*\$\s*\^\{\d+\}\s*\$|(?:[2-9])(?=\d{3},\d{3}))?\s*"
        r"([\d,]+(?:\.\d+)?)\s*亿元",
        "亿元",
    ),
}
# 利润表里「利息净收入」同时有 7-9 月单季列和 1-9 月累计列，表格抽取容易
# 选中单季列（110,976 百万元）当成累计值。正文「2026年前三季度…利息净收入
# 3,596.06亿元」口径明确，这类金额指标一律正文优先，表格只作兜底。
# 资产减值损失同理：正文有集团合计，利润表拆成信用减值/其他两行，正文优先。
_NARRATIVE_FIRST = {
    "利息净收入",
    "非利息收入",
    "手续费及佣金净收入",
    "业务及管理费",
    "资产减值损失",
    "发放贷款和垫款总额",
    "吸收存款总额",
}

_LEADING = re.compile(
    r"^(?:其中[:：]?|[加减][:：]?|[-－—]|(?:[一二三四五六七八九十]|\d+)[、.．])\s*"
)
_TRAILING_PAREN = re.compile(r"[（(][^）)]*[）)]\s*$")
_PRIOR_COL = re.compile(r"上年|上期|期初|去年|同比|增减|变动|比上年|比上期")
_CURRENT_COL = re.compile(r"本报告期|本期|期末|本年|本期末")
# 季报「主要财务数据」常见双列：本报告期(单季) 与 年初至报告期末(累计)。
# 研报里说的「前三季度营收」指累计列，默认取它，不要把单季数当累计值报出去。
_CUMULATIVE_COL = re.compile(
    r"年初至报告期末|年初至本期末|年初至今|年初至|累计|前三季度|上半年|半年度"
    r"|1[-—至]\d{1,2}月"
)
_CHANGE_COL = re.compile(r"比上年同期增减|同比增减|同比|增减|变动")
# 千分位可能是逗号、空格或不换行空格（半年报摘要里常见「356 903」）。
# 只允许「数字 + 空白 + 恰好三位」这种分组，避免把「1 2 3」当成一个数。
_NUMBER = re.compile(
    r"^[（(]?\s*-?\d[\d,，]*(?:[ \t\u00a0]\d{3})*(?:\.\d+)?\s*%?\s*[)）]?$"
)
# 单位说明常带括号尾巴：「百万元人民币（另有说明者除外）」。
# 只保留可换算的单位词，别把整句注记写进研报正文。
_UNIT = re.compile(r"单位[:：]\s*([^\s，,。；;|<（(]+)")
_UNIT_KEY = re.compile(r"^(亿元|百万元|万元|元)")
_MD_SEPARATOR = re.compile(r"^\|[\s:\-|]+\|$")


@dataclass
class Evidence:
    file_id: str
    filename: str
    page: int | None
    page_end: int | None
    section: str
    quote: str


def citation_anchor(file_id: str) -> str:
    """正文引用的短锚点，形如 ``[7b75379b]``。

    完整 file_id 只留在溯源附件和 JSON 里：正文每条结论都跟一串 36 字符
    UUID 会把数字淹没。短锚点是全 ID 的前 8 位前缀，仍可回查溯源表；
    带方括号是为了让校验器在扫数字前能整体剔除，否则 ``75379`` 这类片段
    会被当成一个没授权过的金额。
    """
    short = (file_id or "").strip()[:8]
    return f"[{short}]" if short else ""


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
    # 季报原文自带的「比上年同期增减(%)」。它是披露值，不是我们算出来的，
    # 只在缺少跨期绝对值、无法自行重算同比时作为兜底写进正文。
    reported_yoy: str = ""
    # 数值取自累计列（年初至报告期末 / 1-6月）。compare 靠它判断能否
    # 用相邻累计值相减还原单季，避免把单季数当累计去减。
    cumulative: bool = False
    # 口径指纹，见 report/basis.py。空串表示调用方手工构造的 fact 没标口径，
    # compare 会回退到 cumulative 布尔值，保持旧调用兼容。
    basis: str = ""
    coverage_months: int = 0
    # 范围是否按季报模板推定（主表默认合并）；True 时正文写「按季报模板推定」。
    scope_inferred: bool = False
    # 金额折算前的原单位；空串表示未折算或非金额。
    raw_unit: str = ""
    unit_converted: bool = False


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
        picked: list[IndicatorFact] = []
        if indicator not in _NARRATIVE_FIRST:
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
            picked = _pick_facts(
                rows, by_id, aliases, indicator, period, requested=names
            )
        if not picked and indicator in _NARRATIVE_PATTERNS:
            picked = _narrative_facts(
                indicator, period, file_ids, by_id, requested=names
            )
        if not picked and indicator in _NARRATIVE_FIRST:
            # 正文也没写，才回到表格（并接受它的列口径）
            try:
                rows = store.search_kb_chunks_keyword(
                    [keyword], file_ids=file_ids, top_k=20, kinds=["table"]
                )
            except Exception:
                logger.exception("兜底表格抽取失败 indicator=%s", indicator)
                rows = []
            picked = _pick_facts(
                rows, by_id, aliases, indicator, period, requested=names
            )
        if not picked:
            result.missing.append(f"{indicator}：表格和正文里都没有对上的一行")
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
            picked = _pick_value_column(headers, cells, period)
            if not picked:
                continue
            index, column, value, cumulative = picked
            if not _value_in_source(value, text):
                continue
            best_rank = rank
            best = {
                "value": value,
                "column": column,
                "unit": _unit_of(text),
                "quote": _quote(headers, cells, text),
                "rank": rank,
                "reported_yoy": _reported_yoy(headers, cells, index, column),
                "cumulative": cumulative,
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


def _fact_subject(
    meta: dict,
    row: dict,
    requested: list[str] | None = None,
) -> str:
    """研报展示用主体：优先对齐需求简称，其次文件名，最后才用库里的 subject。

    库内 subject 曾被封面套话污染（公开发行证券 / 股份有限公司），
    不能直接拿来写正文和做缺口对账。
    """
    filename = str(row.get("filename") or meta.get("filename") or "")
    stored = str(meta.get("subject") or row.get("subject") or "").strip()
    from_fn = extract_subject(filename)
    aliases = [
        str(item).strip()
        for item in (meta.get("subject_aliases") or row.get("subject_aliases") or [])
        if str(item).strip()
    ]
    for name in requested or []:
        token = (name or "").strip()
        if not token:
            continue
        # 需求简称能对上文件名、库内主体或别名时，用简称写正文（缺口也对得上）
        # 例：需求「韦尔股份」、库内「豪威集团」且 aliases 含韦尔 → 仍写韦尔股份
        if (
            name_in_text(filename, token)
            or name_in_text(stored, token)
            or name_in_text(from_fn, token)
            or any(
                name_in_text(alias, token) or name_in_text(token, alias)
                for alias in aliases
            )
        ):
            return token
    # 对不上需求名单时：优先库内合理全称，再退回文件名
    if is_plausible_subject(stored):
        return stored
    if is_plausible_subject(from_fn):
        return from_fn
    return from_fn or stored


def _pick_facts(
    rows: list[dict],
    files: dict[str, dict],
    aliases: tuple[str, ...],
    indicator: str,
    period: str,
    *,
    requested: list[str] | None = None,
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
        section = str(row.get("section") or "")
        scope, scope_inferred = infer_scope(
            scope=scope,
            section=section,
            column=str(parsed.get("column") or ""),
            quote=str(parsed.get("quote") or ""),
        )
        scope_rank = 2 if scope == SCOPE_CONSOLIDATED else 1 if scope != SCOPE_PARENT else 0
        # 科目本行优先于「其中」明细，合并口径优先于母公司口径
        rank = (parsed["rank"], scope_rank)
        value = parsed["value"]
        # 报表统一写「单位：百万元人民币（百分比除外）」，比率行不能套金额单位。
        unit = ratio_unit(indicator, parsed["unit"], value)
        # 现金流量表等把「编制单位：公司名」误读时 unit 为空；金额类默认元。
        if not unit and indicator in {
            "营业收入",
            "扣非净利润",
            "净利润",
            "利润总额",
            "经营活动产生的现金流量净额",
            "合同负债",
            "存货",
            "在手订单",
            "资本开支",
            "折旧",
        }:
            unit = "元"
        raw_value, raw_unit = value, unit
        value, unit, raw_unit, unit_converted = normalize_amount(value, unit)
        quote = sanitize_evidence_quote(parsed["quote"])
        if unit_converted:
            quote = sanitize_evidence_quote(
                enrich_quote_for_conversion(
                    quote,
                    raw_value=raw_value,
                    raw_unit=raw_unit,
                    value=value,
                    unit=unit,
                )
            )
        basis, months = basis_of(
            period,
            cumulative=bool(parsed.get("cumulative", False)),
            name=indicator,
            unit=unit,
            value=value,
        )
        fact = IndicatorFact(
            subject=_fact_subject(meta, row, requested),
            period=period,
            name=indicator,
            value=value,
            unit=unit,
            column=parsed["column"],
            scope=scope,
            reported_yoy=parsed.get("reported_yoy", ""),
            cumulative=bool(parsed.get("cumulative", False)),
            basis=basis,
            coverage_months=months,
            scope_inferred=scope_inferred,
            raw_unit=raw_unit if unit_converted else "",
            unit_converted=unit_converted,
            evidence=Evidence(
                file_id=file_id,
                filename=str(row.get("filename") or meta.get("filename") or ""),
                page=row.get("page"),
                page_end=row.get("page_end"),
                section=section,
                quote=quote,
            ),
        )
        previous = found.get(file_id)
        if previous is None or rank > previous[0]:
            found[file_id] = (rank, fact)
    return [item[1] for item in found.values()]


def _narrative_facts(
    indicator: str,
    period: str,
    file_ids: list,
    by_id: dict[str, dict],
    *,
    requested: list[str] | None = None,
) -> list[IndicatorFact]:
    """从正文叙述里抽比率/金额。表格路径优先，这里只做补位。

    每份文件至多一条，必须能回指到原文句子与页码；数字和单位写死在模式里，
    不允许「看起来像」就取。抽不到返回空列表，由调用方写缺口。
    """
    keyword, pattern, unit = _NARRATIVE_PATTERNS[indicator]
    try:
        rows = store.search_kb_chunks_keyword(
            [keyword], file_ids=file_ids, top_k=10, kinds=["text"]
        )
    except Exception:
        logger.exception("抽取正文指标时检索失败 indicator=%s", indicator)
        return []
    regex = re.compile(pattern)
    facts: list[IndicatorFact] = []
    # 同一份文件会被切成多个 chunk，经营分析章节常常横跨两块。不去重的话
    # 一个指标会出两条互斥的数字（集团总额 / 人民币明细 / 脚注后的另一个数）。
    seen_files: set[str] = set()
    for row in rows:
        text = str(row.get("text") or "")
        match = regex.search(text)
        if not match:
            continue
        value = match.group(1).strip()
        if not _value_in_source(value, text):
            continue
        quote = sanitize_evidence_quote(
            _sentence_around(text, match.start(), match.end())
        )
        if not _value_in_source(value, quote):
            quote = sanitize_evidence_quote(
                text[max(0, match.start() - 40) : min(len(text), match.end() + 40)].strip()
            )
        file_id = str(row.get("file_id") or "")
        if file_id in seen_files:
            continue
        seen_files.add(file_id)
        meta = by_id.get(file_id, {})
        is_amount = unit != "%"
        section = str(row.get("section") or "")
        scope, scope_inferred = infer_scope(
            scope=str(row.get("scope") or ""),
            section=section,
            column="经营情况简要分析（正文叙述）",
            quote=quote,
        )
        raw_value, raw_unit = value, unit
        value, unit, raw_unit, unit_converted = normalize_amount(value, unit)
        if unit_converted:
            quote = sanitize_evidence_quote(
                enrich_quote_for_conversion(
                    quote,
                    raw_value=raw_value,
                    raw_unit=raw_unit,
                    value=value,
                    unit=unit,
                )
            )
        basis, months = basis_of(
            period,
            cumulative=is_amount,
            name=indicator,
            unit=unit,
            value=value,
        )
        facts.append(
            IndicatorFact(
                subject=_fact_subject(meta, row, requested),
                period=period,
                name=indicator,
                value=value,
                unit=unit,
                column="经营情况简要分析（正文叙述）",
                scope=scope,
                cumulative=is_amount,
                basis=basis,
                coverage_months=months,
                scope_inferred=scope_inferred,
                raw_unit=raw_unit if unit_converted else "",
                unit_converted=unit_converted,
                evidence=Evidence(
                    file_id=file_id,
                    filename=str(row.get("filename") or meta.get("filename") or ""),
                    page=row.get("page"),
                    page_end=row.get("page_end"),
                    section=str(row.get("section") or ""),
                    quote=quote,
                ),
            )
        )
    return facts


def _sentence_around(text: str, start: int, end: int) -> str:
    """取包含命中的整句作为证据片段，避免只给一个孤立数字。"""
    left = max(text.rfind("。", 0, start), text.rfind("\n", 0, start))
    right = text.find("。", end)
    stop = right + 1 if right >= 0 else len(text)
    return text[left + 1 : stop].strip()[:300]


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


def _pick_value_column(
    headers: list[str], cells: list[str], period: str
) -> tuple[int, str, str, bool] | None:
    """在同一行里挑出本期的数值列，返回 (列下标, 列表头, 数值, 是否累计)。

    优先级：累计列 > 本期列 > 未标明的数字列。季报常把「本报告期」(单季)
    和「年初至报告期末」(累计) 并排，研报默认说的是累计值，取错列会差好几倍。
    上年同期、同比增减、以及表头里写着另一个年份的列一律不参与挑选。
    同优先级保留靠前的列，这样原来只有单列「本报告期」的表格行为不变。

    「是否累计」随数值一起带出去：compare 只有确认三期都是累计口径，
    才允许用相邻累计值相减还原单季，避免把单季数当累计去减。
    """
    year = period[:4] if len(period) >= 4 else ""
    best: tuple[int, int, str, str, bool] | None = None
    for index, cell in enumerate(cells):
        if index == 0 or not _NUMBER.match(cell.strip()):
            continue
        header = headers[index] if index < len(headers) else ""
        if _PRIOR_COL.search(header) or _CHANGE_COL.search(header):
            continue
        other_year = re.search(r"((?:19|20)\d{2})", header)
        if year and other_year and other_year.group(1) != year:
            continue
        cumulative = bool(_CUMULATIVE_COL.search(header))
        # 一季报通常只有「本报告期」列，覆盖即为 1–3 月，与「年初至报告期末」等价；
        # 不标成累计会导致 Q3 环比推导误报「Q1/H1 需同为累计口径」。
        if (
            not cumulative
            and period.endswith("Q1")
            and _CURRENT_COL.search(header)
        ):
            cumulative = True
        if cumulative:
            priority = 3
        elif (year and year in header) or _CURRENT_COL.search(header):
            priority = 2
        else:
            priority = 1
        if best is None or priority > best[0]:
            best = (priority, index, header, cell.strip(), cumulative)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def _reported_yoy(
    headers: list[str], cells: list[str], chosen_index: int, chosen_header: str
) -> str:
    """取与所选数值列同口径的「比上年同期增减」列，作为原文披露的同比。

    只认表头以所选列名开头的那一列（例如选了「年初至报告期末」，就配
    「年初至报告期末比上年同期增减(%)」），避免把单季同比配到累计值上。
    单元格里通常不带 %（表头已写单位），这里补回去。
    """
    base = (chosen_header or "").strip()
    for index in range(chosen_index + 1, len(cells)):
        header = headers[index] if index < len(headers) else ""
        if not header or not _CHANGE_COL.search(header):
            continue
        if base and not (header.startswith(base) or base.startswith(header)):
            continue
        if index >= len(cells) or not _NUMBER.match(cells[index].strip()):
            continue
        return f"{cells[index].strip().rstrip('%')}%"
    return ""


def _value_in_source(value: str, source: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    if not compact:
        return False
    return compact in re.sub(r"\s+", "", source)


def _unit_of(text: str) -> str:
    """只认金额／比率单位词。现金流量表常把编制单位写成「单位：公司全称」，不能当计量单位。"""
    found = _UNIT.search(text or "")
    if not found:
        return ""
    raw = found.group(1)
    matched = _UNIT_KEY.match(raw)
    if matched:
        return matched.group(1)
    # 未命中已知单位词 → 空串，由 compose 对金额类补「元」，避免公司名污染。
    return ""


def sanitize_evidence_quote(text: str) -> str:
    """清洗证据引文里的 PDF／OCR 噪音，避免断字数字进正文与校验。

    - 千分位断字：``3, 70`` → ``370``；``1, 614, 059`` → ``1614059``
    - 百分号断字：``783. %`` → ``783%``
    - 勾选框：``□适用 √不适用`` 等删掉
    - 误入正文的章节标记：``## （三）`` 删掉
    """
    if not text:
        return ""
    cleaned = text
    cleaned = _QUOTE_CHECKBOX.sub(" ", cleaned)
    cleaned = _QUOTE_MD_HEADING.sub(" ", cleaned)
    cleaned = _QUOTE_PCT_BREAK.sub(r"\1%", cleaned)
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _QUOTE_COMMA_BREAK.sub(r"\1\3", cleaned)
    return " ".join(cleaned.split())


def _quote(headers: list[str], cells: list[str], text: str) -> str:
    lines: list[str] = []
    unit = _unit_of(text)
    if unit:
        lines.append(f"单位：{unit}")
    if any(headers):
        lines.append(" | ".join(headers))
    lines.append(" | ".join(cells))
    quote = sanitize_evidence_quote("\n".join(lines).strip())
    return quote if len(quote) <= 500 else quote[:500]


def _as_uuid(value: object) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None
