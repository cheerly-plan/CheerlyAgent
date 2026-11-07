"""经营解读证据：原文清洗、KB 召回、证据包。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ai_agent.core.logging import get_logger
from ai_agent.report.extract import (
    IndicatorFact,
    citation_anchor,
    _as_uuid,
    _files_for,
)
from ai_agent.report.parse import ReportPlan
from ai_agent.report.analyze_heuristics import (
    cross_industry_accounts as _cross_industry_accounts,
    evidence_keywords as _evidence_keywords,
)
from ai_agent.storage.store import store

logger = get_logger(__name__)

EVIDENCE_KEYWORDS = _evidence_keywords()
CROSS_INDUSTRY_ACCOUNTS = _cross_industry_accounts()


@dataclass
class EvidenceItem:
    evidence_id: str
    file_id: str
    filename: str
    page: int | None
    section: str
    quote: str


def is_statement_template(text: str) -> bool:
    return any(
        token in text
        for token in ("合并利润表", "合并资产负债表", "编制单位", "审计类型")
    )


def is_form_chrome(text: str) -> bool:
    """抽后的短句若仍是表头／勾选框，才丢弃。"""
    if "适用" in text and "不适用" in text:
        return True
    if "变动比例" in text and "主要原因" in text:
        return True
    return is_statement_template(text)


def iter_reason_snippets(quote: str) -> list[str]:
    """同一证据块可拆出多条叙述：量价齐升与毛利润增加不要互相抢占。"""
    # 主题分组：每组至多取一条；组内取文中最早命中，避免「量价」截掉「报告期内」开头。
    theme_groups = (
        ("报告期内", "量价", "存储", "出货", "经营业绩"),
        ("毛利润增加", "主要是", "主要系"),
    )
    snippets: list[str] = []
    occupied: list[tuple[int, int]] = []
    for group in theme_groups:
        hits = sorted(
            ((quote.find(marker), marker) for marker in group if marker in quote),
            key=lambda item: item[0],
        )
        for idx, _marker in hits:
            if idx < 0:
                continue
            if any(start <= idx < end for start, end in occupied):
                continue
            tail = quote[idx:]
            caused = tail.find("所致")
            if caused >= 0 and caused < 80:
                piece = tail[: caused + 2]
            else:
                cut = re.search(
                    r"\s+(?:营业收入|利润总额|归属于上市公司|基本每股|稀释每股|加权平均)",
                    tail,
                )
                piece = tail[: cut.start()] if cut else tail
                stop = piece.find("。")
                if stop >= 0:
                    piece = piece[: stop + 1]
            cleaned = keep_operating_fragments(
                strip_quote_numbers(piece.strip(" ；;|"))
            )
            if not cleaned:
                continue
            snippets.append(cleaned)
            occupied.append((idx, idx + max(len(piece), 1)))
            break
        if len(snippets) >= 2:
            break
    if snippets:
        return snippets
    fallback = keep_operating_fragments(reason_only(quote))
    return [fallback] if fallback else []


TIP_PREFIX = re.compile(
    r"需提醒投资者关注的关于公司报告期经营情况的其他重要信息[、,，]?"
)
SHAREHOLDER_NOISE = (
    "转融通",
    "出借股份",
    "无限售流通股",
    "前十名股东",
    "前 名股东",
    "前名股东",
)
OPERATING_MARKERS = (
    "报告期内",
    "公司前三季度",
    "前三季度",
    "本报告期",
    "本期公司",
    "围绕重点",
    "持续优化",
    "经营业绩",
    "量价",
    "毛利润增加",
    "主要是",
    "主要系",
)
OPERATING_CONTENT = (
    "产能",
    "客户",
    "技术",
    "经营",
    "毛利",
    "量价",
    "研发",
    "应用板块",
    "出货",
    "存储",
)
# 利润表模板科目；成串出现 = 科目清单，不是经营解读。
PL_ACCOUNT_NAMES = (
    "营业成本",
    "利息支出",
    "手续费及佣金支出",
    "退保金",
    "赔付支出",
    "提取保险责任准备金净额",
    "提取保险责任准备金",
    "保单红利支出",
    "分保费用",
    "税金及附加",
    "销售费用",
    "管理费用",
    "研发费用",
    "财务费用",
    "资产减值损失",
    "信用减值损失",
    "其他收益",
    "投资收益",
    "公允价值变动收益",
    "营业外收入",
    "营业外支出",
    "所得税费用",
)
# 封测／制造公司正文不该出现的跨行业科目（保险／银行模板残留）。
NARRATIVE_CUES = (
    "优化",
    "布局",
    "增长",
    "下降",
    "提升",
    "推进",
    "持续",
    "实现",
    "所致",
    "主要是",
    "主要系",
    "围绕",
    "保持",
    "量价",
    "出货",
    "发力",
    "向好",
)
HOLLOW_NUMBER_SLOT = re.compile(
    r"(＿|_{2,}|人民币\s+亿?元|为\s+元|同比\s*(?:增长|下降|增减)?\s*[。.%％]?)"
)


def plain_evidence_quote(raw: str) -> str:
    """去掉 HTML 标签，压成单行，并清洗 PDF 断字／勾选框噪音。"""
    from ai_agent.report.extract import sanitize_evidence_quote

    text = re.sub(r"<[^>]+>", " ", raw or "")
    return sanitize_evidence_quote(text)


def is_account_catalog(text: str) -> bool:
    """整段基本是利润表科目名罗列（长电第 2 条引文这类）。"""
    hits = sum(1 for name in PL_ACCOUNT_NAMES if name in (text or ""))
    if hits >= 4:
        return True
    return "其中" in (text or "") and hits >= 2


def has_hollow_number_slots(text: str) -> bool:
    """数字被抽空后留下「人民币 亿元／同比 。」半截句。"""
    return bool(HOLLOW_NUMBER_SLOT.search(text or ""))


def looks_like_operating_sentence(text: str) -> bool:
    if any(mark in text for mark in ("。", "；", "所致")):
        return True
    return any(cue in text for cue in NARRATIVE_CUES)


def is_usable_operating_quote(text: str) -> bool:
    """单片段是否可引用：成句、非科目清单、无跨行业残留、无数字空壳。"""
    cleaned = (text or "").strip()
    if len(cleaned) < 20:
        return False
    if any(token in cleaned for token in CROSS_INDUSTRY_ACCOUNTS):
        return False
    if is_account_catalog(cleaned):
        return False
    if has_hollow_number_slots(cleaned):
        return False
    if not looks_like_operating_sentence(cleaned):
        return False
    return True


def excise_bad_clauses(text: str) -> str:
    """按逗号切开，丢掉空缺／科目罗列／跨行业子句，保留其余。"""
    parts = re.split(r"([，,])", text or "")
    kept: list[str] = []
    idx = 0
    while idx < len(parts):
        chunk = parts[idx]
        delim = ""
        if idx + 1 < len(parts) and parts[idx + 1] in {"，", ","}:
            delim = parts[idx + 1]
            idx += 2
        else:
            idx += 1
        piece = chunk.strip()
        if not piece:
            continue
        if has_hollow_number_slots(piece):
            continue
        if is_account_catalog(piece):
            continue
        if any(token in piece for token in CROSS_INDUSTRY_ACCOUNTS):
            continue
        kept.append(chunk + delim)
    return "".join(kept).strip(" ，,、；;")


def keep_operating_fragments(text: str) -> str:
    """片段级切除：空缺句／科目清单丢掉，经营成句保留（不因半段瑕疵整段扔）。"""
    kept: list[str] = []
    for sentence in re.split(r"(?<=[。；])\s*", text or ""):
        piece = sentence.strip()
        if not piece:
            continue
        if is_account_catalog(piece) and not any(
            cue in piece for cue in ("围绕", "优化", "布局", "量价", "产能落地")
        ):
            continue
        if any(token in piece for token in CROSS_INDUSTRY_ACCOUNTS):
            # 整句都是保险／银行残留：丢；若只是尾巴污染，先切除再判。
            if not any(cue in piece for cue in NARRATIVE_CUES):
                continue
        if has_hollow_number_slots(piece) or any(
            token in piece for token in CROSS_INDUSTRY_ACCOUNTS
        ):
            piece = excise_bad_clauses(piece)
        piece = piece.strip(" ，,、；;")
        if not piece:
            continue
        if not is_usable_operating_quote(piece):
            # 切除后可能变短，但仍有明确经营线索则保留。
            if len(piece) < 16 or not looks_like_operating_sentence(piece):
                continue
            if has_hollow_number_slots(piece) or is_account_catalog(piece):
                continue
        if not piece.endswith(("。", "；", "！", "？")):
            piece += "。"
        kept.append(piece)
    return "".join(kept)


def extract_operating_narrative(text: str) -> str:
    """从拼接块里精准切出经营原文：垃圾片段丢、有效片段留。

    长电常见形态：股东转融通前缀 + 经营句 + 「净利润为人民币 亿元」空缺尾巴。
    不能因尾巴有空缺整段扔掉（否则经营解读退化成指标复读）。
    """
    cleaned = TIP_PREFIX.sub("", text or "")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return ""
    start = -1
    for marker in OPERATING_MARKERS:
        idx = cleaned.find(marker)
        if idx < 0:
            continue
        if start < 0 or idx < start:
            start = idx
    if start >= 0:
        cleaned = cleaned[start:]
    elif any(token in cleaned for token in SHAREHOLDER_NOISE):
        return ""
    if any(token in cleaned for token in SHAREHOLDER_NOISE):
        cut = min(
            (cleaned.find(token) for token in SHAREHOLDER_NOISE if token in cleaned),
            default=-1,
        )
        if 0 <= cut < 40:
            return ""
        if cut >= 40:
            cleaned = cleaned[:cut].rstrip(" ，,、；;")
    if cleaned.startswith("前") and "股东" in cleaned[:24]:
        return ""
    return keep_operating_fragments(cleaned)


def reason_only(quote: str) -> str:
    """变动原因表常带比例数字与其它科目行；只保留原因叙述句。

    定量数字必须来自已校验 facts；原文引述只留定性原因，避免断字残片进正文。
    内容词（量价／存储）优先于「主要是」，防止长原文被短原因句抢占。
    """
    markers = ("量价", "存储", "出货", "报告期内", "主要是", "主要系")
    for marker in markers:
        idx = quote.find(marker)
        if idx < 0:
            continue
        tail = quote[idx:]
        # 「…所致」后常接其它科目变动行，先截到所致。
        caused = tail.find("所致")
        if caused >= 0 and caused < 80:
            tail = tail[: caused + 2]
        else:
            # 切断后续科目行（营业收入_本报告期 197.11 …）
            cut = re.search(
                r"\s+(?:营业收入|利润总额|归属于上市公司|基本每股|稀释每股|加权平均)",
                tail,
            )
            if cut:
                tail = tail[: cut.start()]
            stop = tail.find("。")
            if stop >= 0:
                tail = tail[: stop + 1]
        return strip_quote_numbers(tail.strip(" ；;|"))
    return strip_quote_numbers(quote)


def strip_quote_numbers(text: str) -> str:
    cleaned = re.sub(r"-?\d[\d,，]*(?:\.\d+)?%?", " ", text or "")
    return " ".join(cleaned.split())


def safe_truncate(text: str, limit: int) -> str:
    """截断正文引述时不从数字中间切开，避免校验把半截数字当成未核对。"""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if cut and cut[-1].isdigit() and limit < len(text) and text[limit].isdigit():
        cut = re.sub(r"[\d,，.]+$", "", cut).rstrip(" |,，、；;")
    return cut.rstrip() + "…"


def collect_evidence_pack(
    plan: ReportPlan, facts: list[IndicatorFact]
) -> list[EvidenceItem]:
    """KB 召回 + 本期事实出处，供 analyze / 测试共用。"""
    items = load_kb_evidence(plan)
    seen = {(item.file_id, item.page, item.quote[:60]) for item in items}
    index = len(items)
    for fact in facts:
        if fact.period != plan.period:
            continue
        # akshare / 推算出处只作数值补齐，不当作「季报原文」引述进经营解读。
        filename = fact.evidence.filename or ""
        if filename.startswith(("akshare.", "derived.")):
            continue
        quote = extract_operating_narrative(
            plain_evidence_quote(fact.evidence.quote or "")
        )
        if len(quote) < 8:
            continue
        key = (fact.evidence.file_id, fact.evidence.page, quote[:60])
        if key in seen:
            continue
        seen.add(key)
        items.append(
            EvidenceItem(
                evidence_id=f"e{index}",
                file_id=fact.evidence.file_id,
                filename=fact.evidence.filename,
                page=fact.evidence.page,
                section=fact.evidence.section,
                quote=quote[:500],
            )
        )
        index += 1
        if len(items) >= 12:
            break
    return items


def load_kb_evidence(plan: ReportPlan) -> list[EvidenceItem]:
    try:
        files = _files_for(plan.subjects, plan.period)
    except Exception:
        logger.exception("经营解读定位本期文件失败")
        return []
    file_ids = [_as_uuid(row.get("file_id")) for row in files]
    file_ids = [item for item in file_ids if item is not None]
    if not file_ids:
        return []
    rows: list[dict] = []
    for kinds in (["text"], ["table"]):
        try:
            rows.extend(
                store.search_kb_chunks_keyword(
                    list(EVIDENCE_KEYWORDS),
                    file_ids=file_ids,
                    top_k=8,
                    kinds=kinds,
                )
            )
        except Exception:
            logger.exception("经营解读证据召回失败 kinds=%s", kinds)
    ban = ("被合并方", "合并日", "净利润为：0", "净利润为0")
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        text = extract_operating_narrative(
            plain_evidence_quote(str(row.get("text") or ""))
        )
        if len(text) < 20 or any(token in text for token in ban):
            continue
        file_id = str(row.get("file_id") or "")
        page = row.get("page")
        key = f"{file_id}:{page}:{text[:80]}"
        if key in seen:
            continue
        seen.add(key)
        page_int: int | None
        if isinstance(page, int):
            page_int = page
        elif str(page).isdigit():
            page_int = int(page)
        else:
            page_int = None
        items.append(
            EvidenceItem(
                evidence_id=f"e{index}",
                file_id=file_id,
                filename=str(row.get("filename") or ""),
                page=page_int,
                section=str(row.get("section") or ""),
                quote=text[:500],
            )
        )
        if len(items) >= 8:
            break
    return items


