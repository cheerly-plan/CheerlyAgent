"""巨潮资讯网：按股票代码查定期报告公告并下载 PDF。

只做检索与下载校验；入库走 ``kb.fetch_reports``。默认限速、域名白名单。
选文策略：先看全文 PDF 体积，超过 ``cninfo_full_max_mb`` 改抓摘要；
体积未知且有摘要时也走摘要（避免超大扫描件堵 OCR）；没有摘要仍下全文。
接口参数可能变更，调用方必须 mock 网络做单测。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from ai_agent.core import http
from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger

logger = get_logger(__name__)

_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
_STOCK_URL = "https://www.cninfo.com.cn/new/data/szse_stock.json"
_PDF_HOSTS = frozenset({"static.cninfo.com.cn"})
_PDF_BASE = "https://static.cninfo.com.cn/"

_PERIOD_RE = re.compile(r"^(?P<year>19|20)\d{2}(?:Q[1-4]|H1)?$", re.I)
_CODE_RE = re.compile(r"^\d{6}$")
# 各类报告都排除英文/更正等；全文与摘要都保留，由体积策略二选一
_TITLE_SKIP = re.compile(
    r"(英文|English|更正|修订|审计报告|补充|取消|取消公告|意见书)"
)
_SUMMARY = re.compile(r"摘要")
_H_SHARE = re.compile(r"(^H股|H股公告|港股)", re.I)

_CATEGORY = {
    "Q1": "category_yjdbg_szsh",
    "Q2": "category_bndbg_szsh",  # 二季报口径少见，并入半年报
    "Q3": "category_sjdbg_szsh",
    "Q4": "category_ndbg_szsh",
    "H1": "category_bndbg_szsh",
    "Y": "category_ndbg_szsh",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": (
        "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch"
        "?url=disclosure/list/search"
    ),
}

_stock_cache: dict[str, "StockInfo"] | None = None
_stock_cache_at = 0.0
_last_request_at = 0.0


@dataclass(frozen=True)
class StockInfo:
    code: str
    org_id: str
    name: str
    column: str
    plate: str


@dataclass(frozen=True)
class Announcement:
    announcement_id: str
    title: str
    sec_code: str
    sec_name: str
    adjunct_url: str
    pdf_url: str
    announce_date: str  # YYYY-MM-DD，可能为空
    size_bytes: int | None = None  # 公告元数据里的附件大小，可能为空


@dataclass
class DownloadResult:
    announcement: Announcement
    content: bytes
    filename_stem: str


def normalize_period(period: str) -> str:
    text = (period or "").strip().upper()
    if not _PERIOD_RE.fullmatch(text):
        raise ValueError("period 须为 YYYY / YYYYnQ / YYYYH1，例如 2025Q3")
    return text


def normalize_code(code: str) -> str:
    text = (code or "").strip()
    if not _CODE_RE.fullmatch(text):
        raise ValueError("股票代码须为 6 位数字")
    return text


def period_label(period: str) -> str:
    """规范文件名用的中文报告类型：2025Q3 → 2025年第三季度报告。"""
    period = normalize_period(period)
    year = period[:4]
    if period.endswith("Q1"):
        return f"{year}年第一季度报告"
    if period.endswith("Q2") or period.endswith("H1"):
        return f"{year}年半年度报告"
    if period.endswith("Q3"):
        return f"{year}年第三季度报告"
    if period.endswith("Q4"):
        return f"{year}年年度报告"
    return f"{year}年年度报告"


def category_for_period(period: str) -> str:
    period = normalize_period(period)
    if len(period) == 4:
        return _CATEGORY["Y"]
    if period.endswith("H1"):
        return _CATEGORY["H1"]
    return _CATEGORY[period[-2:]]


def is_summary_title(title: str) -> bool:
    return bool(_SUMMARY.search(title or ""))


def full_report_size_limit_bytes() -> int:
    return max(int(float(settings.cninfo_full_max_mb) * 1024 * 1024), 1)


def se_date_for_period(period: str) -> str:
    """披露窗口：覆盖报告所属年与次年一季度（年报常跨年披露）。"""
    period = normalize_period(period)
    year = int(period[:4])
    return f"{year}-01-01~{year + 1}-04-30"


def market_for_code(code: str) -> tuple[str, str]:
    """返回 ``(column, plate)``。北交所 8/4/92 开头；沪市 6；其余深市。"""
    if code.startswith(("8", "4", "92")):
        return "bj", "bj"
    if code.startswith("6"):
        return "sse", "sh"
    return "szse", "sz"


def lookup_stock(code: str) -> StockInfo:
    code = normalize_code(code)
    cache = _load_stock_map()
    info = cache.get(code)
    if info is None:
        raise LookupError(f"巨潮股票列表里找不到代码 {code}")
    return info


def resolve_code_by_name(name: str) -> StockInfo | None:
    """用公司全称/简称反查股票代码。歧义（同分多个）时返回 None。"""
    needle = re.sub(r"\s+", "", (name or "").strip())
    if not needle:
        return None
    if _CODE_RE.fullmatch(needle):
        try:
            return lookup_stock(needle)
        except LookupError:
            return None
    cache = _load_stock_map()
    exact = [info for info in cache.values() if info.name and info.name == needle]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None
    # 简称落在全称里，或全称包含简称：按简称长度取最长唯一命中
    scored: list[tuple[int, StockInfo]] = []
    for info in cache.values():
        short = (info.name or "").strip()
        if len(short) < 2:
            continue
        if short in needle or needle in short:
            scored.append((len(short), info))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best_len = scored[0][0]
    tops = [info for length, info in scored if length == best_len]
    codes = {info.code for info in tops}
    if len(codes) != 1:
        return None
    return tops[0]


def resolve_codes_in_text(text: str) -> list[StockInfo]:
    """从需求文本里点名的公司/代码解析股票，按简称长度优先、互不重叠。"""
    hay = re.sub(r"\s+", "", text or "")
    if not hay:
        return []
    found: list[StockInfo] = []
    # 正文里直接写的 6 位代码
    for match in re.finditer(r"(?<!\d)(\d{6})(?!\d)", hay):
        try:
            found.append(lookup_stock(match.group(1)))
        except LookupError:
            continue
    cache = _load_stock_map()
    ranked = sorted(
        (info for info in cache.values() if info.name and len(info.name) >= 2),
        key=lambda info: len(info.name),
        reverse=True,
    )
    occupied: list[tuple[int, int]] = []
    for info in ranked:
        start = hay.find(info.name)
        if start < 0:
            continue
        span = (start, start + len(info.name))
        if any(not (span[1] <= old[0] or span[0] >= old[1]) for old in occupied):
            continue
        occupied.append(span)
        if all(item.code != info.code for item in found):
            found.append(info)
    return found


def search_announcements(
    code: str,
    period: str,
    *,
    max_pages: int = 5,
    max_full_bytes: int | None = None,
) -> list[Announcement]:
    """查某代码某报告期的定期报告公告，并按体积策略选出要下载的一份。

    英文版 / H 股 / 更正公告排除；全文与摘要都参与候选，最终只返回 0～1 条。
    ``max_full_bytes`` 传给全文/摘要二选一；深度年报路径可抬高阈值优先全文。
    """
    code = normalize_code(code)
    period = normalize_period(period)
    stock = lookup_stock(code)
    category = category_for_period(period)
    se_date = se_date_for_period(period)
    wanted = period_label(period)
    rows: list[Announcement] = []
    for page in range(1, max(1, max_pages) + 1):
        payload = _query_page(
            stock=stock,
            category=category,
            se_date=se_date,
            page_num=page,
        )
        batch = payload.get("announcements") or []
        if not batch:
            break
        for raw in batch:
            item = _parse_announcement(raw)
            if item is None:
                continue
            if not _title_matches_period(item.title, period, wanted):
                continue
            rows.append(item)
        total_pages = int(payload.get("totalpages") or 0)
        if page >= total_pages and total_pages > 0:
            break
        if len(batch) < 30:
            break
    chosen = choose_report_announcement(rows, max_full_bytes=max_full_bytes)
    return [chosen] if chosen is not None else []


def choose_report_announcement(
    items: list[Announcement],
    *,
    max_full_bytes: int | None = None,
) -> Announcement | None:
    """按体积在全文 / 摘要之间二选一。

    - 只有一种：直接用
    - 两种都有：全文不超过阈值 → 全文；超过或体积未知 → 摘要
    """
    if not items:
        return None
    fulls = _prefer_primary(
        [item for item in items if not is_summary_title(item.title)]
    )
    summaries = _prefer_primary(
        [item for item in items if is_summary_title(item.title)],
        prefer_summary=True,
    )
    full = fulls[0] if fulls else None
    summary = summaries[0] if summaries else None
    if full is None:
        return summary
    if summary is None:
        return full

    limit = (
        max_full_bytes
        if max_full_bytes is not None
        else full_report_size_limit_bytes()
    )
    size = resolve_announcement_size(full)
    if size is None:
        logger.info(
            "巨潮全文体积未知，改用摘要 id=%s title=%s",
            summary.announcement_id,
            summary.title,
        )
        return summary
    if size > limit:
        logger.info(
            "巨潮全文 %.2fMB 超过阈值 %.2fMB，改用摘要 id=%s",
            size / (1024 * 1024),
            limit / (1024 * 1024),
            summary.announcement_id,
        )
        return summary
    return full


def resolve_announcement_size(item: Announcement) -> int | None:
    """公告自带 size → 否则 HEAD Content-Length。"""
    if item.size_bytes is not None and item.size_bytes >= 0:
        return item.size_bytes
    _throttle()
    return http.request_content_length(
        item.pdf_url, headers=_HEADERS, timeout=10
    )


def download_pdf(announcement: Announcement, *, company_name: str, period: str) -> DownloadResult:
    """下载并校验 PDF；落盘名用规范 stem，不含公告日期前缀。"""
    url = announcement.pdf_url
    _assert_pdf_url(url)
    _throttle()
    max_bytes = min(settings.kb_max_file_mb, 50) * 1024 * 1024
    content, content_type = http.request_bytes(
        url,
        headers=_HEADERS,
        timeout=60,
        max_bytes=max_bytes,
    )
    if "pdf" not in content_type.lower() and not content.startswith(b"%PDF"):
        raise ValueError(f"响应不是 PDF：content_type={content_type!r}")
    if not content.startswith(b"%PDF"):
        raise ValueError("PDF 魔数校验失败")
    stem = _canonical_stem(
        company_name or announcement.sec_name,
        period,
        summary=is_summary_title(announcement.title),
    )
    return DownloadResult(announcement=announcement, content=content, filename_stem=stem)


def assert_allowed_pdf_url(url: str) -> None:
    """供 URL 入库等外部入口复用的域名白名单校验。"""
    _assert_pdf_url(url)


def clear_stock_cache() -> None:
    """单测用：清空股票列表缓存。"""
    global _stock_cache, _stock_cache_at
    _stock_cache = None
    _stock_cache_at = 0.0


def _load_stock_map() -> dict[str, StockInfo]:
    global _stock_cache, _stock_cache_at
    now = time.monotonic()
    ttl = max(int(settings.cninfo_stock_cache_ttl_seconds), 60)
    if _stock_cache is not None and now - _stock_cache_at < ttl:
        return _stock_cache
    _throttle()
    payload = http.request_json(_STOCK_URL, headers=_HEADERS, timeout=30)
    rows = payload.get("stockList") if isinstance(payload, dict) else payload
    mapping: dict[str, StockInfo] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = str(row.get("code") or "").strip()
            org_id = str(row.get("orgId") or "").strip()
            name = str(row.get("zwjc") or "").strip()
            if not _CODE_RE.fullmatch(code) or not org_id:
                continue
            column, plate = market_for_code(code)
            mapping[code] = StockInfo(
                code=code, org_id=org_id, name=name, column=column, plate=plate
            )
    if not mapping:
        raise RuntimeError("巨潮股票列表为空或解析失败")
    _stock_cache = mapping
    _stock_cache_at = now
    logger.info("巨潮股票列表已缓存 count=%s", len(mapping))
    return mapping


def _query_page(
    *,
    stock: StockInfo,
    category: str,
    se_date: str,
    page_num: int,
) -> dict:
    _throttle()
    return http.request_form(
        _QUERY_URL,
        {
            "pageNum": page_num,
            "pageSize": 30,
            "column": stock.column,
            "tabName": "fulltext",
            "plate": stock.plate,
            "stock": f"{stock.code},{stock.org_id}",
            "searchkey": "",
            "secid": "",
            "category": f"{category};",
            "trade": "",
            "seDate": se_date,
            "isHLtitle": "true",
        },
        headers=_HEADERS,
        timeout=15,
    )


def _parse_announcement(raw: object) -> Announcement | None:
    if not isinstance(raw, dict):
        return None
    title = _clean_title(str(raw.get("announcementTitle") or ""))
    adjunct = str(raw.get("adjunctUrl") or "").strip().lstrip("/")
    ann_id = str(raw.get("announcementId") or "").strip()
    if not title or not adjunct or not ann_id:
        return None
    if _TITLE_SKIP.search(title) or _H_SHARE.search(title):
        return None
    if not adjunct.lower().endswith(".pdf"):
        return None
    pdf_url = adjunct if adjunct.startswith("http") else _PDF_BASE + adjunct
    try:
        _assert_pdf_url(pdf_url)
    except ValueError:
        return None
    ts = raw.get("announcementTime")
    announce_date = ""
    if isinstance(ts, (int, float)) and ts > 0:
        announce_date = time.strftime("%Y-%m-%d", time.localtime(ts / 1000))
    size_bytes = _parse_adjunct_size(raw.get("adjunctSize"))
    return Announcement(
        announcement_id=ann_id,
        title=title,
        sec_code=str(raw.get("secCode") or "").strip(),
        sec_name=str(raw.get("secName") or "").strip(),
        adjunct_url=adjunct,
        pdf_url=pdf_url,
        announce_date=announce_date,
        size_bytes=size_bytes,
    )


def _parse_adjunct_size(raw: object) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _title_matches_period(title: str, period: str, wanted_label: str) -> bool:
    # 标题常含完整「2025年第三季度报告」；也接受「2025年三季度报告」/「…报告摘要」
    if wanted_label in title:
        return True
    year = period[:4]
    if year not in title:
        return False
    if period.endswith("Q3") and ("第三季度" in title or "三季度" in title):
        return True
    if period.endswith("Q1") and ("第一季度" in title or "一季度" in title):
        return True
    if (period.endswith("H1") or period.endswith("Q2")) and (
        "半年度" in title or "中期" in title
    ):
        return True
    if len(period) == 4 or period.endswith("Q4"):
        return "年度报告" in title and "半年度" not in title and "季度" not in title
    return False


def _prefer_primary(
    items: list[Announcement], *, prefer_summary: bool = False
) -> list[Announcement]:
    """同组内排序：可优先摘要；再优先标题含「股份有限公司」。"""
    if not items:
        return []
    ranked = sorted(
        items,
        key=lambda item: (
            0 if (prefer_summary and is_summary_title(item.title)) else 1,
            0 if "股份有限公司" in item.title else 1,
            -len(item.title),
            item.announcement_id,
        ),
    )
    seen: set[str] = set()
    out: list[Announcement] = []
    for item in ranked:
        if item.announcement_id in seen:
            continue
        seen.add(item.announcement_id)
        out.append(item)
    return out


def _canonical_stem(company: str, period: str, *, summary: bool = False) -> str:
    name = re.sub(r'[\\/:*?"<>|\s]+', "", (company or "").strip()) or "未命名公司"
    # 标题里可能已带「股份有限公司…报告」整句，只取公司主体段
    for marker in ("股份有限公司", "有限公司", "集团", "银行", "证券"):
        idx = name.find(marker)
        if idx >= 0:
            name = name[: idx + len(marker)]
            break
    label = period_label(period)
    if summary and not label.endswith("摘要"):
        label = f"{label}摘要"
    if name.endswith(label):
        return name[:80]
    return f"{name}{label}"[:80]


def _clean_title(title: str) -> str:
    text = re.sub(r"<[^>]+>", "", title or "")
    return re.sub(r"\s+", " ", text).strip()


def _assert_pdf_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("PDF 链接必须是 https")
    host = (parsed.hostname or "").lower()
    if host not in _PDF_HOSTS:
        raise ValueError(f"PDF 域名不在白名单：{host}")
    if not parsed.path.lower().endswith(".pdf"):
        raise ValueError("PDF 路径后缀非法")


def _throttle() -> None:
    global _last_request_at
    gap = max(float(settings.cninfo_request_interval_seconds), 0.2)
    now = time.monotonic()
    wait = gap - (now - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()
