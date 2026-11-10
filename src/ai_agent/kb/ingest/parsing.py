"""文件解析：把上传的原始文件转成带类型的文本段。

支持 PDF / DOCX / XLSX / MD / TXT / CSV。产出 Segment 列表，每段带
kind（正文 / 表格 / 图表）与页码，切片时按 kind 分别处理：

- text：普通正文，PDF 按页拆分，循页码可溯源
- table：Word 表格、Excel 工作表。转成 Markdown 表格保留行列对应关系，
  切片时整表成一片、超长则重复表头（见 chunking）
- figure：PDF 含图页 / DOCX 图片经视觉模型转出的描述。ECharts 这类
  图表的数值不在文件里，只能靠读图近似还原，故单独标记，检索时提示
  「图像识别」以免被当成原文数据

解析失败统一抛 ParseError（带可读原因），由管道记入 file.error。
"""

import re
from dataclasses import dataclass, replace
from itertools import groupby
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.ingest import sections

logger = get_logger(__name__)

TEXT = "text"
TABLE = "table"
FIGURE = "figure"


@dataclass(frozen=True)
class Segment:
    text: str
    # 页码语义：PDF 是页码，Excel 是 sheet 序号，其余为 None
    page: int | None = None
    kind: str = TEXT
    # 所属一级节（如「第十节 管理层讨论与分析」）；无法识别章节的文档为 None
    section: str | None = None
    # 报表口径（consolidated / parent），只对表格段有意义，见 _table_scope
    scope: str | None = None
    # 跨页/拆开后合并的表：起始页在 page，结束页在这里；单页表为 None
    page_end: int | None = None

_SUPPORTED = {".pdf", ".docx", ".xlsx", ".xlsm", ".md", ".markdown", ".txt", ".csv"}


class ParseError(Exception):
    """文件解析失败，消息面向用户展示。"""


def supported(suffix: str) -> bool:
    return suffix.lower() in _SUPPORTED


def supported_label() -> str:
    return "PDF / DOCX / XLSX / MD / TXT / CSV"


def parse_segments(path: Path) -> list[Segment]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix == ".docx":
        # 延迟 import 避免与 office 模块的循环依赖（office 顶层 import parsing 的公共 API）
        from ai_agent.kb.ingest.office import _parse_docx

        return _parse_docx(path)
    if suffix in {".xlsx", ".xlsm"}:
        from ai_agent.kb.ingest.office import _parse_xlsx

        return _parse_xlsx(path)
    if suffix in _SUPPORTED:
        return [
            Segment(path.read_text(encoding="utf-8", errors="replace"))
        ]
    raise ParseError(f"暂不支持的文件类型 {suffix}")


# ——— PDF ———


# OCR 整页里的 HTML 表。表题在表前一行，用来判断合并/母公司口径。
_OCR_TABLE = re.compile(r"<table\b[^>]*>[\s\S]*?</table>", re.IGNORECASE)
# 整页 OCR 没有坐标，页眉页脚的「1 / 16」「第 1 页 / 共 16 页」会混进正文。
# 只删独占一行的页码，句子里的比例不动。
# 视觉模型按「忽略页眉」训练，会把贴顶的证券代码、证券简称、公告编号整行丢掉。
_COVER_FIELD = re.compile(
    r"(?:证券代码|证券简称|股票代码|股票简称|公告编号)\s*[:：]"
)
# 「主要财务数据……如下：」是贴着表的引导句。再往上的正文句子不带。
_TABLE_INTRO = re.compile(r"(?:如下表|如下|下表)\s*[:：]\s*$")
_TABLE_INTRO_MAX = 160
_OCR_PAGE_MARK = re.compile(
    r"^[^\w\u4e00-\u9fff]*"
    r"(?:第\s*)?\d{1,4}\s*页?\s*[/／]\s*(?:共\s*)?\d{1,4}\s*页?"
    r"[^\w\u4e00-\u9fff]*$"
)


def _parse_pdf(path: Path) -> list[Segment]:
    """按页混合：密页用文字层，只有稀字页（图片/扫描页）才送整页 OCR。

    电子年报的正文与三大表本来就在文字层里，逐页 OCR 一遍既慢、又会把
    「有字用字」的精确数字换成视觉模型近似值；更糟的是长文档 OCR 熔断时
    只产出前几十页，旧逻辑只要 OCR 非空就整本采用，后半本全丢（飞测年报
    实测只剩前 30 页）。故：

    - 文字层字符数 ≥ ``kb_ocr_min_text_chars`` = 密页，保留文字层正文与表格；
    - 更少 = 稀字页，送 OCR；**OCR 没产出的页仍保留文字层，绝不整本覆盖**；
    - ``kb_ocr_min_text_chars = 0`` 关闭混合，退回「OCR 有产出就整本采用」旧行为。

    章节标签始终来自文字层，封面字段仍由文字层补回 OCR 结果。OCR 未开启、
    不可用或一页都没识别出来时，走文字层原解析（含表格识别与图表页）。
    混合模式密页依赖文字层表格，不能为躲 SIGSEGV 跳过抽表：
    ``_pdf_tables_isolated`` 已在短命子进程里按页推进，崩溃只丢该页。
    """
    from ai_agent.kb.ingest.figures import _pdf_figures
    from ai_agent.kb.ingest.ocr import enabled as ocr_enabled, ocr_pdf

    prefer_ocr = ocr_enabled()
    min_chars = settings.kb_ocr_min_text_chars
    hybrid = prefer_ocr and min_chars > 0
    include_tables = (not prefer_ocr) or hybrid
    try:
        text_segments, text_lens = _pdf_text(path, include_tables=include_tables)
    except Exception:
        logger.warning("PDF 文本层抽取失败", exc_info=True)
        text_segments, text_lens = [], []

    ocr_targets: list[int] | None = None
    if hybrid:
        ocr_targets = _sparse_pages(path, text_lens, min_chars)

    # 整本都是密页：一次 OCR 都不用跑，直接文字层 + 图表页识别。
    ocr_pages: list[Segment] | None = None
    if ocr_targets != [] or not hybrid:
        ocr_pages = (
            ocr_pdf(path, pages=ocr_targets) if hybrid else ocr_pdf(path)
        )

    if ocr_pages:
        if hybrid:
            merged = _merge_hybrid_pages(ocr_pages, text_segments)
            if merged:
                covered = {s.page for s in ocr_pages if s.page}
                merged.extend(
                    _pdf_figures(path, text_lens, skip_pages=covered or None)
                )
                merged.sort(key=lambda s: s.page or 0)
                logger.info(
                    "kb 混合解析：密页用文字层，OCR 覆盖 %d 页，共 %d 段",
                    len(covered),
                    len(merged),
                )
                return merged
        else:
            segments = _segments_from_ocr(
                ocr_pages, _section_by_page(text_segments), text_segments
            )
            if segments:
                return segments

    # OCR 没产出：若刚才为避 SIGSEGV 跳过了抽表，补跑带表格的文本层。
    if prefer_ocr and not hybrid:
        try:
            text_segments, text_lens = _pdf_text(path, include_tables=True)
        except Exception:
            logger.warning("PDF 文本层（含表格）回退抽取失败", exc_info=True)

    covered = (
        {s.page for s in ocr_pages or [] if s.page} if hybrid else None
    )
    figures = _pdf_figures(path, text_lens, skip_pages=covered or None)
    if any((s.text or "").strip() for s in text_segments):
        text_segments.extend(figures)
        return text_segments
    if not figures:
        raise ParseError("PDF 中没有可提取的文本（可能是扫描件/图片版）")
    return figures


def _sparse_pages(
    path: Path, text_lens: list[int], min_chars: int
) -> list[int] | None:
    """文字层字符数 < ``min_chars`` 的页（1-based），只有这些页值得送 OCR。

    ``_page_char_counts`` 只统计有文字行的页，长度可能短于总页数（末尾
    纯图片页没有文字行）。所以先向 pymupdf 问总页数再补零；取不到页数时
    返回 None，让调用方退回全文 OCR——保守优先，不因拿不到页数就丢内容。
    """
    try:
        import pymupdf

        with pymupdf.open(str(path)) as doc:
            total = doc.page_count
    except Exception:
        logger.warning("混合解析取页数失败，退回全文 OCR", exc_info=True)
        return None
    if total <= 0:
        return []
    counts = list(text_lens[:total]) + [0] * max(0, total - len(text_lens))
    return [i + 1 for i, n in enumerate(counts) if n < min_chars]


def _merge_hybrid_pages(
    ocr_pages: list[Segment], text_segments: list[Segment]
) -> list[Segment]:
    """密页保留文字层，稀字页用 OCR 段替换，按页码归并。

    只替换「OCR 确实产出内容的页」：熔断时没产出的稀字页仍保留文字层，
    避免半截 OCR 把整本电子年报盖掉。已知限制：``_segments_from_ocr`` 的
    报表口径（合并/母公司）在 OCR 页之间连续传递，稀字页分散时中间密页的
    表名不会参与；但表格页文字密度高通常是密页，且深度层数字不取 OCR 值。

    章节标签、封面字段、HTML 表拆分仍复用 ``_segments_from_ocr``，与全文
    OCR 路径逐字一致。
    """
    ocr_segments = _segments_from_ocr(
        ocr_pages, _section_by_page(text_segments), text_segments
    )
    replaced = {s.page for s in ocr_segments if s.page is not None}
    kept = [s for s in text_segments if s.page not in replaced]
    merged = kept + ocr_segments
    merged.sort(key=lambda s: s.page or 0)
    return merged


def _section_by_page(segments: list[Segment]) -> dict[int, str]:
    """每页一个章节标签。页内有多个节时取该页第一段的节，后面的空页沿用上一页。"""
    first: dict[int, str] = {}
    for seg in segments:
        if seg.page and seg.section and seg.page not in first:
            first[seg.page] = seg.section
    if not first:
        return {}
    current: str | None = None
    out: dict[int, str] = {}
    for page in range(1, max(first) + 1):
        if page in first:
            current = first[page]
        if current:
            out[page] = current
    return out


def _segments_from_ocr(
    pages: list[Segment],
    sections_by_page: dict[int, str],
    text_segments: list[Segment],
) -> list[Segment]:
    out: list[Segment] = []
    # 合并资产负债表的表名常常在上一页末尾，续表本身不再印。口径顺着往下传，
    # 直到出现下一张报表名（母公司利润表会把口径改过来）。
    carried_scope: str | None = None
    pending_lead = ""
    for page_seg in pages:
        page = page_seg.page
        section = sections_by_page.get(page) if page else None
        text = _restore_cover_fields(text_segments, page, page_seg.text or "")
        segments, carried_scope, pending_lead = _split_ocr_markdown(
            text, page, section, carried_scope, pending_lead
        )
        out.extend(segments)
    if pending_lead:
        last = pages[-1]
        out.append(
            Segment(
                pending_lead,
                last.page,
                TEXT,
                sections_by_page.get(last.page) if last.page else None,
            )
        )
    return out


def _restore_cover_fields(
    text_segments: list[Segment], page: int | None, ocr_text: str
) -> str:
    """OCR 漏掉的封面字段行插回该页正文开头。已经写进 OCR 的不重复。"""
    if page is None:
        return ocr_text
    ocr_compact = re.sub(r"\s+", "", ocr_text)
    missing: list[str] = []
    seen: set[str] = set()
    for seg in text_segments:
        if seg.page != page or seg.kind != TEXT:
            continue
        for raw in seg.text.splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if not line or len(line) > 120 or not _COVER_FIELD.search(line):
                continue
            compact = re.sub(r"\s+", "", line)
            if compact in ocr_compact or compact in seen:
                continue
            seen.add(compact)
            missing.append(line)
    if not missing:
        return ocr_text
    return "\n".join(missing + [ocr_text.strip()]).strip()


def _split_ocr_markdown(
    text: str,
    page: int | None,
    section: str | None,
    carried_scope: str | None = None,
    pending_lead: str = "",
) -> tuple[list[Segment], str | None, str]:
    """一页 OCR markdown：含 HTML 表的块成表格，其余成正文。

    返回 (段落, 传到下一页的口径, 表名)。表名不一定贴着表格：中间可以有
    勾选行、提问句和空行，也可以落在上一页末尾，跟下一页的表拼在一起。
    """
    body = _strip_ocr_page_marks(text)
    if not body:
        return [], carried_scope, pending_lead
    out: list[Segment] = []
    carried = carried_scope
    if not _OCR_TABLE.search(body):
        merged = "\n".join(part for part in (pending_lead, body) if part)
        fresh = _statement_scope_in(merged)
        if fresh:
            carried = fresh
        return [Segment(merged, page, TEXT, section)], carried, ""
    last = 0
    lead = pending_lead
    for match in _OCR_TABLE.finditer(body):
        preamble = body[last : match.start()]
        fresh = _statement_scope_in(preamble)
        if fresh:
            carried = fresh
        before, caption = _split_table_preamble(preamble)
        if lead:
            if before:
                out.append(Segment(lead, page, TEXT, section))
            else:
                caption = "\n".join(part for part in (lead, caption) if part)
                fresh = _statement_scope_in(lead)
                if fresh:
                    carried = fresh
            lead = ""
        if before:
            out.append(Segment(before, page, TEXT, section))
        table = match.group(0).strip()
        if caption:
            table = f"{caption}\n{table}"
        out.append(Segment(table, page, TABLE, section, scope=carried))
        last = match.end()
    prose, trailing = _split_table_preamble(body[last:])
    if prose:
        out.append(Segment(prose, page, TEXT, section))
        fresh = _statement_scope_in(prose)
        if fresh:
            carried = fresh
    if trailing:
        fresh = _statement_scope_in(trailing)
        if fresh:
            carried = fresh
    return out, carried, trailing


def _statement_scope_in(text: str) -> str | None:
    """一段文字里最后一条报表名的口径。句子里的「企业合并」不算。"""
    from ai_agent.kb.ingest.tables import _table_scope

    found: str | None = None
    for raw in text.splitlines():
        line = _clean_caption_line(raw)
        if not line or len(line) > 40 or line[-1] in "。；，、：:":
            continue
        if not line.endswith("表"):
            continue
        scope = _table_scope(line)
        if scope:
            found = scope
    return found


def _clean_caption_line(raw: str) -> str:
    line = raw.strip().strip("*").strip()
    line = re.sub(r"^#{1,6}\s*", "", line)
    line = re.sub(r"^\d{1,2}\s*[、.．]\s*", "", line)
    line = re.sub(r"^（[一二三四五六七八九十]+）\s*", "", line)
    return line.strip()


def _strip_ocr_page_marks(text: str) -> str:
    kept = [
        line for line in (text or "").splitlines() if not _OCR_PAGE_MARK.match(line.strip())
    ]
    return "\n".join(kept).strip()


def _split_table_preamble(preamble: str) -> tuple[str, str]:
    """表前的标题、引导句、勾选和单位行归这张表，再往上的正文留下。

    空行不截断：标题和表格之间常常空一行，或者夹着
    「公司是否需追溯调整…」「□ 是 √ 否」，以及「……如下：」。
    碰到正文句子才停。
    """
    from ai_agent.kb.ingest.tables import _CAPTION_NOISE, _CAPTION_SHAPE

    lines = preamble.splitlines()
    start = len(lines)
    anchor_noise = False
    blanks = 0
    while start > 0:
        raw = lines[start - 1]
        if not raw.strip():
            blanks += 1
            start -= 1
            continue
        if _is_strong_lead(raw, _CAPTION_NOISE, _CAPTION_SHAPE):
            anchor_noise = _is_noise_line(raw, _CAPTION_NOISE)
            blanks = 0
            start -= 1
            continue
        # 勾选、单位上方那句短提问，隔一行也算表头。再往上的正文不带。
        if _is_weak_lead(raw) and anchor_noise and blanks <= 1:
            anchor_noise = False
            blanks = 0
            start -= 1
            continue
        break
    return "\n".join(lines[:start]).strip(), "\n".join(lines[start:]).strip()


def _is_noise_line(raw: str, noise: re.Pattern[str]) -> bool:
    line = raw.strip()
    return bool(noise.search(line)) and not re.match(r"^#{1,6}\s*\S", line)


def _is_strong_lead(raw: str, noise: re.Pattern[str], shape: re.Pattern[str]) -> bool:
    line = raw.strip()
    if _TABLE_INTRO.search(line) and len(line) <= _TABLE_INTRO_MAX:
        return True
    if re.match(r"^#{1,6}\s*\S", line) and len(line) <= 80:
        return True
    if noise.search(line):
        return True
    cleaned = _clean_caption_line(line)
    if not cleaned or len(cleaned) > 40 or cleaned[-1] in "。；，、：:":
        return False
    return cleaned.endswith("表") or bool(shape.search(cleaned))


def _is_weak_lead(raw: str) -> bool:
    line = raw.strip()
    if _COVER_FIELD.search(line):
        return False
    if len(line) > 40 or line[-1] in "。；，、：:":
        return False
    return True


def _pdf_text(
    path: Path, *, include_tables: bool = True
) -> tuple[list[Segment], list[int]]:
    """行级抽取：剔页眉页脚，按一级节分组后逐页产出 Segment；表格单独成段。

    返回 ([], []) 表示 pymupdf 不可用或文件打不开，由调用方降级——不用
    异常是因为「降级」是正常路径（未装 pymupdf 的部署仍要能解析）。

    这里只做「节 → 页」的拆分，节内跨页聚合留给 chunking：切片需要页码
    溯源，Segment 保持页粒度就不会丢页码。

    表格与正文混排时按 (页, 纵向位置) 排回原顺序——表格的位置决定它
    在哪一节、挨着哪段正文，统统挪到文末会让引用张冠李戴。

    ``include_tables=False``：只抽正文行（给 OCR 路径取章节用）。find_tables
    在部分 PDF 上会原生崩溃，OCR 开着时没必要先踩这个雷。
    """
    # 表格函数已拆到 tables.py，延迟 import 避免循环依赖
    from ai_agent.kb.ingest.tables import (
        _attach_captions,
        _drop_table_lines,
        _merge_continued_tables,
        _pdf_glossary_tables,
        _pdf_tables_isolated,
        _table_meta_above,
        _table_prefix,
        _table_scope,
    )

    try:
        import pymupdf
    except ImportError:
        logger.warning("未安装 pymupdf，PDF 文本退回整页抽取（无章节识别）")
        return [], []
    try:
        doc = pymupdf.open(str(path))
    except Exception:
        logger.warning("pymupdf 打开 PDF 失败，退回整页抽取", exc_info=True)
        return [], []
    try:
        lines = sections.extract_lines(doc)
    except Exception:
        logger.warning("pymupdf 抽取文本行失败，退回整页抽取", exc_info=True)
        return [], []
    # 勾选框符号归一化要在任何按文本的比较（页眉去重、标题识别、表格剔除）
    # 之前做，否则同一符号在两处长得不一样，去重去不干净
    lines = normalize_choice_lines(lines)
    # 表格识别只是「多产出几段」，失败不该拖累正文解析。
    # find_tables 的 SIGSEGV 拦不住，放到短命子进程里按页推进。
    tables: list[tuple[int, float, tuple, list[list[str]]]] = []
    page_heights: dict[int, float] = {}
    if include_tables and settings.kb_pdf_table_extraction:
        try:
            tables, page_heights = _pdf_tables_isolated(path)
        except Exception:
            logger.warning("PDF 表格识别失败，本次只出正文切片", exc_info=True)
        try:
            # 无框释义表（"X 指 Y"）find_tables 按线条识别不到，单独补一轮；
            # 传已有表格框做排除，页上残留框线的部分不会被出两份
            tables += _pdf_glossary_tables(
                doc, [(page, bbox) for page, _y, bbox, _rows in tables]
            )
        except Exception:
            logger.warning("PDF 释义表识别失败，跳过", exc_info=True)
    doc.close()
    if not lines:
        return [], []

    # 图表页判定用的是原始文字量（含页眉、含表格），与历史行为保持一致
    text_lens = _page_char_counts(lines)
    # 表格里的数字字号常小于正文且字符量极大，算正文字号（按量加权取众数）
    # 会把它拉偏，导致真标题被判成页眉——所以先剔掉表格行再定字号
    rest = _drop_table_lines(lines, tables)
    body = sections.body_size(rest)
    kept, stats = sections.strip_noise(rest, body)
    if tables:
        stats["表格"] = len(tables)
    if stats:
        logger.info("PDF 噪声剔除与表格 %s", stats)

    if not settings.kb_section_chunking:
        groups = [sections.Section("", "", start_page=kept[0].page if kept else 1, lines=list(kept))]
    else:
        groups = sections.group_by_section(kept, body)
        kept = [ln for sec in groups for ln in sec.lines]
    # 表格不在正文行里，只能靠同页的标题位置推断它属于哪一小节：
    # 表题/表头之下、下一个标题之上的表，才算这一节的
    markers = _section_markers(groups)

    # (页, 纵向位置, 段)：文本段与表格段按原版面顺序归并
    items: list[tuple[int, float, Segment]] = []
    for y, seg in _page_segments(groups):
        items.append((seg.page or 0, y, seg))
    # 表题接在表格正文前面（一行）：切成多片时每片都会带上它，
    # 「母公司利润表 + 营业收入 + 数值」因此能落在同一个切片里
    for page, y0, _bbox, rows, caption, page_end in _merge_continued_tables(
        _attach_captions(tables, kept), page_heights
    ):
        markdown = _rows_to_markdown(rows)
        if not markdown:
            continue
        # 单位/审计状态跟表题一起写进正文：长表切开后每一片都还知道
        # 「数字是百万元、这张表未经审计」，否则后半段只剩格子
        prefix = _table_prefix(caption, _table_meta_above(kept, page, y0))
        text = f"{prefix}\n{markdown}" if prefix else markdown
        items.append(
            (
                page,
                y0,
                Segment(
                    text,
                    page,
                    TABLE,
                    _label_at(markers, page, y0),
                    scope=_table_scope(caption),
                    page_end=page_end if page_end != page else None,
                ),
            )
        )
    items.sort(key=lambda it: (it[0], it[1]))
    return [seg for _page, _y, seg in items], text_lens


def _section_markers(
    groups: list[sections.Section],
) -> list[tuple[int, float, str]]:
    """(页, 纵坐标, 章节标签)：每个小节首行的位置就是这一节的起点。

    行内顺序即版面顺序，所以标记天然升序，查表走二分即可。用「点」坐标
    而不是 y_ratio，是因为表格的 y0 就是 pymupdf 的页面坐标，两者要能直接比。
    """
    out: list[tuple[int, float, str]] = []
    for sec in groups:
        label = sec.label
        if not label or not sec.lines:
            continue
        first = sec.lines[0]
        y = first.bbox[1] if first.bbox else first.y_ratio * 1000
        out.append((first.page, y, label))
    return out


def _label_at(
    markers: list[tuple[int, float, str]], page: int, y: float
) -> str | None:
    """查某个位置所属的章节：取它上方最近的那个标题，跨页继续沿用。"""
    best: str | None = None
    for m_page, m_y, label in markers:
        if m_page > page or (m_page == page and m_y > y + 1):
            break
        best = label
    return best


def _page_segments(
    groups: list[sections.Section],
) -> list[tuple[float, Segment]]:
    """按 (节, 页) 聚成 Segment，同时带回该段首行的纵坐标（与表格排序混排用）。

    一格小节一页一段：同一页里「（一）主要会计数据」和「（二）非经常性
    损益」各成一段，切片因此不会跨小节拼接，每片都能标到具体小节。
    """
    out: list[tuple[float, Segment]] = []
    for sec in groups:
        label = sec.label or None
        for page, page_lines in groupby(sec.lines, key=lambda ln: ln.page):
            group = list(page_lines)
            # 版面折行接回段落：PDF 每行一个 line，直接按行拼接会把
            # 「占总营」「收94.84%」切成两段，切片边界和引用都断在词中间
            paras = _merge_wrapped(group)
            # 勾选行正好落在小节开头时没有前置问题可接（「（二）非经常性
            # 损益项目和金额」自己成了新小节的首行），把小节标题当问题补上，
            # 否则模型读到的是一句没有提问对象的「☑（已勾选）适用」
            if label and paras and _is_choice_line(paras[0]):
                paras[0] = _attach_choice(label.split(" · ")[-1], paras[0])
            text = "\n".join(paras).strip()
            if not text:
                continue
            first = group[0]
            y = first.bbox[1] if first.bbox else first.y_ratio * 1000
            out.append((y, Segment(text, page, TEXT, label)))
    return out


def _segments_by_page(
    lines: list[sections.Line], *, section: str | None = None
) -> list[Segment]:
    """连续的同类行按页聚成 Segment（行序即页序，可直接 groupby）。"""
    out: list[Segment] = []
    for page, page_lines in groupby(lines, key=lambda ln: ln.page):
        text = "\n".join(ln.text for ln in page_lines).strip()
        if text:
            out.append(Segment(text, page, TEXT, section))
    return out


# 勾选框（单选/多选 selections）。PDF 里的方框和对勾常取自 Wingdings 一类的
# 符号字体：抽出来是**私用区字符**（实测 \uf052 = 已勾、\uf0a3 = 空框），或者
# 是写法各异的符号（√ ✓ ■ □ ☐ ○）。后果是选择题全部失读——模型看到的
# 「\uf052适用□不适用」跟噪声没区别，□ 和私用区字符它分不出哪个代表已勾，
# 于是「单选题选了哪项」从来没被正确读出来（季报里这类问法有十几处：
# 是否需追溯调整、是否经过审计、是否存在xxx情况）。
# 做法：先统一成标准 Unicode，再把状态写成中文——只换成 ☑/☐ 还不够，这两个
# 符号在别的语料里也被混用作纯装饰，写死中文标签最稳。
_CHECKED_CHARS = "\uf052✓✔√■●"
_UNCHECKED_CHARS = "\uf0a3□☐○"
_MARK_CHECKED = "☑"
_MARK_UNCHECKED = "☐"
_HAS_MARK = re.compile(f"[{re.escape(_CHECKED_CHARS + _UNCHECKED_CHARS)}]")

# 勾选行：整行只有若干「符号+选项」，没有别的（「☑（已勾选）适用 ☐（未勾选）
# 不适用」「□是 ☑否」）。判据必须要求「以勾选符开头」——只看长度的话，
# 「公司是否需追溯调整…会计数据：☐（未勾选）是 ☑（已勾选）否」这种已经带上
# 提问的行有 38 个字，会被误判成纯选项行，再被补一个小节标题当提问主体。
_MAX_CHOICE_LEN = 60


def normalize_choices(text: str) -> str:
    """把勾选符号归一化并标注选中状态。

    「\\uf052适用□不适用」→「☑（已勾选）适用 ☐（未勾选）不适用」。
    没有勾选符号的文本原样返回（不加任何标记），绝大多数行的开销为零。
    """
    if not text or not _HAS_MARK.search(text):
        return text
    table = {ord(c): _MARK_CHECKED for c in _CHECKED_CHARS}
    table.update({ord(c): _MARK_UNCHECKED for c in _UNCHECKED_CHARS})
    out = text.translate(table)
    out = out.replace(_MARK_CHECKED, f"{_MARK_CHECKED}（已勾选）")
    out = out.replace(_MARK_UNCHECKED, f"{_MARK_UNCHECKED}（未勾选）")
    # 原文本里两个选项紧挨着（「□适用√不适用」），留个空格免得读成一串
    return re.sub(r"(?<=\S)([☑☐])", r" \1", out)
    return out


def normalize_choice_lines(
    lines: list[sections.Line],
) -> list[sections.Line]:
    """逐行做勾选归一化。Line 是 frozen dataclass，只能整体 replace。"""
    out: list[sections.Line] = []
    for ln in lines:
        new = normalize_choices(ln.text)
        out.append(replace(ln, text=new) if new != ln.text else ln)
    return out


def _is_choice_line(text: str) -> bool:
    """是否是「整行只有选项、没有提问对象」的勾选行。"""
    if not text.startswith((_MARK_CHECKED, _MARK_UNCHECKED)):
        return False
    marks = sum(1 for ch in text if ch in (_MARK_CHECKED, _MARK_UNCHECKED))
    return (
        bool(marks) and marks <= 6 and len(text) <= _MAX_CHOICE_LEN and not _SENT_END.search(text)
    )


def _attach_choice(head: str, choice: str) -> str:
    """把勾选结果接到它的问题后面。

    季报里这些行一律紧跟提问（「持股5%以上股东…出借股份情况」+「☑不适用」），
    合并后模型才知道「不适用」回答的是什么。几何坐标不能当判据：同一问法的
    行有的排到了右边界被当前的折行规则接上，有的没排满就成了两行，结果同一
    份文档里一半合并一半分离，反而更难开除统一处。
    """
    if not head:
        return choice
    if head.endswith(("：", ":", "（", "(", "、", "，")) or _SENT_END.search(head):
        return head + choice
    return f"{head}：{choice}"


# 版面折行：上一行排到右边界、下一行从左边距起，且上一行没有句末标点，
# 那就是同一个自然段被排版折开。不接回去的代价很直观——安克的
# 「占总营」「收94.84%」、「持」「续加大研发」会被切成两条段落，切片边界
# 落在词中间，检索命中后引用出来是半句话，图谱也从半句里抽实体。
_SENT_END = re.compile(r"[。！？；：!?;:…」』”\"）)》】]$")
# 换块信号：编号标题、表题、注释、单位行。上一行即使没有句末标点，
# 后面跟着这些也不该接（「……具体情况如下」+「（1）充电储能类产品」）
_BLOCK_START = re.compile(
    r"^(第[一二三四五六七八九十百]+[节章]|[一二三四五六七八九十]+、|"
    r"（[一二三四五六七八九十\d]+）|\(\d+\)|\d+[、.．]|[①-⑳]|"
    r"表\s*\d|附表|单位[:：]|注[:：]|资料来源)"
)
_EDGE_SLACK = 6.0  # 点：判「贴左右边距」的容差，中文排版两端对齐会差几个点


def _merge_wrapped(lines: list[sections.Line]) -> list[str]:
    """把版面折行的行接回自然段，返回段落文本列表。

    只认同页内的折行：跨页的那一行是另一个版面，接起来会把页眉残留、
    页脚页码掺进正文（它们的坐标早就被剔除了，但相邻行仍在）。
    """
    boxes = [ln.bbox for ln in lines if ln.bbox]
    plain = [ln.text.strip() for ln in lines if ln.text.strip()]
    if len(boxes) < 2:
        return plain
    left = min(b[0] for b in boxes)
    right = max(b[2] for b in boxes)
    slack = max((right - left) * 0.05, _EDGE_SLACK)

    paras: list[str] = []
    prev: sections.Line | None = None
    for ln in lines:
        text = ln.text.strip()
        if not text:
            continue
        if prev is not None and paras:
            if _is_choice_line(text):
                # 勾选行一律并入上一行，不靠几何判据（见 _attach_choice）
                paras[-1] = _attach_choice(paras[-1], text)
            elif _wraps(prev, ln, left, right, slack):
                paras[-1] += text
            else:
                paras.append(text)
        else:
            paras.append(text)
        prev = ln
    return paras


def _wraps(
    prev: sections.Line,
    cur: sections.Line,
    left: float,
    right: float,
    slack: float,
) -> bool:
    if not prev.bbox or not cur.bbox:
        return False
    if prev.bbox[2] < right - slack:
        return False  # 上一行没排满 → 本来就是段末
    if abs(cur.bbox[0] - left) > slack:
        return False  # 下一行缩进/另有栏位 → 不是同一段
    # 上一行也必须从左边距起（允许 2 字缩进）：封面上的「证券代码：…」
    # 与右侧「公告编号：…」排在同一行，前者贴左、后者贴右，不看 x0
    # 会把上下两栏的文字接成一句
    if prev.bbox[0] > left + slack * 4:
        return False
    if _SENT_END.search(prev.text.strip()):
        return False
    if len(prev.norm) <= 40 and _BLOCK_START.match(prev.norm):
        return False  # 上一行本身是标题，别把标题并进正文
    return not _BLOCK_START.match(cur.norm)



def _page_char_counts(lines: list[sections.Line]) -> list[int]:
    """每页原始字符数（含页眉）：图表页的三道筛按整页文字量判定。"""
    counts: dict[int, int] = {}
    for ln in lines:
        counts[ln.page] = counts.get(ln.page, 0) + len(ln.text)
    if not counts:
        return []
    return [counts.get(page, 0) for page in range(1, max(counts) + 1)]


def _pdf_text_fallback(path: Path) -> tuple[list[Segment], list[int]]:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ParseError("服务端缺少 pypdf 依赖，请联系管理员") from e
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise ParseError(f"PDF 文件无法打开（{type(e).__name__}）") from e

    segments: list[Segment] = []
    text_lens: list[int] = []
    for i, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        text_lens.append(len(text))
        if text:
            segments.append(Segment(normalize_choices(text), i + 1, TEXT))
    return segments, text_lens



def _rows_to_markdown(rows: list[list[str]]) -> str:
    """二维单元格 → Markdown 表格。PDF/DOCX/Excel 三路共用同一份输出格式。"""
    if not rows:
        return ""
    # 延迟导入：table_cells 顶部依赖本模块的 _escape_cell / normalize_choices
    from ai_agent.kb.ingest.table_cells import _ensure_row_header

    rows = _ensure_row_header(rows)
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)



def _escape_cell(text: str) -> str:
    # 单元格里的竖线/换行会破坏 Markdown 表格结构
    return " ".join(text.split()).replace("|", "\\|")
