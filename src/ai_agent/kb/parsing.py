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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from itertools import groupby
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb import sections, vision

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
        return _parse_docx(path)
    if suffix in {".xlsx", ".xlsm"}:
        return _parse_xlsx(path)
    if suffix in _SUPPORTED:
        return [
            Segment(path.read_text(encoding="utf-8", errors="replace"))
        ]
    raise ParseError(f"暂不支持的文件类型 {suffix}")


# ——— PDF ———


def _parse_pdf(path: Path) -> list[Segment]:
    segments, text_lens = _pdf_text(path)
    if not segments:
        # pymupdf 不可用 / 无文本层：退回整页抽取（无章节、无页眉剔除），
        # 扫描件走这条路时仍是「抽不出字就报错」，不做假解析
        segments, text_lens = _pdf_text_fallback(path)

    # 图表单独走渲染 + 视觉识别（pypdf 不能把页转成图）
    segments.extend(_pdf_figures(path, text_lens))

    if not segments:
        raise ParseError("PDF 中没有可提取的文本（可能是扫描件/图片版）")
    return segments


def _pdf_text(path: Path) -> tuple[list[Segment], list[int]]:
    """行级抽取：剔页眉页脚，按一级节分组后逐页产出 Segment；表格单独成段。

    返回 ([], []) 表示 pymupdf 不可用或文件打不开，由调用方降级——不用
    异常是因为「降级」是正常路径（未装 pymupdf 的部署仍要能解析）。

    这里只做「节 → 页」的拆分，节内跨页聚合留给 chunking：切片需要页码
    溯源，Segment 保持页粒度就不会丢页码。

    表格与正文混排时按 (页, 纵向位置) 排回原顺序——表格的位置决定它
    在哪一节、挨着哪段正文，统统挪到文末会让引用张冠李戴。
    """
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
    # 表格识别只是「多产出几段」，失败不该拖累正文解析
    try:
        tables = (
            _pdf_tables(doc)
            if settings.kb_pdf_table_extraction
            else []
        )
    except Exception:
        logger.warning("PDF 表格识别失败，本次只出正文切片", exc_info=True)
        tables = []
    finally:
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
        page_section: dict[int, str | None] = {}
    else:
        groups = sections.group_by_section(kept, body)
        page_section = _page_section_map(groups)
        kept_by_sec: list[sections.Line] = []
        for sec in groups:
            kept_by_sec.extend(sec.lines)
        kept = kept_by_sec

    # (页, 纵向位置, 段)：文本段与表格段按原版面顺序归并
    items: list[tuple[int, float, Segment]] = []
    for y, seg in _page_segments(kept, page_section):
        items.append((seg.page or 0, y, seg))
    for page, y0, _bbox, rows in tables:
        markdown = _rows_to_markdown(rows)
        if not markdown:
            continue
        items.append(
            (page, y0, Segment(markdown, page, TABLE, page_section.get(page)))
        )
    items.sort(key=lambda it: (it[0], it[1]))
    return [seg for _page, _y, seg in items], text_lens


def _page_section_map(groups: list[sections.Section]) -> dict[int, str | None]:
    """页 → 所属一级节：表格不在正文行里，只能靠同页的正文行推断归属。"""
    mapping: dict[int, str | None] = {}
    for sec in groups:
        label = sec.path if sec.id else None
        for ln in sec.lines:
            mapping.setdefault(ln.page, label)
    return mapping


def _page_segments(
    lines: list[sections.Line], page_section: dict[int, str | None]
) -> list[tuple[float, Segment]]:
    """按页聚成 Segment，同时带回该页首行的纵坐标（与表格排序混排用）。"""
    out: list[tuple[float, Segment]] = []
    for page, page_lines in groupby(lines, key=lambda ln: ln.page):
        group = list(page_lines)
        text = "\n".join(ln.text for ln in group).strip()
        if not text:
            continue
        first = group[0]
        y = first.bbox[1] if first.bbox else first.y_ratio * 1000
        out.append((y, Segment(text, page, TEXT, page_section.get(page))))
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


# ——— PDF 表格 ———
# PDF 里没有「表格」这个对象，只有画出来的线和摆好的字。pymupdf 的
# find_tables 靠线条（无框表格靠文字对齐）反推单元格，抽出来的是二维
# 字符串数组，转成 Markdown 表格交给切片层（整表成片、超长重复表头）。

_CELL_WS = re.compile(r"\s+")
# 单元格内的换行是排版断行：数字常被拆成 "5176 63\n."（小数点另起一行）
_NUM_DECIMAL = re.compile(r"(\d[\d,]*)\s+(\d+)\s*\.\s*(\d*)")
# 千分位也逃不过断行："114 376 427 06 , , ." 是 114,376,427.06
# （PDF 把逗号和小数点排到单元格末尾，前面只剩一串数字碎片）
_NUM_THOUSANDS = re.compile(r"^(\d+(?:\s+\d+)+)\s*(?:[,]\s*)*\.\s*(\d*)$")
# "2006年 1 6月 -" 是 2006年1-6月：连字符被甩到单元格最后
_NUM_RANGE = re.compile(r"(\d+)\s+(\d+月)\s*-\s*$")
# 中文字之间的空格也是排版字距（"木粉 复合油 ，"），不是真空格
_CJK = "\u4e00-\u9fff\u3040-\u30ff"
_CJK_SPACE = re.compile(f"([{_CJK}])\\s+([{_CJK}])")
# 标点被单独排版时前后会多出空格："复合油 ，" / "（ 注"
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([，。、；：？！）】》」』%])")
_SPACE_AFTER_PUNCT = re.compile(r"([（【《「『])\s+")
# 表头续行（"元）"、"（%"）都很短，超过这个长度基本是数据行不是续行
_MAX_HEADER_CELL = 6
# 纯数字（允许千分位、百分比、负号）：表头碎片不会是这个形状
_NUM_ONLY = re.compile(r"^[\d.,%+\-]+$")


def _pdf_tables(doc) -> list[tuple[int, float, tuple, list[list[str]]]]:
    """逐页找表格，返回 (页码, 顶部纵坐标, bbox, 清洗后的二维行)。"""
    out: list[tuple[int, float, tuple, list[list[str]]]] = []
    for pno in range(doc.page_count):
        try:
            found = doc[pno].find_tables()
        except Exception:
            logger.warning("PDF 第 %d 页表格识别失败，跳过", pno + 1, exc_info=True)
            continue
        for table in found.tables:
            rows = _clean_table(table.extract())
            if len(rows) < 2 or max(len(r) for r in rows) < 2:
                continue
            bbox = tuple(table.bbox)
            out.append((pno + 1, float(bbox[1]), bbox, rows))
    return out


def _clean_table(raw: list[list[str | None]]) -> list[list[str]]:
    """单元格清洗 → 跨行表头合并 → 删掉合并单元格留下的空列。"""
    rows = [[_clean_cell(cell) for cell in row] for row in raw]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    rows = _merge_header(rows)
    rows = _align_header(rows)
    return _drop_empty_cols(rows)


def _clean_cell(value: str | None) -> str:
    text = (value or "").replace("\n", " ")
    text = _CELL_WS.sub(" ", text).strip()
    m = _NUM_THOUSANDS.match(text)
    if m:
        groups = m.group(1).split()
        text = ",".join(groups[:-1]) + "." + groups[-1] + m.group(2)
    else:
        text = _NUM_DECIMAL.sub(
            lambda m: f"{m.group(1)}.{m.group(2)}{m.group(3)}", text
        )
        text = _NUM_RANGE.sub(r"\1-\2", text)
    text = _CJK_SPACE.sub(r"\1\2", text)
    # 数字与中文之间也是排版间距："2005年12月31 日" → "2005年12月31日"
    text = re.sub(rf"([{_CJK}])\s+(\d)", r"\1\2", text)
    text = re.sub(rf"(\d)\s+([{_CJK}])", r"\1\2", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_PUNCT.sub(r"\1", text)
    return _escape_cell(text)


def _align_header(rows: list[list[str]]) -> list[list[str]]:
    """表头归位：合并单元格常把表头挤到右边一列，和数据对不上。

    「销量（吨）」的格子横跨两列时，网格切出来的表头落在 col1，而数值
    落在 col1/col2 —— 直接渲染的话每列都串了位。这里按「有数据的列」
    依次安放表头文字：数量对得上才动，对不上说明不是这种表，不猜。
    """
    if len(rows) < 2:
        return rows
    header, body = rows[0], rows[1:]
    data_cols = [i for i in range(len(header)) if any(row[i] for row in body)]
    fills = [cell for cell in header if cell]
    if not fills or len(fills) != len(data_cols):
        return rows
    aligned = [""] * len(header)
    for col, cell in zip(data_cols, fills):
        aligned[col] = cell
    return [aligned] + body


def _merge_header(rows: list[list[str]]) -> list[list[str]]:
    """表头占了多行时拼回一行，最多吃三行（表头再复杂就认命）。

    两种形态，判据是「第二行的非空列和第一行重不重叠」：

    - **续行**：第二行的列是第一行的子集，且都是短碎片——"销售收入（万"
      + "元）"、"占总收入比例" + "（%"）。长内容说明那是数据行，不动；
    - **分散**：两行的非空列完全不重叠——招股书常见的斜线表头把「年份」
      放第 0 列、「2006年1-6月」放第 1~4 列，分了两三行才写完。

    部分重叠（既像续行又像数据）说不清，一律不动——宁可留着多行表头，
    也不能把两行数据粘成一行。
    """
    for _ in range(3):
        if len(rows) < 2:
            break
        first, second = rows[0], rows[1]
        f_cols = {i for i, cell in enumerate(first) if cell}
        s_cols = {i for i, cell in enumerate(second) if cell}
        if not f_cols or not s_cols:
            break
        if s_cols <= f_cols:
            cells = [second[i] for i in s_cols]
            if any(len(cell) > _MAX_HEADER_CELL for cell in cells):
                break
            # 纯数字的一律是数据行（"100"、"200"），拼进表头就串味了
            if any(_NUM_ONLY.match(cell) for cell in cells):
                break
        elif s_cols & f_cols:
            break
        merged = list(first)
        for i, cell in enumerate(second):
            if not cell:
                continue
            # 第一行该列为空时直接落下来，别把表头碎片丢了
            merged[i] = merged[i] + cell if merged[i] else cell
        rows = [merged] + rows[2:]
    return rows


def _drop_empty_cols(rows: list[list[str]]) -> list[list[str]]:
    """删掉整列为空的列：跨列合并的单元格会留下一串空列。"""
    width = max(len(row) for row in rows)
    keep = [i for i in range(width) if any(row[i] for row in rows)]
    return [[row[i] for i in keep] for row in rows]


def _drop_table_lines(
    lines: list[sections.Line],
    tables: list[tuple[int, float, tuple, list[list[str]]]],
) -> list[sections.Line]:
    """剔掉落在表格框内的文本行：表格已单独成段，正文里不该再有一份。"""
    if not tables:
        return lines
    boxes = [(page, bbox) for page, _y, bbox, _rows in tables]
    kept: list[sections.Line] = []
    for ln in lines:
        if ln.bbox and _hit_any_box(ln, boxes):
            continue
        kept.append(ln)
    return kept


def _hit_any_box(
    ln: sections.Line, boxes: list[tuple[int, tuple[float, float, float, float]]]
) -> bool:
    """行框与表格框的重叠面积占行面积一半以上，就算这行属于表格。"""
    x0, y0, x1, y1 = ln.bbox
    area = max((x1 - x0) * (y1 - y0), 1.0)
    for page, (bx0, by0, bx1, by1) in boxes:
        if page != ln.page:
            continue
        ox = max(0.0, min(x1, bx1) - max(x0, bx0))
        oy = max(0.0, min(y1, by1) - max(y0, by0))
        if ox * oy / area >= 0.5:
            return True
    return False


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
            segments.append(Segment(text, i + 1, TEXT))
    return segments, text_lens


def _describe_parallel(jobs: list[tuple[bytes, str]]) -> list[str | None]:
    """并行送视觉模型，结果顺序与 jobs 一致。

    慢的是视觉调用（每次数秒），页面渲染/图片提取是本地快的——所以先把
     material 准备好，再按 kb_vision_concurrency 分批并行调用。
    熔断后 describe_image 直接返回 None，剩余任务秒过，不会白等。
    """
    workers = max(1, min(settings.kb_vision_concurrency, len(jobs)))
    if workers <= 1:
        return [vision.describe_image(blob, mime) for blob, mime in jobs]
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="kb-vision"
    ) as pool:
        return list(
            pool.map(lambda job: vision.describe_image(job[0], job[1]), jobs)
        )


def _pdf_figures(path: Path, text_lens: list[int] | None = None) -> list[Segment]:
    """含图页渲染成 PNG 交视觉模型描述；关闭或失败都只是少几段，不阻断解析。"""
    if not vision.available():
        return []
    try:
        import pymupdf
    except ImportError:
        logger.warning("未安装 pymupdf，跳过 PDF 图表识别")
        return []

    out: list[Segment] = []
    try:
        doc = pymupdf.open(str(path))
    except Exception:
        logger.warning("pymupdf 打开 PDF 失败，跳过图表识别", exc_info=True)
        return []
    # 先扫出预算内的含图页（不渲染），再分批「渲染 → 并行识别」；
    # 熔断检查放在批间：熔断后后续页不再渲染（与串行版语义一致）
    try:
        targets: list[int] = []
        for i, page in enumerate(doc):
            text_len = text_lens[i] if text_lens and i < len(text_lens) else None
            if not _page_has_chart(page, text_len):
                continue
            if len(targets) >= settings.kb_vision_max_pages:
                logger.info(
                    "kb 图表识别达到上限 %d 页，其余页跳过",
                    settings.kb_vision_max_pages,
                )
                break
            targets.append(i)
        for start in range(0, len(targets), settings.kb_vision_concurrency):
            if not vision.available():
                break
            pngs: list[tuple[int, bytes]] = []
            for i in targets[start : start + settings.kb_vision_concurrency]:
                try:
                    pngs.append(
                        (
                            i,
                            doc[i]
                            .get_pixmap(dpi=settings.kb_vision_dpi)
                            .tobytes("png"),
                        )
                    )
                except Exception:
                    logger.warning("PDF 第 %d 页渲染失败，跳过", i + 1, exc_info=True)
            if not pngs:
                continue
            descs = _describe_parallel([(png, "image/png") for _i, png in pngs])
            for (i, _png), desc in zip(pngs, descs):
                if desc:
                    out.append(Segment(desc, i + 1, FIGURE))
    finally:
        doc.close()
    return out


def _page_has_chart(page, text_len: int | None = None) -> bool:
    """判断这一页值不值得送视觉模型。

    三道筛子，从便宜到贵：

    1. **文字密度**：正文超过 kb_vision_max_text_chars 的页面直接排除。
       招股书/财报里整页表格会画出几百条矢量线，按「路径数 ≥ 150」判断会
       把表格页统统误认成图表（实测 331 页的招股书误判 124 页，预算 20 页
       全烧在目录和财务报表上）；
    2. **位图面积**：只有占页面面积 ≥ kb_vision_min_image_ratio 的图才算数，
       页眉 logo、页脚装饰线、水印都远小于这条线；
    3. **矢量路径**：ECharts 走 SVG 导出时没有位图，退化成看路径数量，
       但仍要求该页文字很少——纯矢量图表页本来就没多少可抽的文字。
    """
    if text_len is not None and text_len > settings.kb_vision_max_text_chars:
        return False
    try:
        if _has_large_image(page):
            return True
        return len(page.get_drawings()) >= settings.kb_vision_vector_paths
    except Exception:
        return False


def _has_large_image(page) -> bool:
    """页面上是否存在占面积够大的位图（按 bbox 算，不看像素总数）。"""
    try:
        rect = page.rect
        page_area = max(rect.width * rect.height, 1.0)
        for info in page.get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            if (x1 - x0) * (y1 - y0) / page_area >= settings.kb_vision_min_image_ratio:
                return True
    except Exception:
        # get_image_info 拿不到就退回「有没有图」的粗判
        return bool(page.get_images(full=True))
    return False


# ——— DOCX ———


def _parse_docx(path: Path) -> list[Segment]:
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as e:
        raise ParseError("服务端缺少 python-docx 依赖，请联系管理员") from e
    try:
        doc = Document(str(path))
    except Exception as e:
        raise ParseError(f"DOCX 文件无法打开（{type(e).__name__}）") from e

    segments: list[Segment] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            text = "\n\n".join(buffer).strip()
            if text:
                segments.append(Segment(text, None, TEXT))
            buffer.clear()

    # 按文档顺序遍历段落与表格：表格位置决定上下文，不能统统挪到文末
    for item in doc.iter_inner_content():
        if isinstance(item, Paragraph):
            line = item.text.strip()
            if line:
                buffer.append(line)
        elif isinstance(item, Table):
            markdown = _table_to_markdown(item)
            if markdown:
                flush()
                segments.append(Segment(markdown, None, TABLE))
    flush()

    segments.extend(_docx_figures(doc))
    segments.extend(_docx_embedded_sheets(doc))

    if not segments:
        raise ParseError("DOCX 中没有文本内容")
    return segments


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """二维单元格 → Markdown 表格。PDF/DOCX/Excel 三路共用同一份输出格式。"""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _table_to_markdown(table) -> str:
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [_escape_cell(cell.text) for cell in row.cells]
        if any(cells):
            rows.append(cells)
    return _rows_to_markdown(rows)


def _docx_figures(doc) -> list[Segment]:
    """Word 里的图片（含粘贴进来的 ECharts 截图）：交给视觉模型转描述。"""
    if not vision.available():
        return []
    try:
        shapes = doc.inline_shapes
        total = len(shapes)
    except Exception:
        return []
    if total > settings.kb_vision_max_pages:
        logger.info(
            "kb 图表识别达到上限 %d 张，其余跳过", settings.kb_vision_max_pages
        )
        total = settings.kb_vision_max_pages
    # 先把图片二进制全部提出来（本地快），再并行送视觉模型
    jobs: list[tuple[bytes, str]] = []
    for i in range(total):
        try:
            blip = shapes[i]._inline.graphic.graphicData.pic.blipFill.blip
            part = doc.part.related_parts[blip.embed]
        except Exception:
            continue
        jobs.append((part.blob, getattr(part, "content_type", "")))
    descs = _describe_parallel(jobs)
    return [Segment(desc, None, FIGURE) for desc in descs if desc]


def _docx_embedded_sheets(doc) -> list[Segment]:
    """Word 里以对象形式嵌入的 Excel：docx 是 zip，内嵌工作簿就在
    word/embeddings/ 下，直接取二进制当独立表格解析。"""
    out: list[Segment] = []
    try:
        rels = list(doc.part.rels.values())
    except Exception:
        return []
    for rel in rels:
        name = (getattr(rel, "target_ref", "") or "").rsplit("/", 1)[-1].lower()
        if not name.endswith((".xlsx", ".xlsm")):
            continue
        try:
            blob = rel.target_part.blob
        except Exception:
            continue
        try:
            out.extend(_parse_xlsx(blob))
        except ParseError:
            logger.warning("DOCX 内嵌 Excel 解析失败：%s", name, exc_info=True)
    return out


# ——— XLSX ———


def _parse_xlsx(source: bytes | Path) -> list[Segment]:
    """每个工作表转一个 Markdown 表格段，page 记 sheet 序号（展示为「表N」）。"""
    try:
        from openpyxl import load_workbook
    except ImportError as e:
        raise ParseError("服务端缺少 openpyxl 依赖，请联系管理员") from e
    # 路径直接交给 openpyxl（read_only 下流式读，超大表不必整体进内存）
    stream = source if isinstance(source, Path) else BytesIO(source)
    try:
        workbook = load_workbook(stream, data_only=True, read_only=True)
    except Exception as e:
        raise ParseError(f"Excel 文件无法打开（{type(e).__name__}）") from e

    segments: list[Segment] = []
    try:
        for index, sheet in enumerate(workbook.worksheets, start=1):
            markdown = _sheet_to_markdown(sheet)
            if markdown:
                segments.append(
                    Segment(f"工作表：{sheet.title}\n{markdown}", index, TABLE)
                )
    finally:
        workbook.close()
    if not segments:
        raise ParseError("Excel 中没有可提取的数据")
    return segments


def _sheet_to_markdown(sheet) -> str:
    rows: list[list[str]] = []
    for row in sheet.iter_rows(values_only=True):
        cells = [_cell_text(value) for value in row]
        if not any(cells):
            continue
        rows.append(cells)
        if len(rows) >= settings.kb_sheet_max_rows:
            logger.info("工作表 %s 超过 %d 行，截断", sheet.title, len(rows))
            break
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    # 空表头补列名：Markdown 表格没有列名就无法读懂每列是什么
    rows[0] = [cell or f"列{i + 1}" for i, cell in enumerate(rows[0])]
    return _rows_to_markdown(rows)


def _cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _escape_cell(str(value))


def _escape_cell(text: str) -> str:
    # 单元格里的竖线/换行会破坏 Markdown 表格结构
    return " ".join(text.split()).replace("|", "\\|")
