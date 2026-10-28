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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb import vision

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
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ParseError("服务端缺少 pypdf 依赖，请联系管理员") from e
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise ParseError(f"PDF 文件无法打开（{type(e).__name__}）") from e

    segments: list[Segment] = []
    # 逐页文字长度留下来给图表页判定用：正文密集的页面不可能是整页图表，
    # 渲染它纯属浪费视觉预算（这里已经抽过一遍，不必再花一次）
    text_lens: list[int] = []
    for i, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        text_lens.append(len(text))
        if text:
            segments.append(Segment(text, i + 1, TEXT))

    # 图表单独走渲染 + 视觉识别（pypdf 不能把页转成图）
    segments.extend(_pdf_figures(path, text_lens))

    if not segments:
        raise ParseError("PDF 中没有可提取的文本（可能是扫描件/图片版）")
    return segments


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


def _table_to_markdown(table) -> str:
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [_escape_cell(cell.text) for cell in row.cells]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


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
    header = [cell or f"列{i + 1}" for i, cell in enumerate(rows[0])]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _escape_cell(str(value))


def _escape_cell(text: str) -> str:
    # 单元格里的竖线/换行会破坏 Markdown 表格结构
    return " ".join(text.split()).replace("|", "\\|")
