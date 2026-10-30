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
from ai_agent.kb import sections

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
        from ai_agent.kb.office import _parse_docx

        return _parse_docx(path)
    if suffix in {".xlsx", ".xlsm"}:
        from ai_agent.kb.office import _parse_xlsx

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
    # 延迟 import 避免与 figures 模块的循环依赖
    from ai_agent.kb.figures import _pdf_figures

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
    # 表格函数已拆到 tables.py，延迟 import 避免循环依赖
    from ai_agent.kb.tables import (
        _attach_captions,
        _drop_table_lines,
        _merge_continued_tables,
        _pdf_glossary_tables,
        _pdf_tables,
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
    # 表格识别只是「多产出几段」，失败不该拖累正文解析
    tables: list[tuple[int, float, tuple, list[list[str]]]] = []
    page_heights: dict[int, float] = {}
    if settings.kb_pdf_table_extraction:
        try:
            tables, page_heights = _pdf_tables(doc)
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
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)



def _escape_cell(text: str) -> str:
    # 单元格里的竖线/换行会破坏 Markdown 表格结构
    return " ".join(text.split()).replace("|", "\\|")
