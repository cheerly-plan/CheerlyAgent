"""PDF 章节识别与页眉页脚剔除。

为什么需要行级信息：**页眉和真标题的文本一字不差**。南岭招股书里有
215 行「第X节…」，其中 203 行是每页重复的 10.5pt 页眉，真标题只有 16
行（16pt）。只按「文本重复 ≥3 页」去重会把第四节、第十~十二节的标题
一起误杀（实测只剩 12 节），所以判据必须是「字号 + 纵向位置 + 编号正则」
三者组合，文本重复只能作辅助。

同样的噪声也是图谱里「管理层讨论与分析」「其他事项说明」这类章节名
实体的来源——它们本来就该是切片的 section 标签，而不是正文内容。

对外只暴露四件事：
    extract_lines()   从 pymupdf 文档取带字号/纵坐标的行
    body_size()       正文字号（按字符量加权取众数）
    strip_noise()     剔除页眉、页脚、目录点线行
    group_by_section() 按一级节（第X节 / 第X章）分组

没有字号信息（扫描件）时 group_by_section 会返回空 id 的单组，
调用方据此降级为按页切分——不猜。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ai_agent.core.logging import get_logger

logger = get_logger(__name__)

# 一级节标题：第X节 / 第X章（招股书用「节」，公司章程类文档用「章」）
_SEC_L1 = re.compile(r"^(第[一二三四五六七八九十百]+[节章])(.*)$")
# 目录行的点线引导符：………… 或 …… 或 .....
_TOC_DOTS = re.compile(r"[.．·…]{3,}")
# 纯页码 / 页脚编号：1-1-141、- 12 -
_FOOTER_NO = re.compile(r"^[\s\-–—]*\d+([-–—]\d+)*[\s\-–—]*$")

TOP_RATIO = 0.07  # 页眉区：页面上部 7%（实测页眉 y=0.052，真标题 y≥0.086）
REPEAT_TOP_RATIO = 0.09  # 跨页重复文本只在贴顶时才当页眉
BOTTOM_RATIO = 0.90  # 页脚区：页面下部 10%（实测页码 y=0.93）
HEADING_SCALE = 1.2  # 标题字号至少是正文的 1.2 倍（12pt 正文 → 14.4pt 起）
MAX_TITLE_LEN = 40  # 标题不会太长


@dataclass(frozen=True)
class Line:
    text: str
    page: int
    size: float
    y_ratio: float  # 行顶相对页面高度（0=顶，1=底）

    @property
    def norm(self) -> str:
        """去空白后的文本：PDF 常在字符间插空格（「杭州 分 公司」）。"""
        return re.sub(r"\s+", "", self.text)


@dataclass
class Section:
    id: str  # 「第十节」；前置内容为空串
    title: str  # 「管理层讨论与分析」
    start_page: int
    end_page: int = 0
    lines: list[Line] = field(default_factory=list)

    @property
    def path(self) -> str:
        """切片/引用用的章节标签：「第十节 管理层讨论与分析」。"""
        return f"{self.id} {self.title}".strip()


def extract_lines(doc) -> list[Line]:
    """把每页文本拆成带字号、纵向位置的行（doc 为 pymupdf.Document）。"""
    out: list[Line] = []
    for pno in range(doc.page_count):
        page = doc[pno]
        height = page.rect.height or 1.0
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"]).strip()
                if not text:
                    continue
                size = round(max(span["size"] for span in line["spans"]), 1)
                out.append(Line(text, pno + 1, size, line["bbox"][1] / height))
    return out


def body_size(lines: list[Line]) -> float:
    """正文字号：按字符量加权取众数（招股书实测 12.0pt）。"""
    weight: Counter[float] = Counter()
    for ln in lines:
        weight[ln.size] += len(ln.text)
    return weight.most_common(1)[0][0] if weight else 12.0


def is_l1_heading(ln: Line, body: float) -> bool:
    return (
        bool(_SEC_L1.match(ln.norm))
        and len(ln.norm) <= MAX_TITLE_LEN
        and ln.size >= body * HEADING_SCALE
    )


def strip_noise(lines: list[Line], body: float) -> tuple[list[Line], dict[str, int]]:
    """剔除页眉、页脚、目录点线行，返回 (保留行, 删除统计)。

    页眉要两遍处理：先按「贴顶 + 小字号」认定哪些**文本**是页眉，再全域
    删除同文本的小字号行——实测有个别页的页眉被排版挤到 y=0.117，纯位置
    规则会漏掉它，切片里就残留一行「第十二节 募股资金运用」。

    跨页重复不能单独当判据：财务表的列头（「2005 年」）也跨页重复，但它们
    在表格中部（实测 y≈0.17），一刀切会把表头数据吃光（删除率 10.8% → 7.0%）。
    """
    pages_by_text: dict[str, set[int]] = defaultdict(set)
    for ln in lines:
        pages_by_text[ln.norm].add(ln.page)
    repeated = {
        t for t, pages in pages_by_text.items() if len(pages) >= 3 and len(t) > 2
    }

    header_texts: set[str] = set()
    for ln in lines:
        size_small = ln.size < body * 1.05
        if not size_small:
            continue
        if ln.y_ratio <= TOP_RATIO or (
            ln.norm in repeated and ln.y_ratio <= REPEAT_TOP_RATIO
        ):
            header_texts.add(ln.norm)

    kept: list[Line] = []
    stats: Counter[str] = Counter()
    for ln in lines:
        size_small = ln.size < body * 1.05
        if ln.y_ratio <= TOP_RATIO and size_small:
            stats["页眉(贴顶小字号)"] += 1
            continue
        if ln.norm in header_texts and size_small:
            stats["页眉(同文本异位)"] += 1
            continue
        if ln.y_ratio >= BOTTOM_RATIO and (size_small or _FOOTER_NO.match(ln.text)):
            stats["页脚(贴底/页码)"] += 1
            continue
        if ln.norm in repeated and size_small and ln.y_ratio <= REPEAT_TOP_RATIO:
            stats["页眉(跨页重复)"] += 1
            continue
        if _TOC_DOTS.search(ln.text) and size_small:
            stats["目录点线行"] += 1
            continue
        kept.append(ln)
    return kept, dict(stats)


def group_by_section(lines: list[Line], body: float) -> list[Section]:
    """按一级节切分。识别不到任何标题时返回单组（id 为空），由调用方降级。

    空章节也保留：亚厦招股书第十五章那 7 页是扫描件签名页，只有标题行、
    正文是图片，丢掉它会让章节树少一章，反而更难排查。
    """
    sections: list[Section] = []
    current = Section("", "", start_page=lines[0].page if lines else 1)
    for ln in lines:
        if not is_l1_heading(ln, body):
            current.lines.append(ln)
            continue
        m = _SEC_L1.match(ln.norm)
        sid, title = (m.group(1), m.group(2)) if m else (ln.norm, "")
        # 同一节标题重复出现（个别文档每页重复且字号也放大）时忽略
        if sections and sections[-1].id == sid:
            continue
        current.end_page = max(ln.page, current.end_page)
        sections.append(current)
        current = Section(sid, title, start_page=ln.page)
    current.end_page = max((ln.page for ln in current.lines), default=current.start_page)
    sections.append(current)

    for sec in sections:
        if sec.lines:
            sec.end_page = max(ln.page for ln in sec.lines)
    found = sum(1 for s in sections if s.id)
    if not found:
        logger.info("未识别到一级节标题（可能无字号元信息），降级为按页切分")
    return sections
