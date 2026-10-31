"""PDF 章节识别与页眉页脚剔除。

为什么需要行级信息：**页眉和真标题的文本一字不差**。南岭招股书里有
215 行「第X节…」，其中 203 行是每页重复的 10.5pt 页眉，真标题只有 16
行（16pt）。只按「文本重复 ≥3 页」去重会把第四节、第十~十二节的标题
一起误杀（实测只剩 12 节），所以判据必须是「字号 + 纵向位置 + 编号正则」
三者组合，文本重复只能作辅助。

两级标题都要认，因为两类文档的骨架不一样：
- 招股书用「第X节」做一级，节内再用「一、（一）」细分；
- 定期报告（季报/年报）**没有「第X节」**，一级就是「一、主要财务数据」，
  二级是「（一）主要会计数据和财务指标」。只认「第X节」时，三份季报的
  71/74/61 个切片一个章节名都没有，图谱 prompt 只能给「第 N 段」。
所以「有没有第X节」决定中文序号算一级还是二级，两级都带上：切片的
section 形如「第十节 管理层讨论与分析 · （一）主营业务」。

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
# 中文序号标题：「一、主要财务数据」（一级，或招股书里的节内一级细分）
_CN_TOP = re.compile(r"^([一二三四五六七八九十]+、)\s*(.*)$")
# 括号序号标题：「（一）主要会计数据和财务指标」（二级）
_CN_SUB = re.compile(r"^（([一二三四五六七八九十]+)）\s*(.*)$")
# 目录行的点线引导符：………… 或 …… 或 .....
_TOC_DOTS = re.compile(r"[.．·…]{3,}")
# 纯页码 / 页脚编号：1-1-141、- 12 -
_FOOTER_NO = re.compile(r"^[\s\-–—]*\d+([-–—]\d+)*[\s\-–—]*$")

TOP_RATIO = 0.07  # 页眉区：页面上部 7%（实测页眉 y=0.052，真标题 y≥0.086）
REPEAT_TOP_RATIO = 0.09  # 跨页重复文本只在贴顶时才当页眉
BOTTOM_RATIO = 0.90  # 页脚区：页面下部 10%（实测页码 y=0.93）
# 报表末页签字栏常落在页脚区，且每张主表重复。不能当页眉/页脚删。
_SIGNATORY_ROLE = re.compile(
    r"(法定代表人|主管会计工作负责人|主管财会工作负责人|"
    r"会计机构负责人|财会机构负责人|财务负责人|"
    r"主管会计工作的负责人)"
)
HEADING_SCALE = 1.2  # 标题字号至少是正文的 1.2 倍（12pt 正文 → 14.4pt 起）
MAX_TITLE_LEN = 40  # 标题不会太长
# 标题行不会以句读收尾；「一、经营活动产生的现金流量：」是报表行不是标题，
# 靠这条和表格行剔除双重挡住（中文序号标题在季报里字号与正文一致，字号
# 判据在这里不起作用，只能靠「形状」）
_NOT_HEADING_TAIL = re.compile(r"[，,、。；;：:！？]$")
# 标题里不出现句中停顿：正文整句被折行后常以「二、…」开头，
# 「二、公司实现营业收入12 亿元，同比增长」这种靠逗号识别
_NOT_HEADING_INNER = re.compile(r"[，,。；;]")


@dataclass(frozen=True)
class Line:
    text: str
    page: int
    size: float
    y_ratio: float  # 行顶相对页面高度（0=顶，1=底）
    # 页面坐标 (x0, y0, x1, y1)：表格抽取后靠它剔除被表格占用的行，
    # 否则同一份内容会在正文切片和表格切片里各存一份
    bbox: tuple[float, float, float, float] | None = None

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
    # 二级标题（「一、主要财务数据」「（一）非经常性损益项目和金额」）。
    # 季报没有第X节，此时 id/title 直接放中文序号标题，sub 放括号标题
    sub: str = ""
    lines: list[Line] = field(default_factory=list)

    @property
    def path(self) -> str:
        """切片/引用用的章节标签：「第十节 管理层讨论与分析」「一、主要财务数据」。"""
        # 「一、」自带分隔符，再加空格会变成「一、 主要财务数据」
        sep = "" if self.id.endswith("、") else " "
        return f"{self.id}{sep}{self.title}".strip()

    @property
    def label(self) -> str:
        """带二级标题的完整标签：没有二级标题时与 path 一致。"""
        return f"{self.path} · {self.sub}" if self.sub and self.path else (self.path or self.sub)


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
                bbox = line["bbox"]
                out.append(
                    Line(text, pno + 1, size, bbox[1] / height, tuple(bbox))
                )
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


def _cn_heading(ln: Line) -> tuple[int, str, str] | None:
    """中文序号标题：返回 (层级, 标记, 标题)。

    层级 1 = 「一、」，2 = 「（一）」。字号不参与判定——季报里一级标题
    和正文同为 10.4pt（实测中国能建），只能靠「短 + 不带句读」的形状。
    正文里偶尔以「一、」开头的整句，因为带逗号/句号会被 _NOT_HEADING_TAIL
    和 _NOT_HEADING_INNER 挡掉。
    """
    if len(ln.norm) > MAX_TITLE_LEN:
        return None
    text = ln.text.strip()
    if _NOT_HEADING_TAIL.search(text) or _NOT_HEADING_INNER.search(ln.norm):
        return None
    m = _CN_TOP.match(ln.norm)
    if m:
        return 1, m.group(1), m.group(2)
    m = _CN_SUB.match(ln.norm)
    if m:
        return 2, f"（{m.group(1)}）", m.group(2)
    return None


def is_signatory(ln: Line) -> bool:
    """报表签字栏（法定代表人等），不能当页眉页脚或表格内文本丢掉。"""
    return bool(_SIGNATORY_ROLE.search(ln.norm))


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
        # 签字栏每张主表都会印，且常落在页脚区，必须先于页眉页脚规则留下。
        if is_signatory(ln):
            kept.append(ln)
            continue
        if ln.y_ratio <= TOP_RATIO and size_small:
            stats["页眉(贴顶小字号)"] += 1
            continue
        if ln.norm in header_texts and size_small:
            stats["页眉(同文本异位)"] += 1
            continue
        # 贴底只删页码。旧规则把「正文字号 × 底部 10%」整段清掉，
        # 资产负债表末的「葛海蛟」正好落在那里。
        if ln.y_ratio >= BOTTOM_RATIO and _FOOTER_NO.match(ln.text):
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
    """按标题切分。识别不到任何标题时返回单组（id 为空），由调用方降级。

    切分点包括一级标题（第X节 / 无「第X节」时的「一、」）和二级标题
    （「一、」/「（一）」）。一个页面里有几个二级标题就出几段，切片不会
    跨小节拼接——「（二）非经常性损益项目和金额」的表格不该和
    「（一）主要会计数据和财务指标」的正文混进同一片。

    空章节也保留：亚厦招股书第十五章那 7 页是扫描件签名页，只有标题行、
    正文是图片，丢掉它会让章节树少一章，反而更难排查。
    """
    if not lines:
        return []
    # 文档骨架有没有「第X节」决定中文序号是几级标题
    has_l1 = any(is_l1_heading(ln, body) for ln in lines)

    sections: list[Section] = []
    l1_id, l1_title, sub = "", "", ""
    current = Section("", "", start_page=lines[0].page)
    for ln in lines:
        heading: tuple[str, str, str, str] | None = None  # kind, id, title, sub
        if is_l1_heading(ln, body):
            m = _SEC_L1.match(ln.norm)
            sid, title = (m.group(1), m.group(2)) if m else (ln.norm, "")
            if sid != l1_id:  # 同一节标题每页重复时忽略（字号也放大的排版）
                heading = ("l1", sid, title, "")
        else:
            cn = _cn_heading(ln)
            if cn:
                level, mark, title = cn
                if level == 1 and not has_l1:
                    if mark != l1_id:
                        heading = ("l1", mark, title, "")
                elif mark != sub:
                    heading = ("sub", l1_id, l1_title, f"{mark}{title}")
        if heading is None:
            current.lines.append(ln)
            continue
        kind, sid, title, new_sub = heading
        if kind == "l1":
            l1_id, l1_title, sub = sid, title, ""
        else:
            sub = new_sub
        current.end_page = max(ln.page, current.end_page)
        sections.append(current)
        current = Section(l1_id, l1_title, start_page=ln.page, sub=sub)
    current.end_page = max((ln.page for ln in current.lines), default=current.start_page)
    sections.append(current)

    for sec in sections:
        if sec.lines:
            sec.end_page = max(ln.page for ln in sec.lines)
    found = sum(1 for s in sections if s.path)
    if not found:
        logger.info("未识别到节标题（可能无字号元信息），降级为按页切分")
    return sections
