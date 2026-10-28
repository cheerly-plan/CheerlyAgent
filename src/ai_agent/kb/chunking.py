"""文档切片。

两级策略：先按段落聚合到目标长度；单段超长时按句子边界（中英文
标点）二次切分。相邻切片保留 overlap 字符，避免关键信息恰好落在
边界上被截断。纯字符数切分（中文 1 字符 ≈ 1 token 量级），不上
语义切分——收益配不上复杂度。

按内容类型分派：
- text：上面的常规策略
- table：整表成一片，超长时按行切并**每片重复表头**，否则行与列名
  脱节，检索命中后无法还原"哪一年的营收"
- figure：图像描述本身很短，直接沿用常规策略（一般不触发切分）

两级栅栏，谁窄听谁的：
- **节栅栏**：同节的连续正文段先合成一条流再聚合，一片不跨章节。
  章节是语义的自然边界（"第十节 管理层讨论与分析"里不会混进"第五节
  风险因素"的句子），跨节拼接只会让引用张冠李戴；
- **页边界**：无章节信息的文档（扫描件、MD/TXT、Excel）仍按段独立切，
  不做跨页合并——页码是溯源的最低要求。

页码溯源：解析层产出 Segment（带 page / section / kind），切片后每片
携带自己的页码区间与类型（PDF 页码、Excel sheet 序号、或 None）。
"""

import re
from dataclasses import dataclass
from pathlib import Path

from ai_agent.kb.parsing import FIGURE, TABLE, TEXT, Segment

# 段落边界：空行或单个换行都算；PDF 抽出来的文本换行很密，宽松处理
_PARA_SPLIT = re.compile(r"\n+")
# 句子边界：中英文常见终止符后的位置（保留标点在句尾）
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")
# Markdown 表格的分隔行：|---|---|
_MD_SEPARATOR = re.compile(r"^\|[\s:\-|]+\|$")


@dataclass
class Chunk:
    text: str
    page: int | None = None  # PDF 页码 / Excel sheet 序号；无页概念的文档为 None
    kind: str = TEXT  # text / table / figure
    page_end: int | None = None  # 同节内聚合导致的跨页切片：末页页码
    section: str | None = None  # 所属一级节，如「第十节 管理层讨论与分析」


def chunk_segments(
    segments: list[Segment],
    *,
    max_chars: int = 500,
    overlap: int = 50,
    within_page: bool = False,
) -> list[Chunk]:
    """按节成流、按段兜底：同节连续正文合并后聚合，表与无章节段独立处理。

    `within_page=True` 时同节内也按页断（一片只落一页，溯源最精确，但
    跨页的长论述会被切断）；默认 False，允许同节内跨页聚合。
    """
    chunks: list[Chunk] = []
    flow: list[Segment] = []

    def flush_flow() -> None:
        if flow:
            chunks.extend(
                _chunk_flow(flow, max_chars=max_chars, overlap=overlap)
            )
            flow.clear()

    for seg in segments:
        if seg.kind == TABLE:
            # 表必须整表成片（超长时重复表头），不能揉进正文流
            flush_flow()
            chunks.extend(
                Chunk(
                    text=piece,
                    page=seg.page,
                    kind=seg.kind,
                    page_end=seg.page,
                    section=seg.section,
                )
                for piece in _chunk_table(seg.text, max_chars=max_chars)
            )
        elif seg.section is None:
            # 无章节信息：保持「一段一世界」，段与段之间不合并
            flush_flow()
            chunks.extend(
                Chunk(
                    text=piece,
                    page=seg.page,
                    kind=seg.kind,
                    page_end=seg.page,
                )
                for piece in chunk_text(
                    seg.text, max_chars=max_chars, overlap=overlap
                )
            )
        else:
            # 两条栅栏：换节必断；开了单页栅栏时换页也断。
            # 断点处不保留重叠——重叠只发生在同一片流内部（超长段落二次
            # 切分时），所以重叠永远不会跨节、跨页（when within_page）。
            if flow and (
                flow[0].section != seg.section
                or (within_page and flow[-1].page != seg.page)
            ):
                flush_flow()
            flow.append(seg)
    flush_flow()
    return chunks


def _chunk_flow(
    segments: list[Segment], *, max_chars: int, overlap: int
) -> list[Chunk]:
    """把同一节内的连续页聚合成切片流，页码取区间。"""
    section = segments[0].section
    kind = segments[0].kind
    out: list[Chunk] = []
    buf: list[str] = []
    pages: list[int] = []
    length = 0

    def flush() -> None:
        nonlocal buf, pages, length
        if buf:
            text = "\n".join(buf).strip()
            if text:
                out.append(
                    Chunk(
                        text=text,
                        page=min(pages) if pages else None,
                        kind=kind,
                        page_end=max(pages) if pages else None,
                        section=section,
                    )
                )
            buf, pages, length = [], [], 0

    for seg in segments:
        for para in _PARA_SPLIT.split(_normalize(seg.text)):
            para = para.strip()
            if not para:
                continue
            if len(para) > max_chars:
                # 超长段落先收掉已聚合的内容，再单独二次切分
                flush()
                out.extend(
                    Chunk(
                        text=piece,
                        page=seg.page,
                        kind=kind,
                        page_end=seg.page,
                        section=section,
                    )
                    for piece in _split_long(para, max_chars, overlap)
                )
                continue
            if length + len(para) > max_chars and buf:
                flush()
            buf.append(para)
            if seg.page is not None:
                pages.append(seg.page)
            # 拼接用的换行符也占长度：漏算会让切片实际超出上限
            # （原实现中位 511 字、13% 超过 550，就是这个原因）
            length += len(para) + 1
    flush()
    return out


def chunk_text(text: str, *, max_chars: int = 500, overlap: int = 50) -> list[str]:
    text = _normalize(text)
    if not text:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            # 超长段落先收掉已聚合的内容，再单独二次切分
            if current:
                chunks.append("\n".join(current))
                current, length = [], 0
            chunks.extend(_split_long(para, max_chars, overlap))
            continue
        if length + len(para) > max_chars and current:
            chunks.append("\n".join(current))
            current, length = [], 0
        current.append(para)
        # 换行符也算长度（与 _chunk_flow 一致），否则切片会悄悄超标
        length += len(para) + 1
    if current:
        chunks.append("\n".join(current))
    return [c.strip() for c in chunks if c.strip()]


def _normalize(text: str) -> str:
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _chunk_table(text: str, *, max_chars: int) -> list[str]:
    """表格切片：短表整片保留；长表按行切，每片重复表头（含工作表名）。"""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if len(text) <= max_chars:
        return [text]

    sep_index = next(
        (i for i, line in enumerate(lines[:5]) if _MD_SEPARATOR.match(line.strip())),
        None,
    )
    if sep_index is not None:
        header, data = lines[: sep_index + 1], lines[sep_index + 1 :]
    else:
        header, data = lines[:1], lines[1:]
    if not data:
        # 不是表格形态（单行超长），退回普通切分
        return chunk_text(text, max_chars=max_chars, overlap=0)

    out: list[str] = []
    buf: list[str] = list(header)
    for line in data:
        if len(buf) > len(header) and _joined_len(buf + [line]) > max_chars:
            out.append("\n".join(buf))
            buf = list(header)
        buf.append(line)
    if len(buf) > len(header):
        out.append("\n".join(buf))
    return out or [text]


def _joined_len(lines: list[str]) -> int:
    return sum(len(line) + 1 for line in lines)


def context_prefix(
    filename: str, *, page: int | None, kind: str, section: str | None = None
) -> str:
    """切片的语境前缀：文档名 + 位置 + 章节。

    切片单看常常缺主语（"同比增长 56%" 是哪家公司的？哪一年的？），
    向量化时把这份语境一起嵌进去，召回才分得清同名指标；同时也让
    模型引用时有据可依。存储的正文保持干净——前缀只在嵌入时拼。
    """
    stem = Path(filename).stem if filename else "文档"
    if kind == TABLE:
        where = f"表{page}" if page is not None else "表格"
    elif kind == FIGURE:
        where = f"第{page}页图表" if page is not None else "图表"
    else:
        where = f"第{page}页" if page is not None else ""
    base = f"《{stem}》{where}" if where else f"《{stem}》"
    return f"{base} · {section}" if section else base


def embedding_input(
    text: str,
    *,
    filename: str,
    page: int | None,
    kind: str,
    section: str | None = None,
) -> str:
    """送进向量模型的文本 = 语境前缀 + 正文。"""
    return f"{context_prefix(filename, page=page, kind=kind, section=section)}\n{text}"


def _split_long(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    out: list[str] = []
    buf = ""
    for sent in (s for s in _SENT_SPLIT.split(paragraph) if s.strip()):
        if len(sent) > max_chars:
            # 单句仍超长（无标点的连续文本）：硬切，保留 overlap
            if buf:
                out.append(buf)
                buf = ""
            step = max(max_chars - overlap, 1)
            for i in range(0, len(sent), step):
                piece = sent[i : i + max_chars]
                if len(piece) > overlap or i + max_chars >= len(sent):
                    out.append(piece)
            continue
        if buf and len(buf) + len(sent) > max_chars:
            out.append(buf)
            # 保留尾部 overlap 作为下一片开头
            buf = buf[-overlap:] if overlap > 0 else ""
        buf += sent
    if buf.strip():
        out.append(buf)
    return [s.strip() for s in out if s.strip()]
