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

页码溯源：解析层产出 Segment（带 page / kind），逐段切片后每片携带
自己的页码与类型（PDF 页码、Excel sheet 序号、或 None）。页边界是
天然分隔，不做跨页合并切片。
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


def chunk_segments(
    segments: list[Segment],
    *,
    max_chars: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for seg in segments:
        if seg.kind == TABLE:
            pieces = _chunk_table(seg.text, max_chars=max_chars)
        else:
            pieces = chunk_text(seg.text, max_chars=max_chars, overlap=overlap)
        chunks.extend(
            Chunk(text=piece, page=seg.page, kind=seg.kind) for piece in pieces
        )
    return chunks


def chunk_text(text: str, *, max_chars: int = 500, overlap: int = 50) -> list[str]:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
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
        length += len(para)
    if current:
        chunks.append("\n".join(current))
    return [c.strip() for c in chunks if c.strip()]


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


def context_prefix(filename: str, *, page: int | None, kind: str) -> str:
    """切片的语境前缀：文档名 + 位置。

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
    return f"《{stem}》{where}" if where else f"《{stem}》"


def embedding_input(text: str, *, filename: str, page: int | None, kind: str) -> str:
    """送进向量模型的文本 = 语境前缀 + 正文。"""
    return f"{context_prefix(filename, page=page, kind=kind)}\n{text}"


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
