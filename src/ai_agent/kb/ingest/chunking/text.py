"""纯文本切片：按段落聚合，超长段按句子边界二次切分。

由原 chunking.py 拆分而来，逻辑未改动。
"""

import re

from ai_agent.kb.ingest.parsing import Segment, TABLE

from ai_agent.kb.ingest.chunking.common import (
    Chunk,
)

from ai_agent.kb.ingest.chunking.html import (
    _can_merge_html_tables,
    _html_chunk_kind,
    _html_pieces,
    _merge_html_tables,
)


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
        for piece_kind, piece in _html_pieces(para):
            if piece_kind == "table":
                if current:
                    chunks.append("\n".join(current))
                    current, length = [], 0
                chunks.append(piece)
                continue
            if len(piece) > max_chars:
                if current:
                    chunks.append("\n".join(current))
                    current, length = [], 0
                chunks.extend(_split_long(piece, max_chars, overlap))
                continue
            if length + len(piece) > max_chars and current:
                chunks.append("\n".join(current))
                current, length = [], 0
            current.append(piece)
            # 换行符也算长度（与 _chunk_flow 一致），否则切片会悄悄超标
            length += len(piece) + 1
    if current:
        chunks.append("\n".join(current))
    return [c.strip() for c in chunks if c.strip()]

def _normalize(text: str) -> str:
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()

def _chunk_plain_pages(
    segments: list[Segment], *, max_chars: int, overlap: int
) -> list[Chunk]:
    """无章节的页各自切片。上一页以表结尾、下一页以表开头且列数相同，则接成一张。"""
    out: list[Chunk] = []
    for seg in segments:
        pieces = [
            Chunk(
                text=piece,
                page=seg.page,
                kind=_html_chunk_kind(seg.kind, piece),
                page_end=seg.page,
            )
            for piece in chunk_text(seg.text, max_chars=max_chars, overlap=overlap)
        ]
        if out and pieces and _can_merge_html_tables(out[-1], pieces[0]):
            out[-1] = Chunk(
                text=_merge_html_tables(out[-1].text, pieces[0].text),
                page=out[-1].page,
                kind=TABLE,
                page_end=pieces[0].page,
            )
            out.extend(pieces[1:])
        else:
            out.extend(pieces)
    return out

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

def _plain_len(text: str) -> int:
    """去掉 HTML 标签后的可见字数，跟向量化正文同一量级。"""
    return len(re.sub(r"<[^>]+>", "", text or ""))

def _same_page(a: Segment, b: Segment) -> bool:
    return a.page is not None and a.page == b.page

# 段落边界：空行或单个换行都算；PDF 抽出来的文本换行很密，宽松处理
_PARA_SPLIT = re.compile(r"\n+")

# 句子边界：中英文常见终止符后的位置（保留标点在句尾）
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")
