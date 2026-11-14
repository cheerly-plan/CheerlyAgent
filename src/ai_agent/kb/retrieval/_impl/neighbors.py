"""邻接扩展：命中片前后补段、同表行合并。

由原 retrieval.py 拆分而来，逻辑未改动。
"""

from __future__ import annotations

import re
import uuid

from ai_agent.storage.store import store
from ai_agent.core.config import settings

from ai_agent.core.logging import get_logger

from ai_agent.kb.retrieval._impl.types import (
    Hit,
)

logger = get_logger(__name__)


# 测试用 monkeypatch.setattr(retrieval, NAME, ...) 打补丁的知识点：
# 直接 NAME(...) 走的是本模块 globals，绕过模块属性，补丁会失效；
# 统一经 _public() 取「对外模块」的当前属性再调用，补丁才生效。
def _public():
    import ai_agent.kb.retrieval.retrieval as public

    return public


def _table_header(text: str) -> str:
    """表格切片的表头指纹。

    * Markdown：前 3 个非空行（标题 / 单位 / 列名）；
    * HTML：**首行的单元格文字**。整段 HTML 是一行，按行取指纹会让同一张表
      的两片各不相同——实测兆易创新 Q3 那张 OCR 表就在 top_k 里出现了两次。
    """
    raw = text or ""
    if raw.lstrip().lower().startswith("<table"):
        match = _FIRST_ROW.search(raw)
        if match:
            cells = [_CELL.sub(r"\1", cell) for cell in _CELL.findall(match.group(1))]
            return re.sub(r"\s+", "", "".join(cells))
        return re.sub(r"\s+", "", raw[:120])
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return "".join(re.sub(r"\s+", "", ln) for ln in lines[:3])

def _merge_same_table_rows(rows: list[dict]) -> list[dict]:
    """同一文件同一页、表头相同的多片表格合并成一条。

    长表每片都重印表头（刻意的，避免切片独立后失去列名），于是同表几片一起
    进 top_k，既浪费名额（模型只拿到碎片），又重复占上下文。

    命中不相邻时（seq 1 和 5）按 seq 顺序合并，**空档也补进来**：那是同一张
    表的中间行，答案可能就在里面。空档超过 ``_MERGE_MAX_GAP`` 片或合并后超过
    ``_MERGE_MAX_CHARS`` 字就放弃这次合并，保持原样——宁可多占名额，也不能
    把一张长表整块塞进上下文。
    """
    groups: dict[tuple[str, int, str], list[int]] = {}
    for index, row in enumerate(rows):
        if (row.get("kind") or "") != "table":
            continue
        header = _table_header(row.get("text") or "")
        if not header:
            continue
        key = (str(row.get("file_id")), int(row.get("page") or 0), header)
        groups.setdefault(key, []).append(index)

    drop: set[int] = set()
    merged_rows: dict[int, dict] = {}
    for (file_id, _page, _header), indexes in groups.items():
        if len(indexes) < 2:
            continue
        members = [rows[i] for i in indexes]
        seqs = sorted(int(m.get("seq") or 0) for m in members)
        gaps = [
            seq
            for seq in range(seqs[0], seqs[-1] + 1)
            if seq not in set(seqs)
        ]
        # 空档小就补进来（同表的中间行，答案可能就在里面）；
        # 空档大就只合并命中的那几片，中间用一行标注省略——既保住完整的表名
        # 与列名，也不至于为了一段推测的相关性把整张长表塞进上下文。
        extra: list[dict] = []
        if gaps and len(gaps) <= _MERGE_MAX_GAP:
            extra = _fetch_merged_gap(file_id, seqs, gaps)
        ordered = sorted([*members, *extra], key=lambda m: int(m.get("seq") or 0))
        text = _public()._join_table_texts(
            [(m.get("text") or "", int(m.get("seq") or 0)) for m in ordered]
        )
        if len(text) > _MERGE_MAX_CHARS:
            continue
        # 距离为 0.0 是完全匹配，不能用 `or 1.0` 兜底（会把最佳成员误判成最差）
        def _distance(index: int) -> float:
            value = rows[index].get("distance")
            return 1.0 if value is None else float(value)

        best = min(indexes, key=_distance)
        merged = dict(rows[best])
        merged["text"] = text
        merged["seq"] = seqs[0]
        pages = [int(m.get("page") or 0) for m in members if m.get("page")]
        if pages:
            merged["page"] = min(pages)
            merged["page_end"] = max(pages)
        distances = [m["distance"] for m in members if m.get("distance") is not None]
        if distances:
            merged["distance"] = min(distances)
        merged_rows[best] = merged
        drop.update(i for i in indexes if i != best)

    if not merged_rows and not drop:
        return rows
    out: list[dict] = []
    for index, row in enumerate(rows):
        if index in merged_rows:
            out.append(merged_rows[index])
        elif index in drop:
            continue
        else:
            out.append(row)
    return out

def _fetch_merged_gap(file_id: str, seqs: list[int], gaps: list[int]) -> list[dict]:
    """把空档里的切片取回来（同表中间行）。取不到就不补，不影响主流程。"""
    try:
        import uuid as _uuid

        pairs = [(_uuid.UUID(file_id), seq) for seq in seqs]
        grouped = store.kb_chunk_neighbors(pairs, radius=_MERGE_MAX_GAP + 1)
    except Exception:
        logger.warning("合并同表切片时读取空档失败，跳过补全", exc_info=True)
        return []
    found: list[dict] = []
    for seq in gaps:
        rows = grouped.get((file_id, seq)) or []
        if rows:
            found.append(dict(rows[0], file_id=file_id))
    return found

def _join_table_texts(segments: list[tuple[str, int]]) -> str:
    """按 seq 顺序拼接同表切片，重复表头只留第一份；空缺处标注省略。"""
    if not segments:
        return ""
    texts = [text for text, _seq in segments]
    seqs = [seq for _text, seq in segments]
    head_lines = [
        ln for ln in texts[0].splitlines() if ln.strip()
    ][:3]
    out = list(texts[0].splitlines())
    for index, text in enumerate(texts[1:], start=1):
        omitted = seqs[index] - seqs[index - 1] - 1
        if omitted > 0:
            out.append(f"…（同表另有 {omitted} 片未展开）…")
        lines = text.splitlines()
        # 跳过与前一片重复的表头行（标题 / 表名 / 列名 / 分隔线）
        start = 0
        head_set = {re.sub(r"\s+", "", ln) for ln in head_lines}
        while start < len(lines) and (
            not lines[start].strip()
            or re.sub(r"\s+", "", lines[start]) in head_set
            or re.fullmatch(r"\|[\s\-|]+\|", lines[start].strip() or "-")
        ):
            start += 1
        out.extend(lines[start:])
    return "\n".join(out).strip()

def _attach_neighbors(hits: list[Hit]) -> list[Hit]:
    radius = settings.kb_neighbor_expand
    if radius <= 0 or not hits:
        return hits
    try:
        pairs = [(uuid.UUID(hit.file_id), hit.seq) for hit in hits]
        grouped = store.kb_chunk_neighbors(pairs, radius=radius)
    except Exception:
        # 邻接只是锦上添花，取不到就按原样返回，不能把整个检索拖挂
        logger.warning("kb 邻接片段读取失败，跳过上下文扩展", exc_info=True)
        return hits
    for hit in hits:
        before: list[tuple[int, str]] = []
        after: list[tuple[int, str]] = []
        for delta in range(1, radius + 1):
            prev = grouped.get((hit.file_id, hit.seq - delta))
            if prev:
                before.append((hit.seq - delta, _public().render_otsl(prev[0]["text"])))
            nxt = grouped.get((hit.file_id, hit.seq + delta))
            if nxt:
                after.append((hit.seq + delta, _public().render_otsl(nxt[0]["text"])))
        hit.before = [text for _, text in sorted(before)]
        hit.after = [text for _, text in sorted(after)]
    return hits

# 同一张长表被切成多片时，命中常常落在几片不相邻的切片上（1 和 5）。
# 合并时把中间的空档一起补进来——模型要的数很可能正在空档里——但设上限，
# 免得一张表把整个上下文吃光。
_MERGE_MAX_GAP = 2        # 空档最多补 2 片

_MERGE_MAX_CHARS = 2000   # 合并后的正文上限

_FIRST_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)

_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
