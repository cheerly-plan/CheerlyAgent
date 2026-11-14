"""当前会话历史检索：摘要之外，按关键词找回旧对话原文。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai_agent.core.config import settings
from ai_agent.core.text import clip_text
from ai_agent.prompts.tools import (
    READ_HISTORY_AFTER_DESC,
    READ_HISTORY_BEFORE_DESC,
    READ_HISTORY_DESC,
    READ_HISTORY_SEQ_DESC,
    SEARCH_HISTORY_DESC,
    SEARCH_HISTORY_LIMIT_DESC,
    SEARCH_HISTORY_QUERY_DESC,
)
from ai_agent.storage.db import current_user_id
from ai_agent.tools.base import Tool
from ai_agent.tools.context import current_session_id


class SearchHistoryArgs(BaseModel):
    query: str = Field(description=SEARCH_HISTORY_QUERY_DESC)
    limit: int = Field(default=8, ge=1, le=20, description=SEARCH_HISTORY_LIMIT_DESC)


def search_history(query: str, limit: int = 8) -> str:
    text = (query or "").strip()
    if not text:
        return "请提供检索关键词，例如 search_history(query=\"合同负债\")。"
    session_id = current_session_id()
    if not session_id:
        return "当前没有可检索的会话历史。"
    try:
        from ai_agent.storage.store import store

        rows = store.search_chat_history(
            current_user_id(), session_id, text, limit=limit
        )
    except Exception:
        return "检索会话历史失败，请稍后重试或换一个关键词。"

    # 关键词命中不足时补一次语义召回：原文说「预收账款」、用户问「合同负债」
    # 这类换词场景只在向量上才连得上。语义结果按 seq 去重后追加。
    if len(rows) < limit:
        rows = _merge_semantic(rows, text, session_id, limit)
    if not rows:
        return f"没有找到包含「{text}」的历史对话。"

    lines: list[str] = []
    for row in rows:
        role = "用户" if row.get("role") == "user" else "助手"
        content = clip_text(str(row.get("content") or "").strip(), 500)
        if content:
            lines.append(f"[seq={row.get('seq')}] {role}：{content}")
    return "\n\n".join(lines)


search_history_tool = Tool(
    name="search_history",
    description=SEARCH_HISTORY_DESC,
    func=search_history,
    args_model=SearchHistoryArgs,
    max_output_chars=4000,
)


class ReadHistoryArgs(BaseModel):
    seq: int = Field(description=READ_HISTORY_SEQ_DESC)
    before: int = Field(default=2, ge=0, le=20, description=READ_HISTORY_BEFORE_DESC)
    after: int = Field(default=2, ge=0, le=20, description=READ_HISTORY_AFTER_DESC)


def read_history(seq: int, before: int = 2, after: int = 2) -> str:
    session_id = current_session_id()
    if not session_id:
        return "当前没有可读取的会话历史。"
    try:
        from ai_agent.storage.store import store

        rows = store.load_chat_around(
            current_user_id(), session_id, seq, before=before, after=after
        )
    except Exception:
        return "读取会话历史失败，请稍后重试。"
    if not rows:
        return f"seq={seq} 附近没有可读的历史对话。"

    lines: list[str] = []
    for row in rows:
        role = "用户" if row.get("role") == "user" else "助手"
        content = clip_text(str(row.get("content") or "").strip(), 600)
        if content:
            mark = " ←" if row.get("seq") == int(seq) else ""
            lines.append(f"[seq={row.get('seq')}]{mark} {role}：{content}")
    return "\n\n".join(lines)


read_history_tool = Tool(
    name="read_history",
    description=READ_HISTORY_DESC,
    func=read_history,
    args_model=ReadHistoryArgs,
    max_output_chars=6000,
)


def _merge_semantic(
    keyword_rows: list[dict], query: str, session_id: str, limit: int
) -> list[dict]:
    """关键词 + 语义两路按 RRF 融合；任一路失败都退回另一路。

    关键词语料和向量语料的分数量纲不可比（ILIKE 只有「中/不中」，
    余弦相似度是连续值），所以按名次融合而不是加权求和：
    两路都排前的条目自然最靠前。
    """
    top_k = int(settings.history_semantic_top_k)
    if top_k <= 0:
        return keyword_rows
    try:
        from ai_agent.storage.embeddings import embed_query
        from ai_agent.storage.store import store

        vector = embed_query(query)
        semantic = store.search_chat_semantic(
            current_user_id(),
            session_id,
            vector,
            limit=top_k,
            threshold=settings.history_semantic_threshold,
        )
    except Exception:
        return keyword_rows
    if not semantic:
        return keyword_rows

    merged = _rrf_rows(
        [keyword_rows, semantic],
        weights=[
            float(settings.memory_fusion_weight_keyword),
            float(settings.memory_fusion_weight_vector),
        ],
        limit=limit,
    )
    merged.sort(key=lambda row: int(row.get("seq") or 0))
    return merged[: max(1, limit)]


def _rrf_rows(
    ranked_lists: list[list[dict]], *, weights: list[float], limit: int
) -> list[dict]:
    """按 seq 融合多路历史检索结果（与 progressive.fuse_rrf 同一公式）。"""
    from ai_agent.memory.progressive import RRF_K

    scores: dict[int, float] = {}
    rows: dict[int, dict] = {}
    for path_rank, items in enumerate(ranked_lists):
        weight = weights[path_rank] if path_rank < len(weights) else 1.0
        if weight <= 0:
            continue
        for position, row in enumerate(items):
            seq = int(row.get("seq") or 0)
            if seq <= 0:
                continue
            rows.setdefault(seq, row)
            scores[seq] = scores.get(seq, 0.0) + weight / (RRF_K + position + 1)
    ordered = sorted(scores.items(), key=lambda pair: -pair[1])
    out: list[dict] = []
    for seq, score in ordered[: max(1, limit)]:
        out.append({**rows[seq], "score": score})
    return out
