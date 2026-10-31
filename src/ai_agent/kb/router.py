"""文档路由：模型不指定 files 时，自动选出该查哪几份文件。

为什么需要这一层：让模型对着文件清单挑 files，清单一到几十上百份，
既撑爆上下文，模型也挑不准（还会编造清单外的文件名）。摘要向量把
「哪些文件相关」变成一次 ANN 查询，文件清单因此不必再进提示词。

两路合成，各补对方盲区：
- **提及优先**：问句里点名了主体/别名/文件名（「A 和 B 的营业收入」），
  确定性命中，不看向量距离——对比类问句里两家都得进得来，靠向量排序
  保证不了这一点（离混合向量更近的那家会独占名额，正是之前踩过的坑）。
- **语义补足**：没点名时（「哪家公司的毛利率最高」）由摘要向量补齐，
  距离超过 kb_doc_route_max_distance 的文件不进候选。
"""

from __future__ import annotations

import uuid

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.memory.embeddings import embed_query
from ai_agent.memory.store import store

logger = get_logger(__name__)


def mentioned_file_ids(query: str) -> list[uuid.UUID]:
    """问句直接点名的文件（按文件名 / 主体名 / 别名匹配），按文件清单顺序。

    复用文件名解析那一套判据（见 kb/retrieval.py），所以「中国能建」这种
    简称能命中全称，「季度报告」这种噪音不会误命中。
    """
    from ai_agent.kb.retrieval import (
        _filename_hint_matches,
        _subject_token_matches,
    )

    text = str(query or "").strip()
    if not text:
        return []
    try:
        rows = store.list_kb_files()
    except Exception:
        logger.warning("文档路由读取文件清单失败，跳过点名匹配", exc_info=True)
        return []
    hits: list[uuid.UUID] = []
    for row in rows:
        if row.get("status") != "ready":
            continue
        fid = row["file_id"]
        if fid in hits:
            continue
        named = _filename_hint_matches(text, str(row.get("filename") or "")) or (
            _subject_token_matches(
                text,
                str(row.get("subject") or ""),
                list(row.get("subject_aliases") or []),
            )
        )
        if named:
            hits.append(fid)
    return hits


def route_files(
    query: str, *, file_ids: list[uuid.UUID] | None = None
) -> list[uuid.UUID]:
    """路由出与问题相关的文档列表。

    返回空列表表示「路由没选出任何东西」，调用方应退回会话范围 / 全库，
    而不是把它当成「限定为空」——后者的语义是模型明确指定了却解析不到，
    这两种情况对工具的引导完全不同。

    ``file_ids`` 非空时只在给定范围内挑：会话已限定范围时，路由不应该把
    范围外的文件放进来。
    """
    if not settings.kb_doc_route_enabled:
        return []
    text = str(query or "").strip()
    if not text:
        return []
    top_k = max(1, settings.kb_doc_route_top_k)
    picked = mentioned_file_ids(text)
    if file_ids:
        allowed = set(file_ids)
        picked = [fid for fid in picked if fid in allowed]
    # 点名的文件优先且全部保留（对比问句里两家都得在），多于此上限直接截断
    if len(picked) >= top_k:
        return picked[:top_k]
    try:
        vector = embed_query(text)
    except Exception:
        logger.warning("文档路由向量化失败，退回点名结果", exc_info=True)
        return picked
    try:
        rows = store.search_kb_files(
            vector,
            file_ids=list(file_ids) if file_ids else None,
            top_k=top_k,
            max_distance=settings.kb_doc_route_max_distance,
        )
    except Exception:
        logger.warning("文档路由检索失败，退回点名结果", exc_info=True)
        return picked
    seen = set(picked)
    for row in rows:
        fid = row["file_id"]
        if fid in seen:
            continue
        seen.add(fid)
        picked.append(fid)
        if len(picked) >= top_k:
            break
    return picked
