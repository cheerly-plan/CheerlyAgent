"""文档路由：模型不指定 files 时，自动选出该查哪几份文件。

为什么需要这一层：让模型对着文件清单挑 files，清单一到几十上百份，
既撑爆上下文，模型也挑不准（还会编造清单外的文件名）。摘要向量把
「哪些文件相关」变成一次 ANN 查询，文件清单因此不必再进提示词。

点名和摘要二选一，不混用：
- **提及优先**：问句里点名了主体/别名/文件名（「A 和 B 的营业收入」），
  命中即收口，不再用摘要补别的文件。补进来的常常是别的公司，既挤占
  已经点到的名额，也让后面没法把公司名从检索词里剥掉。
- **语义补足**：一句都没点名时（「哪家公司的毛利率最高」）才走摘要向量，
  距离超过 kb_doc_route_max_distance 的文件不进候选。
"""

from __future__ import annotations

import uuid

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.storage.embeddings import embed_query
from ai_agent.storage.store import store

logger = get_logger(__name__)


def mentioned_file_ids(query: str) -> list[uuid.UUID]:
    """问句直接点名的文件（按文件名 / 主体名 / 别名匹配），按文件清单顺序。

    复用文件名解析那一套判据（见 kb/retrieval.py）。简称只认别名表里的整词，
    「季度报告」这种噪音不会误命中。
    """
    from ai_agent.kb.retrieval.retrieval import (
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
    ready = [row for row in rows if row.get("status") == "ready"]
    hits: list[uuid.UUID] = []
    for row in ready:
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
    # 「中科飞测」这种简称不在别名表里，但是主体名里独有的一段
    from ai_agent.kb.ingest.meta import fragment_subjects

    catalog = [
        (str(row.get("subject") or ""), list(row.get("subject_aliases") or []))
        for row in ready
    ]
    mentioned = fragment_subjects(text, catalog)
    if mentioned:
        for row in ready:
            fid = row["file_id"]
            if fid in hits:
                continue
            if str(row.get("subject") or "") in mentioned:
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
    # 点名已经圈定文件。再用摘要去补，名额会被别的公司占掉，
    # 公司名也剥不掉（范围里混进了没点名的文件）。
    #
    # 这里**不截断**：名字匹配是确定性的，问句点到哪家就命中哪家的全部报告。
    # top_k 是给下面「摘要向量补位」用的上限，套到点名结果上会出事——
    # 中科飞测有 9 份报告，截到 6 份正好把 2025Q3/2026Q3 丢掉，再按问句里的
    # 「前三季度」收窄就变成 0 份，整条检索直接判「未找到」。
    if picked:
        return picked
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
