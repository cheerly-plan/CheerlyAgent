"""检索入口：search_knowledge / preview_search 编排。

由原 retrieval.py 拆分而来，逻辑未改动。
"""

from __future__ import annotations

import uuid

from ai_agent.core.config import settings
from ai_agent.storage.store import store

from ai_agent.core.logging import get_logger

from ai_agent.kb.retrieval._impl.fusion import (
    _RRF_K,
)

from ai_agent.kb.retrieval._impl.rerank import (
    _RERANK_LIMIT,
)

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


def search_knowledge(
    query: str,
    *,
    file_ids: list[uuid.UUID] | None = None,
    top_k: int = 3,
    filters: dict | None = None,
    per_file: bool = False,
    diversify_query: str | None = None,
) -> list[Hit]:
    """知识库检索主入口。检索词向量化失败会抛异常，由调用方转成友好文案。

    `file_ids` 为空表示搜全库——检索范围完全由调用方决定（工具层传入模型
    指定的 files，或会话级 kb_scope），本层不再根据查询内容自动推断文件范围。
    `filters`（可选）是 chunk 元数据过滤（kinds/subjects/periods/scopes/
    sections），由调用方（工具层）构造后传入：scope 从 query 意图推断、
    kinds/sections 由模型显式指定。带过滤召回为空时自动回退「不过滤」再查
    一次，避免过滤过窄直接挂空。
    """
    embedding = _public().embed_query(query)
    hits = _public()._core_search(
        query,
        embedding,
        file_ids=file_ids,
        top_k=top_k,
        filters=filters,
        per_file=per_file,
        diversify_query=diversify_query,
    )
    if not hits and filters:
        # 元数据过滤后召回为空，多半是约束过窄（如 scope 强约束把母公司切片
        # 全排除了），回退不过滤再查一次，保证至少有结果而非直接挂空
        logger.info("kb 元数据过滤后召回为空，回退不过滤重查: %s", filters)
        hits = _public()._core_search(
            query,
            embedding,
            file_ids=file_ids,
            top_k=top_k,
            filters=None,
            per_file=per_file,
            diversify_query=diversify_query,
        )
    return hits

def search_knowledge_multi(
    queries: list[str],
    *,
    file_ids: list[uuid.UUID] | None = None,
    top_k: int = 3,
    filters: dict | None = None,
    per_file: bool = False,
    diversify_query: str | None = None,
) -> list[Hit]:
    """多查询检索并融合（multi-query RRF）。

    把多条改写变体分别检索后，按 (file_id, seq) 去重并跨查询累加 RRF 分数
    （score = Σ 1/(60+rank)），再统一重排、截断、邻接。单条查询时退化为
    search_knowledge，保证行为一致。任一变体向量化失败仅跳过该变体。
    """
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        return []
    if len(queries) == 1:
        return _public().search_knowledge(
            queries[0],
            file_ids=file_ids,
            top_k=top_k,
            filters=filters,
            per_file=per_file,
            diversify_query=diversify_query,
        )
    merged: dict[tuple[str, int], dict] = {}
    scores: dict[tuple[str, int], float] = {}
    for q in queries:
        try:
            embedding = _public().embed_query(q)
        except Exception:
            logger.exception("multi-query 向量化失败，跳过变体：%s", q[:60])
            continue
        rows = _public()._retrieve_candidates(
            q, embedding, file_ids=file_ids, top_k=top_k, filters=filters
        )
        if not rows and filters:
            rows = _public()._retrieve_candidates(
                q, embedding, file_ids=file_ids, top_k=top_k, filters=None
            )
        for rank, row in enumerate(rows):
            key = (str(row["file_id"]), int(row["seq"]))
            if key not in merged:
                merged[key] = dict(row)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
    if not merged:
        return []
    ranked = sorted(
        merged.values(),
        key=lambda r: scores[(str(r["file_id"]), int(r["seq"]))],
        reverse=True,
    )
    if settings.kb_rerank_enabled and len(ranked) > top_k:
        # 重排用主查询（queries[0]，即改写后的原始问句）做排序
        head = ranked[:_RERANK_LIMIT]
        ranked = _public()._rerank(queries[0], head, top_k) + ranked[_RERANK_LIMIT:]
    ranked = _public()._diversify(
        diversify_query or queries[0], ranked, top_k, per_file=per_file
    )
    ranked = _public()._merge_same_table_rows(ranked)
    hits = [_public()._to_hit(row) for row in ranked[:top_k]]
    return _public()._attach_neighbors(hits)

def preview_search(
    query: str,
    *,
    file_ids: list[uuid.UUID] | None,
    top_k: int = 5,
) -> list[Hit]:
    """知识库页的检索演示。

    ``file_ids`` 为空表示全库。这里不走 ``search_knowledge`` 的文件名收窄：
    演示要的是「没勾选 = 全部」，不能再被标签推断缩成一部分文件。
    """
    embedding = _public().embed_query(query)
    hits = _public()._core_search(query, embedding, file_ids=file_ids, top_k=top_k)
    limit = settings.kb_search_max_distance
    return [hit for hit in hits if hit.distance is None or hit.distance <= limit]

def _retrieve_candidates(
    query: str,
    embedding: list[float],
    *,
    file_ids: list[uuid.UUID] | None,
    top_k: int,
    filters: dict | None = None,
) -> list[dict]:
    """一轮检索的候选召回：召回 → 关键词融合 → 口径偏好 → 去重（不含重排/截断/邻接）。

    供 search_knowledge（单 query）与 search_knowledge_multi（多 query 融合）共用。
    """
    candidates = max(top_k, settings.kb_search_candidates)
    rows = store.search_kb_chunks(
        embedding,
        file_ids=file_ids,
        top_k=candidates,
        max_distance=settings.kb_search_max_distance,
        **(filters or {}),
    )
    rows = _public()._fuse_with_keywords(query, rows, file_ids, filters=filters)
    if not rows:
        return []
    # 口径偏好放在去冗余与重排之前：它改的是「谁先被看到」，重排是从
    # 这 12 条候选里挑，候选顺序本身会影响模型的注意力
    rows = _public()._apply_scope_preference(query, rows)
    return _public()._dedupe(rows)

def _core_search(
    query: str,
    embedding: list[float],
    *,
    file_ids: list[uuid.UUID] | None,
    top_k: int,
    filters: dict | None = None,
    per_file: bool = False,
    diversify_query: str | None = None,
) -> list[Hit]:
    """真正的一轮检索：候选召回 → 重排 → 邻接。"""
    rows = _public()._retrieve_candidates(
        query, embedding, file_ids=file_ids, top_k=top_k, filters=filters
    )
    if not rows:
        return []
    if settings.kb_rerank_enabled and len(rows) > top_k:
        # 只把前 _RERANK_LIMIT 条交给模型：候选再多，提示词变长、延迟也变高，
        # 而排在第 13 名之外的本来也进不了 top_k
        head = rows[:_RERANK_LIMIT]
        rows = _public()._rerank(query, head, top_k) + rows[_RERANK_LIMIT:]
    rows = _public()._diversify(
        diversify_query or query, rows, top_k, per_file=per_file
    )
    rows = _public()._merge_same_table_rows(rows)
    hits = [_public()._to_hit(row) for row in rows[:top_k]]
    return _public()._attach_neighbors(hits)
