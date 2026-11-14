"""精排：cross-encoder / LLM 兜底、指标命中加权。

由原 retrieval.py 拆分而来，逻辑未改动。
"""

from __future__ import annotations

import json
import re

from ai_agent.core.config import settings

from ai_agent.core.logging import get_logger

logger = get_logger(__name__)


# 测试用 monkeypatch.setattr(retrieval, NAME, ...) 打补丁的知识点：
# 直接 NAME(...) 走的是本模块 globals，绕过模块属性，补丁会失效；
# 统一经 _public() 取「对外模块」的当前属性再调用，补丁才生效。
def _public():
    import ai_agent.kb.retrieval.retrieval as public

    return public


_rerank_cache: dict[tuple, list[float]] = {}

def _candidate_text(row: dict) -> str:
    """喂给重排器的候选文本：文件名 + 位置 + 单元格文字。

    HTML 标签先去掉再截断。否则 800 字窗口全花在 `<td>` 上，表身还是看不见。
    """
    body = _public().html_table_plain(_public().render_otsl(row.get("text") or ""))
    return f"{row.get('filename') or ''} · {_public()._where(row)}\n{_public()._clip(body)}"

def _llm_rerank_scores(query: str, docs: list[str]) -> list[float]:
    """LLM 重排：让模型对候选编号排序，折算成分数（名次越靠前分越高）。

    仅作为 cross-encoder 不可用时的兜底；分数制与 cross-encoder 后端输出同构，
    `_rerank` 无需感知具体后端。
    """
    listing = "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(docs))
    prompt = _public().build_rerank_user(query, listing, len(docs))
    try:
        answer = _public().complete_chat_once(prompt)
    except Exception:
        logger.warning("kb LLM 重排调用失败", exc_info=True)
        raise
    order = _public()._parse_order(answer, len(docs))
    if not order:
        raise ValueError("kb LLM 重排未解析出有效顺序")
    n = len(docs)
    scores = [0.0] * n
    for rank, idx in enumerate(order):
        if 1 <= idx <= n:
            scores[idx - 1] = float(n - rank)
    return scores

def _rerank(query: str, rows: list[dict], top_k: int) -> list[dict]:
    """重排：调用可插拔 Reranker 对候选打分，按分数降序返回。

    候选序列用 file_id#seq 签名，重排候选变了（范围/阈值不同）即失效，不会串味。
    后端由 settings.kb_rerank_backend 决定（见 ai_agent.kb.rerank）。失败或分数
    数量不符时退回向量顺序，绝不丢候选。
    """
    sig = tuple((r.get("file_id"), r.get("seq")) for r in rows)
    cache_key = (query, sig)
    scores = _rerank_cache.get(cache_key)
    if scores is None:
        docs = [_public()._candidate_text(r) for r in rows]
        try:
            scores = _public().get_reranker().rerank(query, docs)
        except Exception:
            logger.warning("kb 重排调用失败，退回向量顺序", exc_info=True)
            return rows
        if len(scores) != len(rows):
            logger.warning("kb 重排返回分数数量与候选不符，退回向量顺序")
            return rows
        _rerank_cache[cache_key] = scores
        if len(_rerank_cache) > _RERANK_CACHE_MAX:
            # dict 保插入序，弹出最旧一项，避免进程内无限增长（修复旧实现无淘汰）
            _rerank_cache.pop(next(iter(_rerank_cache)))
    adjusted = _public()._boost_indicator_hits(query, rows, list(scores))
    order = sorted(range(len(adjusted)), key=lambda i: adjusted[i], reverse=True)
    return [rows[i] for i in order]

def _boost_indicator_hits(
    query: str, rows: list[dict], scores: list[float]
) -> list[float]:
    """问句里整段出现的行项目，给对应表格切片一点加分。

    只认指标名整段出现在问句里。「营收」不会因此顶上「营业收入」那一行。
    加分是分数的一成多，够把接近的两片换位，不够把明显不相关的片抬上来。
    """
    text = query or ""
    boosted: list[float] = []
    for row, score in zip(rows, scores):
        names = row.get("indicators") or []
        if not isinstance(names, list):
            names = []
        hit = any(
            isinstance(name, str) and len(name) >= 2 and name in text for name in names
        )
        boosted.append(score * 1.12 if hit else score)
    return boosted

def complete_chat_once(prompt: str) -> str:
    """单独抽出便于测试打桩：重排是短任务，模型与超时都独立配置。

    enable_thinking 必须关：主模型带思考链，20 秒的预算全花在推理上，
    实测直接超时；重排只需要输出一组编号，不需要想。
    """
    from ai_agent.llm.client import complete_chat

    return complete_chat(
        [{"role": "user", "content": prompt}],
        model=settings.kb_rerank_model or None,
        timeout=settings.kb_rerank_timeout_seconds,
        max_retries=0,
        max_tokens=120,
        # 重排要的是可复现的排序，不是措辞多样性
        temperature=settings.kb_rerank_temperature,
        extra_body={
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

def _parse_order(answer: str, size: int) -> list[int]:
    match = _JSON_ARRAY.search(answer or "")
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    order: list[int] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, int) and 1 <= item <= size and item not in order:
            order.append(item)
    return order

# 送重排的候选上限：再多提示词就太长，而且第 13 名之后本来就进不了 top_k
_RERANK_LIMIT = 12

_RERANK_CACHE_MAX = 1024

_JSON_ARRAY = re.compile(r"\[[^\[\]]*\]")
