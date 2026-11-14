"""关键词扩展：分词、短词相邻短语、停用词过滤。

由原 retrieval.py 拆分而来，逻辑未改动。
"""

from __future__ import annotations

import re
import uuid

from ai_agent.core.config import settings
from ai_agent.storage.store import store

from ai_agent.core.logging import get_logger

from ai_agent.kb.retrieval._impl.types import (
    _STOPWORDS,
)

logger = get_logger(__name__)


# 测试用 monkeypatch.setattr(retrieval, NAME, ...) 打补丁的知识点：
# 直接 NAME(...) 走的是本模块 globals，绕过模块属性，补丁会失效；
# 统一经 _public() 取「对外模块」的当前属性再调用，补丁才生效。
def _public():
    import ai_agent.kb.retrieval.retrieval as public

    return public


def _extract_keywords(query: str) -> list[str]:
    """从查询里提取关键词：jieba TF-IDF 打头的中文词 + 英文/数字 token。

    后者单独补一轮：jieba 会把 "OT-01"、"L3" 这类编号切碎或丢掉，
    而它们恰恰是向量最容易丢、子串匹配最准的东西。
    """
    keywords: list[str] = []
    _public().load_short_words()
    try:
        import jieba.analyse

        keywords.extend(
            jieba.analyse.extract_tags(query, topK=settings.kb_hybrid_keywords)
        )
    except Exception:
        # jieba 加载失败不该拖垮检索，退化为纯向量
        logger.warning("jieba 关键词提取失败，退回纯向量", exc_info=True)
        return []
    keywords.extend(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_.]{1,}", query))
    unique: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        kw = kw.strip()
        if len(kw) < 2 or kw.lower() in _STOPWORDS or kw.lower() in seen:
            continue
        seen.add(kw.lower())
        unique.append(kw)
        if len(unique) >= settings.kb_hybrid_keywords:
            break
    return unique

def _adjacent_short_phrases(query: str) -> list[tuple[str, str, str]]:
    """问句里相邻的两个二字词，拼成一个不超过四字的短词。

    返回 (短词, 左词, 右词)。已经是一个词的（「净利润」「注册资本」，以及
    短词表里的词）不会被拆开再拼。单字和停用词不参与。
    """
    _public().load_short_words()
    try:
        import jieba

        tokens = [token.strip() for token in jieba.cut(query or "") if token.strip()]
    except Exception:
        logger.warning("jieba 短词切分失败，跳过短词", exc_info=True)
        return []
    found: list[tuple[str, str, str]] = []
    index = 0
    while index < len(tokens) - 1:
        left, right = tokens[index], tokens[index + 1]
        if (
            _SHORT_WORD.fullmatch(left)
            and _SHORT_WORD.fullmatch(right)
            and left not in _STOPWORDS
            and right not in _STOPWORDS
            and len(left) + len(right) <= 4
        ):
            found.append((left + right, left, right))
            index += 2
            continue
        index += 1
    return found

def _apply_short_phrases(
    query: str,
    keywords: list[str],
    file_ids: list[uuid.UUID] | None,
) -> list[str]:
    """两个二字在库里以词的形式出现过，就用整段替换拆开的两半。

    后面还紧跟汉字的不算出现过：「合并利润」只是「合并利润表」的前缀，
    仍退回「合并」和「利润」。一个边界命中都没有、过半切片都含有、
    或写在排除名单里的，都不替换。词频取不到时也不替换。
    """
    pairs = _public()._adjacent_short_phrases(query)
    if not pairs:
        return keywords
    try:
        total, freq = store.keyword_boundary_freq(
            [phrase for phrase, _, _ in pairs], file_ids=file_ids
        )
    except Exception:
        logger.warning("kb 短词词频统计失败，保留拆开的词", exc_info=True)
        return keywords
    if total <= 0:
        return keywords
    blocked = _public().read_extra()[1]
    chosen: list[str] = []
    drop: set[str] = set()
    for phrase, left, right in pairs:
        if phrase in blocked:
            continue
        hits = freq.get(phrase, 0)
        if hits <= 0 or hits * 2 > total:
            continue
        chosen.append(phrase)
        drop.update((left, right))
    if not chosen:
        return keywords
    logger.info("kb 关键词改用短词：%s", chosen)
    merged: list[str] = []
    seen: set[str] = set()
    for keyword in [*chosen, *[word for word in keywords if word not in drop]]:
        key = keyword.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(keyword)
        if len(merged) >= settings.kb_hybrid_keywords:
            break
    return merged

# 相邻两个二字词才拼。单字（「前」）不参与，避免「前」+「三季度」变成
# 「前三季度」，对不上正文里的「第三季度」。两边都是二字，加起来正好四字。
_SHORT_WORD = re.compile(r"^[\u4e00-\u9fff]{2}$")
