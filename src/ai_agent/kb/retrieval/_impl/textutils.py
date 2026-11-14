"""文本小工具：截断、三字组、Jaccard 相似度。

由原 retrieval.py 拆分而来，逻辑未改动。
"""

from __future__ import annotations

import re

from ai_agent.kb.ingest.chunking import TABLE_GROUP_CHARS


# 测试用 monkeypatch.setattr(retrieval, NAME, ...) 打补丁的知识点：
# 直接 NAME(...) 走的是本模块 globals，绕过模块属性，补丁会失效；
# 统一经 _public() 取「对外模块」的当前属性再调用，补丁才生效。
def _public():
    import ai_agent.kb.retrieval.retrieval as public

    return public


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _RERANK_CLIP else f"{text[:_RERANK_CLIP]}…"

def _trigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i : i + 3] for i in range(len(compact) - 2)}

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0

# 送重排的正文上限，与 HTML 表行组是同一个字数：一组表重排能看完，
# 不会停在表头。正文切片本来就短于这个上限，放宽后整段也能读完。
_RERANK_CLIP = TABLE_GROUP_CHARS
