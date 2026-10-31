import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from openai import OpenAI

logger = get_logger(__name__)

_client = OpenAI(
    base_url=settings.base_url,
    api_key=settings.api_key,
)


# 单次 embedding 调用超时与重试：在线 API 抖动是检索不稳定的根因之一，
# 超时即重试，避免一次偶发失败拖垮整条检索。入库批量调用也会受益。
_EMBED_TIMEOUT = 30.0
_EMBED_RETRIES = 2


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    for attempt in range(_EMBED_RETRIES + 1):
        try:
            resp = _client.embeddings.create(
                input=texts,
                model=settings.embedding_model,
                dimensions=settings.embedding_dim,
                timeout=_EMBED_TIMEOUT,
            )
            break
        except Exception as e:
            if attempt < _EMBED_RETRIES:
                time.sleep(min(0.5 * (attempt + 1), 2.0))
                logger.warning("embed 第 %d 次失败，重试: %s", attempt + 1, e)
                continue
            raise LLMError(
                f"Embed text failed ({settings.embedding_model} @ {settings.base_url}): {e}"
            ) from e
    items = sorted(resp.data, key=lambda x: x.index)
    vectors = [item.embedding for item in items]
    dim = settings.embedding_dim
    if any(len(v) != dim for v in vectors):
        raise LLMError(f"embedding 维度不是 {dim}，请改 EMBEDDING_DIM")
    return vectors

@lru_cache(maxsize=2048)
def _cached_query_vector(query: str) -> tuple[float, ...]:
    """查询向量的进程内缓存：同一句话语义恒定，复用可消除多次 embedding
    调用的微变/超时抖动，并省一次 API 调用。进程生命周期内 embedding 模型
    不变，故无需感知版本；入库路径走 embed_texts 不进此缓存。"""
    return tuple(embed_texts([query])[0])


def embed_query(query: str) -> list[float]:
    return list(_cached_query_vector(query.strip()))


def embed_texts_parallel(
    texts: list[str], *, concurrency: int = 3
) -> list[list[float]]:
    """批间并行的向量化，结果顺序与输入严格一致。

    DashScope 单批上限 10 条（EMBEDDING_BATCH），大文件就是几十个串行
    往返；批与批互相独立，并发后墙钟约缩到 1/concurrency。
    任一批失败照常抛 LLMError（失败语义与 embed_texts 相同，由调用方决定
    是整份 failed 还是跳过）。
    """
    if not texts:
        return []
    batch = max(1, settings.kb_embedding_batch)
    bounds = list(range(0, len(texts), batch))
    if concurrency <= 1 or len(bounds) <= 1:
        return embed_texts(texts)
    vectors: list[list[float] | None] = [None] * len(texts)
    with ThreadPoolExecutor(
        max_workers=min(concurrency, len(bounds)), thread_name_prefix="kb-embed"
    ) as pool:
        futures = [
            pool.submit(embed_texts, texts[start : start + batch])
            for start in bounds
        ]
        # 任一批抛 LLMError 就在这里炸出来，池子退出时等齐在途批次
        for start, future in zip(bounds, futures):
            vectors[start : start + batch] = future.result()
    return vectors  # type: ignore[return-value]