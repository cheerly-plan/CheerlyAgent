from concurrent.futures import ThreadPoolExecutor

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from openai import OpenAI

_client = OpenAI(
    base_url=settings.base_url,
    api_key=settings.api_key,
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        resp = _client.embeddings.create(
            input=texts,
            model=settings.embedding_model,
            dimensions=settings.embedding_dim,
        )
    except Exception as e:
        raise LLMError(
            f"Embed text failed ({settings.embedding_model} @ {settings.base_url}): {e}"
        ) from e
    items = sorted(resp.data, key=lambda x: x.index)
    vectors = [item.embedding for item in items]
    dim = settings.embedding_dim
    if any(len(v) != dim for v in vectors):
        raise LLMError(f"embedding 维度不是 {dim}，请改 EMBEDDING_DIM")
    return vectors

def embed_query(query:str) -> list[float]:
    return embed_texts([query])[0]


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