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
        raise LLMError(f"Embed text failed: {e}") from e
    items = sorted(resp.data, key=lambda x: x.index)
    vectors = [item.embedding for item in items]
    dim = settings.embedding_dim
    if any(len(v) != dim for v in vectors):
        raise LLMError(f"embedding 维度不是 {dim}，请改 EMBEDDING_DIM")
    return vectors

def embed_query(query:str) -> list[float]:
    return embed_texts([query])[0]