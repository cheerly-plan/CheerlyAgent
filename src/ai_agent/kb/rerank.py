"""可插拔重排后端。

历史实现把重排写死成「调一次大模型、让模型输出候选编号序」（`LlmReranker`）。
每次检索都多烧一次生成模型、延迟高、且依赖生成模型的确定性。这里把重排抽象成
`Reranker.rerank(query, docs) -> list[float]`（分数制，高=相关）：

- `CrossEncoderReranker`：本地 bge-reranker-v2-m3 cross-encoder，便宜 1~2 个
  数量级、延迟低、完全确定，作为主推后端；
- `LlmReranker`：保留原 LLM 重排逻辑，作为兜底；

由 `settings.kb_rerank_backend` 选择（默认 "llm" 保持兼容）。`FlagEmbedding`/`torch`
较重，仅在选中 local_ce 且真正调用时才 import，默认 llm 后端无需安装这些依赖。
"""

from __future__ import annotations

import logging

from ai_agent.core.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    def rerank(self, query: str, docs: list[str]) -> list[float]:
        raise NotImplementedError


class LlmReranker(Reranker):
    """原 LLM 重排：让模型输出候选编号序，再折算成分数。兜底保留。"""

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        from ai_agent.kb import retrieval

        return retrieval._llm_rerank_scores(query, docs)


class CrossEncoderReranker(Reranker):
    """本地 bge-reranker-v2-m3 cross-encoder。懒加载单例，确定性打分。

    `FlagEmbedding`/`torch` 较重，仅在选中 local_ce 且真正调用时才 import，
    默认 llm 后端无需安装这些依赖。模型权重跨调用复用模块级单例，避免每次
    检索重复加载。
    """

    # 加载好的 FlagReranker 单例，跨 CrossEncoderReranker 实例复用
    _shared = None

    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.kb_rerank_local_model

    @property
    def ranker(self):
        if CrossEncoderReranker._shared is None:
            import os

            # 权重已缓存到本地（首次经 HF_ENDPOINT=https://hf-mirror.com 拉取）。
            # 受限网络下 huggingface.co 的元数据校验可能长时间挂起而非秒回错误，
            # 导致整个重排请求卡死。强制离线加载：只读本地缓存、绝不触网，从根上
            # 消除网络挂起；若本地确实无权重则快速抛错，由 _rerank 的 try/except 兜底。
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as e:  # 依赖可选，缺了给清晰报错
                raise RuntimeError(
                    "本地重排需要 FlagEmbedding（uv add --group rerank FlagEmbedding），"
                    "或把 kb_rerank_backend 设回 llm"
                ) from e
            import torch

            # use_fp16 只在 GPU 有意义；CPU 上 False 更稳，避免 half 不支持
            CrossEncoderReranker._shared = FlagReranker(
                self._model, use_fp16=torch.cuda.is_available()
            )
        return CrossEncoderReranker._shared

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        pairs = [[query, d] for d in docs]
        return [float(s) for s in self.ranker.compute_score(pairs, normalize=True)]


# 首次选择的后端实例缓存：settings 在进程内固定，避免每次检索重建/重判。
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """返回当前配置选中的重排后端；local_ce 加载失败时自动退回 llm。"""
    global _reranker
    if _reranker is not None:
        return _reranker
    backend = (settings.kb_rerank_backend or "llm").lower()
    if backend == "local_ce":
        try:
            _reranker = CrossEncoderReranker()
        except Exception:  # 依赖缺失/模型加载失败 → 退回 LLM，不阻断检索
            logger.warning("本地重排加载失败，退回 LLM 重排", exc_info=True)
    if _reranker is None:
        _reranker = LlmReranker()
    return _reranker
