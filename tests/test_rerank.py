"""重排后端可插拔改造的单元测试。"""

import pytest

from ai_agent.kb.retrieval import retrieval


@pytest.fixture(autouse=True)
def _clear_rerank_cache():
    # _rerank_cache 是模块级缓存（按 query+候选签名命中），跨测试必须清空，
    # 否则相同 query/候选的用例会命中上一条用例的缓存、根本不调重排器
    retrieval._rerank_cache.clear()
    yield
    retrieval._rerank_cache.clear()


class _FakeReranker:
    def __init__(self, scores):
        self._scores = list(scores)

    def rerank(self, query, docs):
        assert len(docs) == len(self._scores), "docs/scores 长度不一致"
        return list(self._scores)


class _BoomReranker:
    def rerank(self, query, docs):
        raise RuntimeError("reranker down")


class _CountingReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query, docs):
        self.calls += 1
        return [0.1, 0.9, 0.5]


def _rows(n):
    return [
        {
            "file_id": "f",
            "seq": i,
            "filename": f"文件{i}.pdf",
            "text": f"正文{i}",
            "distance": 0.3,
        }
        for i in range(n)
    ]


def test_rerank_orders_by_score(monkeypatch):
    monkeypatch.setattr(retrieval, "get_reranker", lambda: _FakeReranker([0.1, 0.9, 0.5]))
    ordered = retrieval._rerank("q", _rows(3), top_k=3)
    assert [r["seq"] for r in ordered] == [1, 2, 0]


def test_rerank_falls_back_on_error(monkeypatch):
    monkeypatch.setattr(retrieval, "get_reranker", lambda: _BoomReranker())
    rows = _rows(3)
    ordered = retrieval._rerank("q", rows, top_k=3)
    # 后端抛错 → 退回向量顺序（原顺序），绝不丢候选
    assert [r["seq"] for r in ordered] == [0, 1, 2]


def test_rerank_cache_is_consistent(monkeypatch):
    counter = _CountingReranker()
    monkeypatch.setattr(retrieval, "get_reranker", lambda: counter)
    rows = _rows(3)
    a = retrieval._rerank("q", rows, top_k=3)
    b = retrieval._rerank("q", rows, top_k=3)
    assert [r["seq"] for r in a] == [r["seq"] for r in b]
    assert counter.calls == 1  # 第二次走缓存，不重复重排


def test_llm_rerank_scores_maps_order_to_scores(monkeypatch):
    # 模型返回 [3,1,2]：第 3 名最相关 → 分数最高（3），其次第 1 名（2）、第 2 名（1）
    monkeypatch.setattr(retrieval, "complete_chat_once", lambda prompt: "[3, 1, 2]")
    scores = retrieval._llm_rerank_scores("q", ["d0", "d1", "d2"])
    assert scores[2] == 3.0
    assert scores[0] == 2.0
    assert scores[1] == 1.0
