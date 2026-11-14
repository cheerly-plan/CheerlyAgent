"""RRF 多路融合：按名次融合，不依赖分数量纲。"""

from ai_agent.core.config import settings
from ai_agent.memory.progressive import RRF_K, fuse_rrf, rerank_items
from ai_agent.storage.db import MemoryItem


def _item(id_: str, text: str = "文本", score: float = 0.0) -> MemoryItem:
    return MemoryItem(
        id=id_,
        user_id="u1",
        text=text,
        created_at="2026-11-14T00:00:00+00:00",
        score=score,
    )


def test_fuse_rrf_rewards_items_in_both_lists():
    a, b, c = _item("a"), _item("b"), _item("c")
    fused = fuse_rrf([[a, b], [a, c]])
    # a 在两路都排第一，必然最靠前
    assert fused[0].id == "a"
    assert {item.id for item in fused} == {"a", "b", "c"}


def test_fuse_rrf_uses_rank_not_raw_score():
    """分数量纲不可比：原始 score 极小的条目靠名次仍能胜出。"""
    tiny = _item("tiny", score=0.001)
    huge = _item("huge", score=999.0)
    # 两路都把 tiny 排第一；huge 原始分再高也不参与比较
    fused = fuse_rrf([[tiny, huge], [tiny, huge]], top_k=2)
    assert fused[0].id == "tiny"


def test_fuse_rrf_respects_weights():
    a, b = _item("a"), _item("b")
    # a 只在权重高的向量路出现
    fused = fuse_rrf([[a], [b]], weights=[1.0, 0.1])
    assert fused[0].id == "a"


def test_fuse_rrf_zero_weight_drops_path():
    a, b = _item("a"), _item("b")
    fused = fuse_rrf([[a], [b]], weights=[1.0, 0.0])
    assert [item.id for item in fused] == ["a"]


def test_fuse_rrf_top_k_and_empty():
    assert fuse_rrf([]) == []
    fused = fuse_rrf([[_item("a"), _item("b"), _item("c")]], top_k=2)
    assert len(fused) == 2


def test_fuse_rrf_score_matches_formula():
    a = _item("a")
    fused = fuse_rrf([[a], [a]])
    assert abs(fused[0].score - 2 * (1 / (RRF_K + 1))) < 1e-9


def test_rerank_promotes_item_that_wins_both_paths(monkeypatch):
    """同时被向量路和词面路排第一的条目，融合后必须最靠前。"""
    monkeypatch.setattr(settings, "memory_fusion", "rrf")
    monkeypatch.setattr(settings, "memory_transient_penalty", 1.0)
    items = [
        _item("both", "合同负债 100 万"),
        _item("vec", "别的主题"),
        _item("lex", "无关近况"),
    ]
    # both 既在向量路第一，也因字面命中排词面路第一
    ranked = rerank_items(items, "合同负债", top_k=3)
    assert ranked[0].id == "both"


def test_rerank_weight_decides_single_path_winner(monkeypatch):
    """只在一路领先时，由该路权重决定：词面权重调高则字面命中胜出。"""
    monkeypatch.setattr(settings, "memory_fusion", "rrf")
    monkeypatch.setattr(settings, "memory_transient_penalty", 1.0)
    monkeypatch.setattr(settings, "memory_fusion_weight_keyword", 2.0)
    items = [
        _item("v1", "完全不相关的旧近况"),
        _item("lex", "合同负债 100 万"),
    ]
    ranked = rerank_items(items, "合同负债", top_k=2)
    assert ranked[0].id == "lex"


def test_rerank_promotes_lexical_hit_above_pure_vector_noise(monkeypatch):
    """字面强命中虽然只在一路第一，仍应排在「两路都靠后」的噪声之前。"""
    monkeypatch.setattr(settings, "memory_fusion", "rrf")
    monkeypatch.setattr(settings, "memory_transient_penalty", 1.0)
    items = [
        _item("v1", "无关的近况甲"),
        _item("v2", "无关的近况乙"),
        _item("lex", "合同负债 100 万"),
        _item("tail", "无关的近况丙"),
    ]
    ranked = rerank_items(items, "合同负债", top_k=4)
    lex_pos = [item.id for item in ranked].index("lex")
    assert lex_pos <= 1


def test_rerank_linear_mode_still_available(monkeypatch):
    monkeypatch.setattr(settings, "memory_fusion", "linear")
    items = [_item("a", "合同负债"), _item("b", "无关")]
    ranked = rerank_items(items, "合同负债", top_k=2)
    assert ranked[0].id == "a"


def test_rerank_applies_transient_penalty(monkeypatch):
    monkeypatch.setattr(settings, "memory_fusion", "rrf")
    monkeypatch.setattr(settings, "memory_transient_penalty", 0.1)
    noisy = _item("noise", "用户正在查询中国银行股东信息")
    stable = _item("stable", "财报数字以亿元为单位")
    ranked = rerank_items([noisy, stable], "财报数字", top_k=2)
    assert ranked[0].id == "stable"
