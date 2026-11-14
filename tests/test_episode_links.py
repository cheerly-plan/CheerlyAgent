"""情景记忆关联链接：相似但不合并的条目互相记 id，召回时一跳扩展。"""

from unittest.mock import patch

import pytest

from ai_agent.core.config import settings
from ai_agent.storage.embeddings import embed_query, embed_texts
from ai_agent.storage.store import MemoryStore

UID = "episode-link-user"


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """把文本映射成确定性向量：同主题靠近，不同主题远离。"""
    dim = settings.embedding_dim
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        if "深圳" in text or "华南" in text:
            vec[0] = 1.0
        elif "研报" in text:
            vec[1] = 1.0
        else:
            vec[2] = 1.0
        # 加一点主题内扰动，让相似度落在 link 区间而不是完全相同
        offset = 0.9 if "提醒" in text else 0.0
        vec[(3 if offset else 4)] = offset
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


@pytest.fixture
def link_store(monkeypatch):
    monkeypatch.setattr("ai_agent.memory.episode_store.embed_texts", _fake_embed)
    monkeypatch.setattr("ai_agent.memory.episode_store.embed_query", lambda q: _fake_embed([q])[0])
    monkeypatch.setattr(settings, "memory_link_threshold", 0.4)
    monkeypatch.setattr(settings, "memory_merge_threshold", 0.99)
    store = MemoryStore()
    try:
        yield store
    finally:
        store.clear_episodes(user_id=UID)


def _related(store: MemoryStore, item_id: str) -> list[str]:
    with store._get_pool().connection() as conn:
        row = conn.execute(
            "SELECT related_ids FROM agent_episode WHERE id::text = %s",
            (item_id,),
        ).fetchone()
    return list(row["related_ids"] or []) if row else []


def test_related_episodes_are_linked_both_ways(link_store):
    first = link_store.write_episode("下周要去深圳出差", user_id=UID)
    second = link_store.write_episode("报告提到华南业务时提醒我", user_id=UID)

    assert first.id != second.id, "未达合并阈值，不应合并"
    assert second.id in _related(link_store, first.id)
    assert first.id in _related(link_store, second.id)


def test_unrelated_episodes_are_not_linked(link_store):
    travel = link_store.write_episode("下周要去深圳出差", user_id=UID)
    report = link_store.write_episode("我在做一份研报", user_id=UID)

    assert _related(link_store, travel.id) == []
    assert _related(link_store, report.id) == []


def test_link_disabled_by_zero_threshold(link_store, monkeypatch):
    monkeypatch.setattr(settings, "memory_link_threshold", 0)
    first = link_store.write_episode("下周要去深圳出差", user_id=UID)
    link_store.write_episode("报告提到华南业务时提醒我", user_id=UID)
    assert _related(link_store, first.id) == []


def test_expand_related_returns_one_hop(link_store):
    first = link_store.write_episode("下周要去深圳出差", user_id=UID)
    second = link_store.write_episode("报告提到华南业务时提醒我", user_id=UID)

    hits = [first]
    hits[0] = type(first)(**{**first.__dict__, "related": (second.id,)})
    expanded = link_store.expand_related(hits, user_id=UID, limit=3)
    assert [item.id for item in expanded] == [second.id]


def test_expand_related_skips_already_hit(link_store):
    first = link_store.write_episode("下周要去深圳出差", user_id=UID)
    second = link_store.write_episode("报告提到华南业务时提醒我", user_id=UID)

    hits = [
        type(first)(**{**first.__dict__, "related": (second.id,)}),
        second,
    ]
    assert link_store.expand_related(hits, user_id=UID, limit=3) == []


def test_link_failure_does_not_break_write(link_store):
    """链接是增强：内部查询炸了，写入本身仍然成功。"""
    original = MemoryStore._link_related
    calls = {"n": 0}

    def flaky(self, user_id, item_id, vector):
        calls["n"] += 1
        # 模拟 _link_related 内部 DB 失败：它自己 catch，不能冒泡到写入
        with patch.object(
            MemoryStore, "_get_pool", side_effect=RuntimeError("db down")
        ):
            return original(self, user_id, item_id, vector)

    with patch.object(MemoryStore, "_link_related", flaky):
        item = link_store.write_episode("下周要去深圳出差", user_id=UID)
    assert calls["n"] == 1
    assert item.text == "下周要去深圳出差"
