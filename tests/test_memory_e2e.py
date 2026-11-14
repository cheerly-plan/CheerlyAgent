"""记忆端到端：写入 → 注入 → 晋升 → 归档 → 删除。

这个用例不打模型：抽取结果、embedding、情景合并都是确定性的桩，真正跑的是
DB、门控之后的写入、候选晋升、归档蒸馏和管理删除。CI 用 pgvector 服务起一个
空库执行，所以它能在没有模型密钥的 PR 上做阻断门禁。

本地没有可用 Postgres 时跳过；CI 里设置 ``MEMORY_E2E_REQUIRED=1`` 后，建库/
建表失败会直接失败，而不是伪装成 skip 放行。
"""

from __future__ import annotations

import os

import pytest

from ai_agent.agents.loop import _profile_prompt, _recall_block
from ai_agent.core.config import settings
from ai_agent.memory import long_term
from ai_agent.memory.archive import distill_memory
from ai_agent.memory.episode_merge import merge_episode_texts
from ai_agent.memory.promotion import observe_episode, review_pending_candidates
from ai_agent.storage.db import MemoryItem
from ai_agent.storage.store import MemoryStore

UID = "memory_e2e_user"
PROFILE_TEXT = "用户偏好简洁的中文回答"
EPISODE_TEXT = "在做记忆模块的端到端评测"

_REQUIRED = os.environ.get("MEMORY_E2E_REQUIRED") == "1"


def _fake_embed(texts: list[str]) -> list[list[float]]:
    dim = settings.embedding_dim
    vecs: list[list[float]] = []
    for text in texts:
        v = [0.0] * dim
        for word in text.lower().split():
            v[hash(word) % dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        vecs.append([x / norm for x in v])
    return vecs


def _cleanup(s: MemoryStore) -> None:
    s.clear_long_term(user_id=UID)
    with s._get_pool().connection() as conn:
        conn.execute("DELETE FROM agent_memory_setting WHERE user_id = %s", (UID,))
        conn.commit()


@pytest.fixture()
def mem_store(monkeypatch):
    monkeypatch.setattr("ai_agent.memory.episode_store.embed_texts", _fake_embed)
    monkeypatch.setattr(
        "ai_agent.memory.episode_store.embed_query", lambda q: _fake_embed([q])[0]
    )
    monkeypatch.setattr(
        "ai_agent.memory.episode_store.compact_episode_texts",
        lambda parts, keeper="": merge_episode_texts(parts),
    )
    s = MemoryStore()
    try:
        s.ensure_schema()
    except Exception as exc:  # pragma: no cover - 无库路径
        if _REQUIRED:
            pytest.fail(f"记忆 E2E 要求可用 Postgres：{exc}")
        pytest.skip(f"Postgres 不可用: {exc}")
    _cleanup(s)
    yield s
    _cleanup(s)
    s.close()


def _episode_row(s: MemoryStore) -> dict:
    rows, _ = s.list_episodes_page(UID, include_expired=True)
    assert len(rows) == 1
    return rows[0]


def _seed_evidence(s: MemoryStore, episode_id: str) -> None:
    with s._get_pool().connection() as conn:
        conn.execute(
            """
            UPDATE agent_episode
            SET hit_count = 5,
                sessions_seen = ARRAY['s1','s2']::text[],
                last_seen_at = now()
            WHERE user_id = %s AND id::text = %s
            """,
            (UID, episode_id),
        )
        conn.commit()


def test_memory_lifecycle_write_inject_promote_archive_delete(mem_store, monkeypatch):
    # 1) 写入：走真实的 capture → 门控 → 抽取结果 → profile/episode 落库
    monkeypatch.setattr(long_term, "_gate_blocks", lambda *a, **k: False)
    monkeypatch.setattr(
        long_term,
        "extract_turn_memories",
        lambda *a, **k: [
            long_term.ExtractedMemory(
                kind="profile",
                op="add",
                facet="preference",
                content=PROFILE_TEXT,
            ),
            long_term.ExtractedMemory(kind="episode", content=EPISODE_TEXT),
        ],
    )
    long_term.capture_long_term(
        UID,
        [
            {"role": "user", "content": "我偏好简洁回答，也在做记忆评测"},
            {"role": "assistant", "content": "好的"},
        ],
        "",
        "我偏好简洁回答，也在做记忆评测",
        chat_session_id="s1",
    )

    assert any(item.text == PROFILE_TEXT for item in mem_store.list_profile(UID))
    episode = _episode_row(mem_store)

    # 2) 注入：画像索引 + 自动召回都能看到刚写入的内容
    prompt = _profile_prompt(UID, user_input="记忆评测")
    assert PROFILE_TEXT in prompt
    recall = _recall_block(UID, EPISODE_TEXT)
    assert EPISODE_TEXT in recall

    # 3) 晋升：补足跨会话证据 → 观察态产出候选 → 证据达标自动升 L1
    _seed_evidence(mem_store, episode["id"])
    decision = observe_episode(
        MemoryItem(
            id=episode["id"],
            user_id=UID,
            text=episode["text"],
            created_at=episode["created_at"],
        ),
        user_id=UID,
        chat_session_id="s3",
    )
    assert decision is not None and decision.stage == "candidate"
    pending = mem_store.list_candidates(UID, status="pending")
    assert [row["text"] for row in pending] == [EPISODE_TEXT]
    assert review_pending_candidates(user_id=UID) == 1
    assert not [
        row for row in mem_store.list_candidates(UID, status="pending")
        if row["text"] == EPISODE_TEXT
    ]
    assert any(item.text == EPISODE_TEXT for item in mem_store.list_profile(UID))

    # 4) 归档：已晋升候选 + 稳定 episode 都进 L4 只读副本
    report = distill_memory(user_id=UID)
    assert report["archived"] == 2
    archive = mem_store.list_archive_rows(UID)
    assert {row["kind"] for row in archive} == {"promoted", "episode"}
    # 归档来源是「已晋升候选 + 稳定 episode」；本例两者都来自同一条近况，
    # 直接写入的 PROFILE_TEXT 仍只属于 L1，不会被复制成候选归档。
    assert {row["text"] for row in archive} == {EPISODE_TEXT}

    # 5) 删除：清空全部长期记忆后，四层都不可见
    mem_store.clear_long_term(user_id=UID)
    assert mem_store.list_profile_page(UID, include_inactive=True)[0] == []
    assert mem_store.list_episodes_page(UID, include_expired=True)[0] == []
    assert mem_store.list_candidates(UID, status="pending") == []
    assert mem_store.list_candidates(UID, status="promoted") == []
    assert mem_store.list_archive_rows(UID) == []