"""画像冲突消解与时效：新事实覆盖旧事实，易变事实软过期。"""

import pytest

from ai_agent.core.config import settings
from ai_agent.memory.profile import (
    ProfileAction,
    _profile_expires_at,
    apply_profile_action,
    find_superseded,
    profile_overlap,
)
from ai_agent.storage.db import MemoryItem, user_scope
from ai_agent.storage.store import MemoryStore

UID = "profile_supersede_user"


@pytest.fixture()
def mem_store():
    store = MemoryStore()
    try:
        store.ensure_schema()
    except Exception as e:  # pragma: no cover - 无库时跳过
        pytest.skip(f"Postgres 不可用: {e}")
    with store._get_pool().connection() as conn:
        conn.execute("DELETE FROM agent_user_profile WHERE user_id = %s", (UID,))
        conn.commit()
    with user_scope(UID):
        yield store
    with store._get_pool().connection() as conn:
        conn.execute("DELETE FROM agent_user_profile WHERE user_id = %s", (UID,))
        conn.commit()
    store.close()


def _texts(store: MemoryStore, *, active: bool = True) -> list[str]:
    items = store.list_profile(UID, include_inactive=not active)
    return sorted(item.text for item in items)


def test_overlap_helper_is_symmetric_enough():
    assert profile_overlap("财报数据默认用万元", "财报数据默认用亿元") > 0.6
    assert profile_overlap("财报数据默认用万元", "不喜欢用表格") < 0.3


def test_find_superseded_picks_same_topic_variants():
    existing = [
        MemoryItem(id="1", user_id=UID, text="财报数据默认用万元", created_at="", facet="preference"),
        MemoryItem(id="2", user_id=UID, text="不喜欢用表格", created_at="", facet="preference"),
    ]
    hits = find_superseded("以后财报数据都用亿元", "preference", existing)
    assert [item.id for item in hits] == ["1"]


def test_identical_text_is_not_superseded():
    """同一条重写是幂等刷新，不该把自己标成失效。"""
    existing = [
        MemoryItem(id="1", user_id=UID, text="财报数据默认用万元", created_at="", facet="preference")
    ]
    assert find_superseded("财报数据默认用万元", "preference", existing) == []


def test_new_fact_supersedes_old_one(mem_store):
    apply_profile_action(ProfileAction("add", "preference", "财报数据默认用万元"))
    apply_profile_action(ProfileAction("add", "preference", "以后财报数据都用亿元"))
    assert _texts(mem_store) == ["以后财报数据都用亿元"]
    # 旧条目留在库里供审计
    assert "财报数据默认用万元" in _texts(mem_store, active=False)


@pytest.mark.parametrize(
    ("facet", "old", "new"),
    [
        ("name", "张三", "张三丰"),
        ("occupation", "前端开发", "资深前端开发"),
        ("locale", "北京", "北京市"),
        ("language", "中文", "简体中文"),
    ],
)
def test_mutex_facet_update_keeps_new_value_active(mem_store, facet, old, new):
    """互斥 facet 用同一 id 覆盖，不能再把覆盖后的同一行标成 superseded。

    回归：曾经 occupation「前端开发」→「资深前端开发」写入成功后，
    list_profile 立刻返回空——新旧行是同一个 id，supersede 把新值也标掉了。
    """
    apply_profile_action(ProfileAction("add", facet, old))
    apply_profile_action(ProfileAction("update", facet, new))

    active = mem_store.list_profile(UID)
    assert [item.text for item in active] == [new]
    with mem_store._get_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT superseded_at
            FROM agent_user_profile
            WHERE user_id = %s AND facet = %s
            """,
            (UID, facet),
        ).fetchone()
    assert row is not None
    assert row["superseded_at"] is None


def test_unrelated_preference_coexists(mem_store):
    apply_profile_action(ProfileAction("add", "preference", "财报数据默认用万元"))
    apply_profile_action(ProfileAction("add", "preference", "不喜欢用表格回答"))
    assert sorted(_texts(mem_store)) == ["不喜欢用表格回答", "财报数据默认用万元"]


def test_readding_superseded_text_revives_it(mem_store):
    apply_profile_action(ProfileAction("add", "preference", "财报数据默认用万元"))
    apply_profile_action(ProfileAction("add", "preference", "以后财报数据都用亿元"))
    apply_profile_action(ProfileAction("add", "preference", "财报数据默认用万元"))
    assert "财报数据默认用万元" in _texts(mem_store)


def test_volatile_facet_expires_and_stable_facet_does_not(mem_store):
    assert _profile_expires_at("occupation") is not None
    assert _profile_expires_at("locale") is not None
    assert _profile_expires_at("preference") is None
    assert _profile_expires_at("name") is None


def test_expired_fact_is_hidden_but_kept(mem_store):
    apply_profile_action(ProfileAction("add", "occupation", "半导体行业研究员"))
    with mem_store._get_pool().connection() as conn:
        conn.execute(
            "UPDATE agent_user_profile SET expires_at = now() - interval '1 day' "
            "WHERE user_id = %s",
            (UID,),
        )
        conn.commit()
    assert _texts(mem_store) == []
    assert _texts(mem_store, active=False) == ["半导体行业研究员"]
