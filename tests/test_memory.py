import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from ai_agent.core.config import settings
from ai_agent.memory.episode_merge import merge_episode_text, merge_episode_texts
from ai_agent.memory.store import MemoryStore


def fake_embed(texts: list[str]) -> list[list[float]]:
    dim = settings.embedding_dim
    vecs: list[list[float]] = []
    for text in texts:
        v = [0.0] * dim
        for word in text.lower().split():
            v[hash(word) % dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        vecs.append([x / norm for x in v])
    return vecs


_TEST_USERS = ("a", "b")


def _episode_texts(store: MemoryStore, user_id: str) -> list[str]:
    with store._get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT text
            FROM agent_episode
            WHERE user_id = %s
            ORDER BY created_at, text
            """,
            (user_id,),
        ).fetchall()
    return [row["text"] for row in rows]


def _chat_seqs(store: MemoryStore, user_id: str, chat_session_id: str) -> list[int]:
    with store._get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT seq
            FROM agent_chat_history
            WHERE user_id = %s AND chat_session_id = %s
            ORDER BY seq
            """,
            (user_id, chat_session_id),
        ).fetchall()
    return [int(row["seq"]) for row in rows]


def _clear_test_users(store: MemoryStore) -> None:
    for uid in _TEST_USERS:
        store.clear_long_term(user_id=uid)
        store.clear_chat(uid)
        store.clear_sessions(uid)


@pytest.fixture
def mem_store(monkeypatch):
    monkeypatch.setattr(settings, "memory_merge_threshold", 0.6)
    monkeypatch.setattr("ai_agent.memory.episode_store.embed_texts", fake_embed)
    monkeypatch.setattr(
        "ai_agent.memory.episode_store.embed_query", lambda q: fake_embed([q])[0]
    )
    monkeypatch.setattr(
        "ai_agent.memory.episode_store.compact_episode_texts", merge_episode_texts
    )
    store = MemoryStore()
    try:
        store.ensure_schema()
    except Exception as e:
        pytest.skip(f"Postgres 不可用: {e}")
    _clear_test_users(store)
    yield store
    _clear_test_users(store)
    store.close()


def test_vector_search_is_semantic(mem_store):
    mem_store.write_episode("我日常用 pytest 写单测", user_id="a")
    hits = mem_store.search_episode("pytest 单测", user_id="a")
    assert hits
    assert "pytest" in hits[0].text


def test_store_isolates_users(mem_store):
    mem_store.write_episode("喜欢 pytest", user_id="a")
    mem_store.write_episode("喜欢 jest", user_id="b")
    hits = mem_store.search_episode("pytest", user_id="a")
    assert all(item.user_id == "a" for item in hits)
    assert all("jest" not in item.text for item in hits)


def test_merge_episode_text_skips_contained_or_duplicate():
    assert merge_episode_text("最近在学习 Node.js BFF", "最近在学习 node.js bff") == (
        "最近在学习 Node.js BFF"
    )
    assert merge_episode_text("学习 Node.js BFF 使用 Koa 框架", "学习 Node.js BFF") == (
        "学习 Node.js BFF 使用 Koa 框架"
    )
    assert merge_episode_text("学习 Node.js BFF", "学习 Node.js BFF 使用 Koa 框架") == (
        "学习 Node.js BFF 使用 Koa 框架"
    )


def test_merge_episode_texts_combines_related_facts():
    merged = merge_episode_texts(
        [
            "最近在学习nodejs bff",
            "学习 Node.js BFF 使用 Koa 框架",
            "用 Koa 实现 BFF 数据聚合",
        ]
    )
    assert "nodejs" in merged.lower() or "node.js" in merged.lower()
    assert "koa" in merged.lower()
    assert "数据聚合" in merged
    assert merged.count("；") == 2


def test_write_episode_dedup(mem_store):
    first = mem_store.write_episode("喜欢 uv", user_id="a")
    second = mem_store.write_episode("喜欢 uv", user_id="a")
    assert first.id == second.id
    assert _episode_texts(mem_store, "a") == ["喜欢 uv"]


def test_write_episode_merges_similar(mem_store):
    first = mem_store.write_episode("learning nodejs bff koa", user_id="a")
    second = mem_store.write_episode("learning nodejs bff aggregation", user_id="a")
    assert second.id == first.id
    texts = _episode_texts(mem_store, "a")
    assert len(texts) == 1
    assert "koa" in texts[0]
    assert "aggregation" in texts[0]


def test_write_episode_stores_compacted_sentence(mem_store, monkeypatch):
    monkeypatch.setattr(
        "ai_agent.memory.episode_store.compact_episode_texts",
        lambda parts: "learning nodejs bff with koa aggregation",
    )
    first = mem_store.write_episode("learning nodejs bff koa", user_id="a")
    second = mem_store.write_episode("learning nodejs bff aggregation", user_id="a")
    assert second.id == first.id
    assert _episode_texts(mem_store, "a") == [
        "learning nodejs bff with koa aggregation"
    ]


def test_write_episode_keeps_unrelated(mem_store):
    first = mem_store.write_episode("learning nodejs bff koa", user_id="a")
    second = mem_store.write_episode("bought a piano yesterday", user_id="a")
    assert first.id != second.id
    assert len(_episode_texts(mem_store, "a")) == 2


def test_profile_is_listed_without_embedding(mem_store):
    mem_store.write_profile("喜欢 pytest", user_id="a")
    mem_store.write_episode("上周用 pytest 修过登录", user_id="a")
    profile = mem_store.list_profile("a")
    assert [item.text for item in profile] == ["喜欢 pytest"]


def test_search_episode_does_not_return_profile(mem_store):
    mem_store.write_profile("喜欢 pytest", user_id="a")
    hits = mem_store.search_episode("pytest", user_id="a")
    assert hits == []


def test_profile_isolates_users(mem_store):
    mem_store.write_profile("喜欢 pytest", user_id="a")
    mem_store.write_profile("喜欢 jest", user_id="b")
    profile = mem_store.list_profile("a")
    assert [item.text for item in profile] == ["喜欢 pytest"]


def test_profile_write_dedup(mem_store):
    first = mem_store.write_profile("喜欢 uv", user_id="a")
    second = mem_store.write_profile("喜欢 uv", user_id="a")
    assert first.id == second.id


def test_write_profile_replaces_same_facet(mem_store):
    first = mem_store.write_profile("我是画家", user_id="a", facet="occupation")
    second = mem_store.write_profile("我是后端工程师", user_id="a", facet="occupation")
    assert first.id == second.id
    profile = mem_store.list_profile("a")
    assert [item.text for item in profile] == ["我是后端工程师"]
    assert profile[0].facet == "occupation"


def test_write_profile_keeps_different_facets(mem_store):
    mem_store.write_profile("我是后端", user_id="a", facet="occupation")
    mem_store.write_profile("喜欢 pytest", user_id="a", facet="preference")
    texts = {item.facet: item.text for item in mem_store.list_profile("a")}
    assert texts == {"occupation": "我是后端", "preference": "喜欢 pytest"}


def test_write_profile_accumulates_preference(mem_store):
    """累积型 facet：不同偏好各自一行，互不覆盖。"""
    mem_store.write_profile("喜欢简洁回答", user_id="a", facet="preference")
    mem_store.write_profile("对历史话题感兴趣", user_id="a", facet="preference")
    mem_store.write_profile("周末不想被打扰", user_id="a", facet="preference")
    texts = [item.text for item in mem_store.list_profile("a")]
    assert sorted(texts) == sorted(
        ["喜欢简洁回答", "对历史话题感兴趣", "周末不想被打扰"]
    )


def test_write_profile_preference_same_text_idempotent(mem_store):
    first = mem_store.write_profile("喜欢简洁回答", user_id="a", facet="preference")
    second = mem_store.write_profile("喜欢简洁回答", user_id="a", facet="preference")
    assert first.id == second.id
    assert len(mem_store.list_profile("a")) == 1


def test_delete_single_preference_keeps_others(mem_store):
    mem_store.write_profile("喜欢简洁回答", user_id="a", facet="preference")
    mem_store.write_profile("对历史话题感兴趣", user_id="a", facet="preference")
    deleted = mem_store.delete_profile_facet(
        "preference", user_id="a", text="喜欢简洁回答"
    )
    assert deleted is True
    texts = [item.text for item in mem_store.list_profile("a")]
    assert texts == ["对历史话题感兴趣"]


def test_delete_profile_facet(mem_store):
    mem_store.write_profile("住北京", user_id="a", facet="locale")
    assert mem_store.delete_profile_facet("locale", user_id="a") is True
    assert mem_store.list_profile("a") == []


def test_chat_roundtrip_keeps_tool_calls(mem_store):
    mem_store.append_chat(
        "a",
        "s1",
        [
            {"role": "system", "content": "不应入库"},
            {"role": "user", "content": "北京天气"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "get_weather",
                        "arguments": {"city": "北京"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "get_weather",
                "content": "20度",
            },
            {"role": "assistant", "content": "20度"},
        ],
    )
    loaded = mem_store.load_chat("a", "s1")
    assert [m["role"] for m in loaded] == ["user", "assistant", "tool", "assistant"]
    assert loaded[1]["tool_calls"][0]["name"] == "get_weather"
    assert loaded[1]["tool_calls"][0]["arguments"] == {"city": "北京"}
    assert loaded[2]["tool_call_id"] == "c1"
    assert loaded[2]["content"] == "20度"


def test_chat_append_is_incremental(mem_store):
    mem_store.append_chat("a", "s1", [{"role": "user", "content": "1"}])
    mem_store.append_chat("a", "s1", [{"role": "assistant", "content": "a1"}])
    loaded = mem_store.load_chat("a", "s1")
    assert [m["content"] for m in loaded] == ["1", "a1"]
    assert _chat_seqs(mem_store, "a", "s1") == [1, 2]


def test_chat_isolates_sessions(mem_store):
    mem_store.append_chat("a", "s1", [{"role": "user", "content": "会话1"}])
    mem_store.append_chat("a", "s2", [{"role": "user", "content": "会话2"}])
    mem_store.append_chat("b", "s1", [{"role": "user", "content": "用户b"}])
    assert [m["content"] for m in mem_store.load_chat("a", "s1")] == ["会话1"]
    assert [m["content"] for m in mem_store.load_chat("a", "s2")] == ["会话2"]
    assert [m["content"] for m in mem_store.load_chat("b", "s1")] == ["用户b"]


def test_chat_keeps_duplicate_content(mem_store):
    mem_store.append_chat(
        "a",
        "s1",
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好"},
        ],
    )
    loaded = mem_store.load_chat("a", "s1")
    assert len(loaded) == 4
    assert _chat_seqs(mem_store, "a", "s1") == [1, 2, 3, 4]


def test_chat_skips_system_only(mem_store):
    mem_store.append_chat("a", "s1", [{"role": "system", "content": "sys"}])
    assert mem_store.load_chat("a", "s1") == []


def test_load_recent_turns_keeps_complete_tool_chain(mem_store):
    mem_store.append_chat(
        "a",
        "s1",
        [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "天气"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "get_weather",
                "content": "20度",
            },
            {"role": "assistant", "content": "20度"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "a3"},
        ],
    )
    loaded = mem_store.load_recent_turns("a", "s1", turns=2)
    assert [m["content"] for m in loaded if m["role"] != "tool"] == [
        "天气",
        "",
        "20度",
        "3",
        "a3",
    ]
    assert [m["role"] for m in loaded] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
    ]


def test_session_summary_roundtrip_and_clears_with_chat(mem_store):
    session = mem_store.create_session("a", "sum-1")
    assert mem_store.get_session_summary("a", session.id) == ""
    mem_store.update_session_summary("a", session.id, "用户在问天气")
    assert mem_store.get_session_summary("a", session.id) == "用户在问天气"
    mem_store.clear_chat("a", session.id)
    assert mem_store.get_session_summary("a", session.id) == ""


def test_load_chat_limit_returns_recent(mem_store):
    mem_store.append_chat(
        "a",
        "s1",
        [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "a2"},
        ],
    )
    loaded = mem_store.load_chat("a", "s1", limit=2)
    assert [m["content"] for m in loaded] == ["2", "a2"]


def test_load_chat_page_returns_recent_and_cursor(mem_store):
    mem_store.append_chat(
        "a",
        "s1",
        [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "a3"},
        ],
    )
    page = mem_store.load_chat_page("a", "s1", limit=2)
    assert [m["content"] for m in page.messages] == ["3", "a3"]
    assert page.has_more is True
    assert page.next_before_seq == 5

    older = mem_store.load_chat_page("a", "s1", limit=2, before_seq=page.next_before_seq)
    assert [m["content"] for m in older.messages] == ["2", "a2"]
    assert older.has_more is True
    assert older.next_before_seq == 3

    first = mem_store.load_chat_page("a", "s1", limit=2, before_seq=older.next_before_seq)
    assert [m["content"] for m in first.messages] == ["1", "a1"]
    assert first.has_more is False
    assert first.next_before_seq is None


def test_load_chat_page_keeps_tool_chain_intact(mem_store):
    mem_store.append_chat(
        "a",
        "s1",
        [
            {"role": "user", "content": "旧"},
            {"role": "assistant", "content": "旧答"},
            {"role": "user", "content": "查天气"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "get_weather", "arguments": {"city": "北京"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "get_weather",
                "content": "20度",
            },
            {"role": "assistant", "content": "现在20度"},
        ],
    )
    page = mem_store.load_chat_page("a", "s1", limit=3)
    assert [m["role"] for m in page.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert page.messages[0]["content"] == "查天气"
    assert page.has_more is True


def test_clear_long_term_does_not_delete_chat(mem_store):
    mem_store.append_chat("a", "s1", [{"role": "user", "content": "留下"}])
    mem_store.write_episode("情景", user_id="a")
    mem_store.write_profile("画像", user_id="a")
    mem_store.clear_long_term(user_id="a")
    assert mem_store.search_episode("情景", user_id="a") == []
    assert mem_store.list_profile("a") == []
    assert [m["content"] for m in mem_store.load_chat("a", "s1")] == ["留下"]


def test_clear_chat_does_not_delete_long_term(mem_store):
    mem_store.append_chat("a", "s1", [{"role": "user", "content": "对话"}])
    mem_store.write_episode("情景", user_id="a")
    mem_store.write_profile("画像", user_id="a")
    mem_store.clear_chat("a", "s1")
    assert mem_store.load_chat("a", "s1") == []
    assert mem_store.search_episode("情景", user_id="a")
    assert [item.text for item in mem_store.list_profile("a")] == ["画像"]


def test_append_chat_concurrent_seq_is_unique(mem_store):
    def write(i: int) -> None:
        mem_store.append_chat(
            "a",
            "s1",
            [
                {"role": "user", "content": f"u{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(8)))

    loaded = mem_store.load_chat("a", "s1")
    assert len(loaded) == 16
    assert _chat_seqs(mem_store, "a", "s1") == list(range(1, 17))


def test_create_session_defaults_unnamed_and_is_idempotent(mem_store):
    first = mem_store.create_session("a", "key-1")
    second = mem_store.create_session("a", "key-1")
    other = mem_store.create_session("a", "key-2")
    assert first.title == "新会话"
    assert first.id == second.id
    assert other.id != first.id
    listed = mem_store.list_sessions("a")
    assert {item.id for item in listed} == {first.id, other.id}


def test_update_session_title_only_when_unnamed(mem_store):
    session = mem_store.create_session("a", "title-1")
    assert mem_store.update_session_title("a", session.id, "天气咨询") == "天气咨询"
    assert mem_store.update_session_title("a", session.id, "别改") is None
    assert mem_store.get_session("a", session.id).title == "天气咨询"


def test_cleanup_deletes_empty_unnamed_sessions_only(mem_store):
    empty = mem_store.create_session("a", "empty")
    named = mem_store.create_session("a", "named")
    filled = mem_store.create_session("a", "filled")
    mem_store.update_session_title("a", named.id, "已命名")
    mem_store.append_chat("a", filled.id, [{"role": "user", "content": "你好"}])

    deleted = mem_store.cleanup_empty_unnamed_sessions(max_age_seconds=0)
    assert deleted == 1
    assert mem_store.get_session("a", empty.id) is None
    assert mem_store.get_session("a", named.id) is not None
    assert mem_store.get_session("a", filled.id) is not None


def test_rename_session_overwrites_existing_title(mem_store):
    session = mem_store.create_session("a", "rename-1")
    mem_store.update_session_title("a", session.id, "旧标题")
    assert mem_store.update_session_title("a", session.id, "新标题", only_if_default=False) == "新标题"
    assert mem_store.get_session("a", session.id).title == "新标题"


def test_delete_session_hides_session_and_clears_chat(mem_store):
    session = mem_store.create_session("a", "del-1")
    mem_store.append_chat("a", session.id, [{"role": "user", "content": "你好"}])
    assert mem_store.delete_session("a", session.id) is True
    assert mem_store.get_session("a", session.id) is None
    assert mem_store.load_chat("a", session.id) == []
    assert mem_store.delete_session("a", session.id) is False


def test_cleanup_keeps_recent_empty_unnamed_sessions(mem_store):
    session = mem_store.create_session("a", "recent")
    deleted = mem_store.cleanup_empty_unnamed_sessions(max_age_seconds=3600)
    assert deleted == 0
    assert mem_store.get_session("a", session.id) is not None


def _age_session(store: MemoryStore, session_id: str, seconds: int = 7200) -> None:
    with store._get_pool().connection() as conn:
        conn.execute(
            """
            UPDATE agent_session
            SET created_at = now() - (%s * interval '1 second'),
                updated_at = now() - (%s * interval '1 second')
            WHERE id::text = %s
            """,
            (seconds, seconds, session_id),
        )
        conn.commit()


def test_cleanup_skips_idle_session_after_touch(mem_store):
    """聊天开始会刷新 updated_at：即使创建很久了，只要刚用过就不能当空会话删。"""
    session = mem_store.create_session("a", "stale-then-chat")
    _age_session(mem_store, session.id)
    assert mem_store.touch_session("a", session.id) is True
    assert mem_store.cleanup_empty_unnamed_sessions(max_age_seconds=3600) == 0
    assert mem_store.get_session("a", session.id) is not None


def test_cleanup_skips_session_with_running_checkpoint(mem_store):
    session = mem_store.create_session("a", "in-flight")
    _age_session(mem_store, session.id)
    assert mem_store.create_checkpoint(
        str(uuid.uuid4()), "a", session.id, [{"role": "user", "content": "hi"}]
    )
    assert mem_store.cleanup_empty_unnamed_sessions(max_age_seconds=0) == 0
    assert mem_store.get_session("a", session.id) is not None


def test_idempotent_create_revives_and_refreshes_idle_session(mem_store):
    first = mem_store.create_session("a", "reuse-key")
    _age_session(mem_store, first.id)
    reused = mem_store.create_session("a", "reuse-key")
    assert reused.id == first.id
    assert mem_store.cleanup_empty_unnamed_sessions(max_age_seconds=3600) == 0
    assert mem_store.get_session("a", first.id) is not None


def test_checkpoint_session_mutex_and_cleanup(mem_store):
    """DB 级互斥：同会话第二个 running 写入被部分唯一索引拒绝；
    failed/interrupted 行保留 7 天供恢复，过期后由清理任务删除。"""
    import uuid as _uuid

    uid = "cp_mutex_user"
    try:
        assert (
            mem_store.create_checkpoint(str(_uuid.uuid4()), uid, "cp_s1", [])
            is True
        )
        # 同会话第二个并发 run：路由层预检后的竞态兜底
        assert (
            mem_store.create_checkpoint(str(_uuid.uuid4()), uid, "cp_s1", [])
            is False
        )
        # 不同会话不受影响
        assert (
            mem_store.create_checkpoint(str(_uuid.uuid4()), uid, "cp_s2", [])
            is True
        )
        # 过期 failed 行被清理，未过期的不动
        expired_run = str(_uuid.uuid4())
        fresh_run = str(_uuid.uuid4())
        assert (
            mem_store.create_checkpoint(expired_run, uid, "cp_s3", [])
            is True
        )
        mem_store.mark_checkpoint(expired_run, "failed")
        assert mem_store.create_checkpoint(fresh_run, uid, "cp_s4", []) is True
        mem_store.mark_checkpoint(fresh_run, "interrupted")
        with mem_store._get_pool().connection() as conn:
            conn.execute(
                "UPDATE agent_checkpoint SET updated_at = now() - interval '8 days' "
                "WHERE run_id = %s",
                (expired_run,),
            )
            conn.commit()
        deleted = mem_store.cleanup_old_checkpoints(max_age_seconds=7 * 24 * 3600)
        assert deleted >= 1
        assert mem_store.load_resumable_checkpoint(expired_run) is None
        assert mem_store.load_resumable_checkpoint(fresh_run) is not None

        leftover = str(_uuid.uuid4())
        assert mem_store.create_checkpoint(leftover, uid, "cp_s5", []) is True
        mem_store.mark_checkpoint(leftover, "interrupted")
        assert mem_store.discard_resumable_checkpoints(uid, "cp_s5") == 1
        assert mem_store.load_resumable_checkpoint(leftover) is None
        assert str(mem_store.latest_resumable_run(uid, "cp_s4")) == fresh_run
    finally:
        with mem_store._get_pool().connection() as conn:
            conn.execute("DELETE FROM agent_checkpoint WHERE user_id = %s", (uid,))
            conn.commit()


def test_ensure_schema_collapses_stale_running_checkpoints(mem_store):
    """进程重启场景：启动时把残留 running 收敛为 interrupted，可被 resume 认领。"""
    import uuid as _uuid

    uid = "cp_restart_user"
    run_id = str(_uuid.uuid4())
    try:
        assert mem_store.create_checkpoint(run_id, uid, "cp_restart_s1", []) is True
        mem_store.ensure_schema()  # 模拟重启后的 schema 收敛
        with mem_store._get_pool().connection() as conn:
            row = conn.execute(
                "SELECT status FROM agent_checkpoint WHERE run_id = %s", (run_id,)
            ).fetchone()
        assert row is not None and row["status"] == "interrupted"
        # 收敛后即可被恢复
        assert mem_store.load_resumable_checkpoint(run_id) is not None
    finally:
        with mem_store._get_pool().connection() as conn:
            conn.execute("DELETE FROM agent_checkpoint WHERE user_id = %s", (uid,))
            conn.commit()


def test_new_run_invalidates_old_resumable_checkpoints(mem_store):
    """发起新问题即作废同会话旧的 failed/interrupted checkpoint：
    否则刷新后"继续生成"会恢复过期 run，替换落库还会删掉中间的新问答。"""
    import uuid as _uuid

    uid, sid = "cp_invalidate_user", "cp_invalidate_s1"
    stale_run = str(_uuid.uuid4())
    try:
        assert mem_store.create_checkpoint(stale_run, uid, sid, []) is True
        mem_store.mark_checkpoint(stale_run, "interrupted")
        assert str(mem_store.latest_resumable_run(uid, sid)) == stale_run

        # 用户改发新问题：新 run 的 C0 同事务清掉旧的可恢复行
        new_run = str(_uuid.uuid4())
        assert mem_store.create_checkpoint(new_run, uid, sid, []) is True
        assert mem_store.load_resumable_checkpoint(stale_run) is None
        assert mem_store.latest_resumable_run(uid, sid) is None

        # 新 run 自己再中断 → 可恢复的是它，不是更早的
        mem_store.mark_checkpoint(new_run, "interrupted")
        assert str(mem_store.latest_resumable_run(uid, sid)) == new_run
    finally:
        with mem_store._get_pool().connection() as conn:
            conn.execute("DELETE FROM agent_checkpoint WHERE user_id = %s", (uid,))
            conn.commit()
