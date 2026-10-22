from time import monotonic, sleep

from ai_agent.memory.short_term import (
    SUMMARY_ASSISTANT,
    SUMMARY_USER_PREFIX,
    ShortTermMemory,
    assemble,
    compact,
    compress,
    crude_summary,
)


def test_compact_keeps_system_and_recent_turns():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "3"},
        {"role": "assistant", "content": "a3"},
    ]
    out = compact(messages, keep=2)
    assert out[0]["role"] == "system"
    assert [m["content"] for m in out[1:]] == ["2", "a2", "3", "a3"]


def test_compact_drops_leading_tool():
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "1", "name": "x", "arguments": {}}],
        },
        {"role": "tool", "tool_call_id": "1", "name": "x", "content": "ok"},
        {"role": "user", "content": "hi"},
    ]
    out = compact(messages, keep=2)
    assert out[0]["role"] == "system"
    assert out[1]["role"] != "tool"
    assert out[1]["content"] == "hi"


def test_compact_keeps_tool_chain_inside_turn():
    messages = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "天气"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "1", "name": "get_weather", "arguments": {}}],
        },
        {"role": "tool", "tool_call_id": "1", "name": "get_weather", "content": "20度"},
        {"role": "assistant", "content": "20度"},
    ]
    out = compact(messages, keep=1)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "assistant"]
    assert out[0]["content"] == "天气"


def test_assemble_inserts_summary_pair():
    messages = assemble("sys", "用户在问天气", [{"role": "user", "content": "现在呢"}])
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == SUMMARY_USER_PREFIX + "用户在问天气"
    assert messages[2]["content"] == SUMMARY_ASSISTANT
    assert messages[3]["content"] == "现在呢"


def test_assemble_skips_empty_summary():
    messages = assemble("sys", "  ", [{"role": "user", "content": "hi"}])
    assert [m["role"] for m in messages] == ["system", "user"]


def test_compress_summarizes_overflow_turns():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "3"},
        {"role": "assistant", "content": "a3"},
    ]
    seen: list[list[str]] = []

    def fake_summarize(previous: str, overflow):
        seen.append([m["content"] for m in overflow])
        return "旧轮次摘要"

    recent, summary = compress(
        messages, keep=2, old_summary="", summarize=fake_summarize
    )
    assert seen == [["1", "a1"]]
    assert summary == "旧轮次摘要"
    assert [m["content"] for m in recent] == ["2", "a2", "3", "a3"]


def test_compress_does_not_summarize_within_window():
    messages = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": "a2"},
    ]
    recent, summary = compress(
        messages, keep=2, old_summary="", summarize=lambda *_: "不该调用"
    )
    assert summary == ""
    assert [m["content"] for m in recent] == ["1", "a1", "2", "a2"]


def test_compress_strips_existing_summary_pair():
    messages = assemble(
        "sys",
        "已有摘要",
        [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "a3"},
        ],
    )
    recent, summary = compress(
        messages, keep=2, old_summary="已有摘要", summarize=lambda *_: "新摘要"
    )
    assert summary == "新摘要"
    assert [m["content"] for m in recent] == ["2", "a2", "3", "a3"]


def test_crude_summary_keeps_user_and_assistant():
    text = crude_summary(
        "旧摘要",
        [
            {"role": "user", "content": "我叫小明"},
            {"role": "assistant", "content": "你好小明"},
            {"role": "tool", "content": "忽略"},
        ],
    )
    assert "旧摘要" in text
    assert "我叫小明" in text
    assert "你好小明" in text
    assert "忽略" not in text


def test_save_compacts_to_max_turns():
    memory = ShortTermMemory(max_turns=2)
    memory.save(
        "s1",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "a3"},
        ],
        summary="旧摘要",
    )
    loaded = memory.load("s1")
    assert loaded is not None
    assert loaded.summary == "旧摘要"
    assert [m["content"] for m in loaded.messages] == ["2", "a2", "3", "a3"]


def test_short_term_isolates_users_with_same_session():
    memory = ShortTermMemory(max_turns=10)
    memory.save("s1", [{"role": "user", "content": "用户a"}], user_id="a")
    memory.save("s1", [{"role": "user", "content": "用户b"}], user_id="b")
    assert memory.load("s1", user_id="a").messages[0]["content"] == "用户a"
    assert memory.load("s1", user_id="b").messages[0]["content"] == "用户b"


def test_short_term_evicts_oldest_session():
    memory = ShortTermMemory(max_turns=10, max_sessions=2)
    memory.save("s1", [{"role": "user", "content": "1"}], user_id="a")
    memory.save("s2", [{"role": "user", "content": "2"}], user_id="b")
    memory.save("s3", [{"role": "user", "content": "3"}], user_id="c")
    assert memory.load("s1", user_id="a") is None
    assert memory.load("s2", user_id="b").messages[0]["content"] == "2"
    assert memory.load("s3", user_id="c").messages[0]["content"] == "3"


def test_short_term_lru_keeps_recently_used():
    memory = ShortTermMemory(max_turns=10, max_sessions=2)
    memory.save("s1", [{"role": "user", "content": "1"}], user_id="a")
    memory.save("s2", [{"role": "user", "content": "2"}], user_id="b")
    memory.load("s1", user_id="a")
    memory.save("s3", [{"role": "user", "content": "3"}], user_id="c")
    assert memory.load("s2", user_id="b") is None
    assert memory.load("s1", user_id="a").messages[0]["content"] == "1"
    assert memory.load("s3", user_id="c").messages[0]["content"] == "3"


def test_short_term_expires_idle_session():
    now = {"t": 0.0}
    memory = ShortTermMemory(
        max_turns=10,
        max_sessions=10,
        ttl_seconds=10,
        clock=lambda: now["t"],
    )
    memory.save("s1", [{"role": "user", "content": "1"}], user_id="a")
    now["t"] = 9
    assert memory.load("s1", user_id="a").messages[0]["content"] == "1"
    now["t"] = 19
    assert memory.load("s1", user_id="a") is None


def test_short_term_expires_idle_even_when_under_limit():
    now = {"t": 0.0}
    memory = ShortTermMemory(
        max_turns=10,
        max_sessions=10,
        ttl_seconds=10,
        clock=lambda: now["t"],
    )
    memory.save("s1", [{"role": "user", "content": "1"}], user_id="a")
    now["t"] = 11
    memory.save("s2", [{"role": "user", "content": "2"}], user_id="b")
    assert memory.load("s1", user_id="a") is None
    assert memory.load("s2", user_id="b").messages[0]["content"] == "2"


def test_short_term_periodic_cleanup_expires_without_access():
    now = {"t": 0.0}
    memory = ShortTermMemory(
        max_turns=10,
        max_sessions=10,
        ttl_seconds=10,
        cleanup_interval_seconds=0.02,
        clock=lambda: now["t"],
    )
    memory.save("s1", [{"role": "user", "content": "1"}], user_id="a")
    now["t"] = 11
    memory.start_cleanup()
    try:
        deadline = monotonic() + 1
        while monotonic() < deadline:
            with memory._lock:
                empty = not memory._sessions
            if empty:
                break
            sleep(0.01)
        with memory._lock:
            assert dict(memory._sessions) == {}
    finally:
        memory.stop_cleanup()
