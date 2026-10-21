from time import monotonic, sleep

from ai_agent.memory.short_term import ShortTermMemory, compact


def test_compact_keeps_system_and_recent():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": "a2"},
    ]
    out = compact(messages, keep=2)
    assert out[0]["role"] == "system"
    assert [m["content"] for m in out[1:]] == ["2", "a2"]


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
        ],
    )
    loaded = memory.load("s1")
    assert [m["content"] for m in loaded] == ["sys", "2", "a2"]


def test_short_term_isolates_users_with_same_session():
    memory = ShortTermMemory(max_turns=10)
    memory.save("s1", [{"role": "user", "content": "用户a"}], user_id="a")
    memory.save("s1", [{"role": "user", "content": "用户b"}], user_id="b")
    assert memory.load("s1", user_id="a")[0]["content"] == "用户a"
    assert memory.load("s1", user_id="b")[0]["content"] == "用户b"


def test_short_term_evicts_oldest_session():
    memory = ShortTermMemory(max_turns=10, max_sessions=2)
    memory.save("s1", [{"role": "user", "content": "1"}], user_id="a")
    memory.save("s2", [{"role": "user", "content": "2"}], user_id="b")
    memory.save("s3", [{"role": "user", "content": "3"}], user_id="c")
    assert memory.load("s1", user_id="a") == []
    assert memory.load("s2", user_id="b")[0]["content"] == "2"
    assert memory.load("s3", user_id="c")[0]["content"] == "3"


def test_short_term_lru_keeps_recently_used():
    memory = ShortTermMemory(max_turns=10, max_sessions=2)
    memory.save("s1", [{"role": "user", "content": "1"}], user_id="a")
    memory.save("s2", [{"role": "user", "content": "2"}], user_id="b")
    memory.load("s1", user_id="a")
    memory.save("s3", [{"role": "user", "content": "3"}], user_id="c")
    assert memory.load("s2", user_id="b") == []
    assert memory.load("s1", user_id="a")[0]["content"] == "1"
    assert memory.load("s3", user_id="c")[0]["content"] == "3"


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
    assert memory.load("s1", user_id="a")[0]["content"] == "1"
    now["t"] = 19
    assert memory.load("s1", user_id="a") == []


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
    assert memory.load("s1", user_id="a") == []
    assert memory.load("s2", user_id="b")[0]["content"] == "2"


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