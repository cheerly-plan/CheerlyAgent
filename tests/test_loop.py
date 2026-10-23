import contextvars
from unittest.mock import patch

import pytest

from ai_agent.agents.loop import run, run_stream
from ai_agent.core.exceptions import LLMError
from ai_agent.memory.short_term import SUMMARY_USER_PREFIX, short_term_memory
from ai_agent.memory.store import MemoryItem


def _fake_summarize(previous, overflow):
    bits = [m.get("content") or "" for m in overflow if m.get("role") == "user"]
    prefix = f"{previous};" if previous else ""
    return prefix + ",".join(bits) if bits else previous


@pytest.fixture(autouse=True)
def _isolate_loop_memory():
    short_term_memory.clear()
    with (
        patch("ai_agent.agents.loop.store.list_profile", return_value=[]),
        patch("ai_agent.agents.loop.store.load_recent_turns", return_value=[]),
        patch("ai_agent.agents.loop.store.get_session_summary", return_value=""),
        patch("ai_agent.agents.loop.store.update_session_summary"),
        patch("ai_agent.agents.loop.store.append_chat"),
        patch("ai_agent.agents.loop.summarize_session", side_effect=_fake_summarize),
        patch("ai_agent.agents.loop.capture_long_term"),
    ):
        yield
    short_term_memory.clear()


def _ok_stream(messages, tools):
    yield ("text", "ok")
    yield ("result", {"content": "ok", "tool_calls": []})


def test_loop_returns_model_text_when_no_tool():
    def fake_stream(messages, tools):
        yield ("text", "你好")
        yield ("result", {"content": "你好", "tool_calls": []})

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        assert run("你好") == "你好"


def test_loop_runs_weather_tool():
    calls = {"n": 0}

    def fake_stream(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "get_weather",
                            "arguments": {"city": "北京"},
                        }
                    ],
                },
            )
            return
        tool_msg = messages[-1]
        yield ("text", tool_msg["content"])
        yield ("result", {"content": tool_msg["content"], "tool_calls": []})

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        result = run("北京天气怎么样？")

    assert calls["n"] == 2
    assert "北京" in result
    assert "20度" in result


def test_loop_keeps_session_history():
    seen: list[list[str]] = []

    def fake_stream(messages, tools):
        seen.append([m["role"] for m in messages])
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        run("第一句", chat_session_id="s1")
        run("第二句", chat_session_id="s1")

    assert seen[0] == ["system", "user"]
    assert seen[1] == ["system", "user", "assistant", "user"]
    history = short_term_memory.load("s1")
    assert history is not None
    assert [m["role"] for m in history.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert history.messages[-1]["content"] == "ok"
    assert history.summary == ""


def test_loop_summarizes_overflow_turns():
    original = short_term_memory.max_turns
    short_term_memory.max_turns = 2
    seen: list[list[str]] = []

    def fake_stream(messages, tools):
        seen.append([m.get("content") or "" for m in messages if m["role"] != "system"])
        yield from _ok_stream(messages, tools)

    try:
        with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
            run("第一句", chat_session_id="s-compact")
            run("第二句", chat_session_id="s-compact")
            run("第三句", chat_session_id="s-compact")
            run("第四句", chat_session_id="s-compact")
        history = short_term_memory.load("s-compact")
    finally:
        short_term_memory.max_turns = original

    assert history is not None
    assert history.summary == "第一句;第二句"
    assert [m["content"] for m in history.messages] == [
        "第三句",
        "ok",
        "第四句",
        "ok",
    ]
    assert seen[3][0].startswith(SUMMARY_USER_PREFIX)
    assert "第一句" in seen[3][0]
    assert seen[3][2:] == ["第二句", "ok", "第三句", "ok", "第四句"]


def test_loop_injects_profile_memory():
    profile = MemoryItem(
        id="1",
        user_id="u1",
        text="喜欢 pytest",
        created_at="t",
    )
    captured: list[str] = []

    def fake_stream(messages, tools):
        captured.append(messages[0]["content"])
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.store.list_profile", return_value=[profile]),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
    ):
        run("测试习惯", user_id="u1", chat_session_id="s2")

    assert "喜欢 pytest" in captured[0]


def test_run_stream_survives_context_switch_between_yields():
    def fake_stream(messages, tools):
        yield ("text", "你好")
        yield ("result", {"content": "你好", "tool_calls": []})

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        gen = run_stream("你好", user_id="u1", chat_session_id="s3")
        first = next(gen)
        rest = contextvars.Context().run(lambda: list(gen))

    assert first["type"] == "text"
    assert rest[-1]["type"] == "done"


def test_run_stream_appends_new_messages_only():
    appended: list[list[str]] = []

    def fake_append(user_id, chat_session_id, messages):
        appended.append([m["role"] for m in messages])

    with (
        patch("ai_agent.agents.loop.store.append_chat", side_effect=fake_append),
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        run("第一句", user_id="u1", chat_session_id="s1")
        run("第二句", user_id="u1", chat_session_id="s1")

    assert appended[0] == ["user", "assistant"]
    assert appended[1] == ["user", "assistant"]


def test_run_stream_appends_tool_turn():
    calls = {"n": 0}
    appended: list[list[dict]] = []

    def fake_stream(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "get_weather",
                            "arguments": {"city": "北京"},
                        }
                    ],
                },
            )
            return
        yield ("text", "20度")
        yield ("result", {"content": "20度", "tool_calls": []})

    def fake_append(user_id, chat_session_id, messages):
        appended.append(messages)

    with (
        patch("ai_agent.agents.loop.store.append_chat", side_effect=fake_append),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
    ):
        run("北京天气怎么样？", user_id="u1", chat_session_id="s-tool")

    roles = [m["role"] for m in appended[0]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert appended[0][1]["tool_calls"][0]["name"] == "get_weather"
    assert appended[0][2]["tool_call_id"] == "call_1"
    assert "20度" in appended[0][3]["content"]


def test_run_stream_hydrates_from_db_when_memory_empty():
    history = [
        {"role": "user", "content": "上一句"},
        {"role": "assistant", "content": "上次答"},
    ]
    seen: list[list[str]] = []

    def fake_stream(messages, tools):
        seen.append([m["role"] for m in messages])
        yield from _ok_stream(messages, tools)

    with (
        patch("ai_agent.agents.loop.store.load_recent_turns", return_value=history),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
    ):
        run("下一句", user_id="u1", chat_session_id="s-miss")

    assert seen[0] == ["system", "user", "assistant", "user"]


def test_run_stream_hydrates_summary_conversation():
    history = [
        {"role": "user", "content": "第二句"},
        {"role": "assistant", "content": "第二答"},
    ]
    seen: list[list[str]] = []

    def fake_stream(messages, tools):
        seen.append([m.get("content") or "" for m in messages])
        yield from _ok_stream(messages, tools)

    with (
        patch("ai_agent.agents.loop.store.load_recent_turns", return_value=history),
        patch("ai_agent.agents.loop.store.get_session_summary", return_value="第一轮在问天气"),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
    ):
        run("第三句", user_id="u1", chat_session_id="s-sum")

    assert seen[0][0]
    assert seen[0][1].startswith(SUMMARY_USER_PREFIX)
    assert "第一轮在问天气" in seen[0][1]
    assert seen[0][3:] == ["第二句", "第二答", "第三句"]


def test_run_stream_skips_db_load_when_short_term_exists():
    poison = [{"role": "user", "content": "不该出现"}]
    seen: list[list[str]] = []

    def fake_stream(messages, tools):
        seen.append([m.get("content") for m in messages if m["role"] != "system"])
        yield from _ok_stream(messages, tools)

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        run("第一句", user_id="u1", chat_session_id="s-hot")

    with (
        patch("ai_agent.agents.loop.store.load_recent_turns", return_value=poison),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
    ):
        run("第二句", user_id="u1", chat_session_id="s-hot")

    assert "不该出现" not in seen[1]
    assert seen[1] == ["第一句", "ok", "第二句"]


def test_run_stream_captures_long_term_after_reply():
    with (
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
        patch("ai_agent.agents.loop.capture_long_term") as capture,
    ):
        run("我想用 Koa 做 BFF", user_id="u1", chat_session_id="s-extract")
    capture.assert_called_once()
    user_id, turn, summary, user_input = capture.call_args.args
    assert user_id == "u1"
    assert user_input == "我想用 Koa 做 BFF"
    assert summary == ""
    assert [m["role"] for m in turn] == ["user", "assistant"]
    assert turn[0]["content"] == "我想用 Koa 做 BFF"


def test_run_stream_survives_append_chat_failure():
    with (
        patch(
            "ai_agent.agents.loop.store.append_chat",
            side_effect=RuntimeError("db down"),
        ),
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        assert run("你好", user_id="u1", chat_session_id="s-fail") == "ok"


def test_run_stream_survives_load_chat_failure():
    with (
        patch(
            "ai_agent.agents.loop.store.load_recent_turns",
            side_effect=RuntimeError("db down"),
        ),
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        assert run("你好", user_id="u1", chat_session_id="s-load-fail") == "ok"


def test_run_stream_requests_chat_window():
    with (
        patch(
            "ai_agent.agents.loop.store.load_recent_turns", return_value=[]
        ) as load_recent_turns,
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        run("hi", user_id="u1", chat_session_id="s-lim")
    load_recent_turns.assert_called_once_with(
        "u1", "s-lim", turns=short_term_memory.max_turns
    )


def test_loop_isolates_short_term_by_user():
    with patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream):
        run("给a", user_id="a", chat_session_id="same")
        run("给b", user_id="b", chat_session_id="same")
    assert short_term_memory.load("same", user_id="a").messages[-2]["content"] == "给a"
    assert short_term_memory.load("same", user_id="b").messages[-2]["content"] == "给b"


def test_loop_persists_summary_when_overflow():
    original = short_term_memory.max_turns
    short_term_memory.max_turns = 1
    updated: list[str] = []

    def fake_update(user_id, session_id, summary):
        updated.append(summary)

    try:
        with (
            patch(
                "ai_agent.agents.loop.store.update_session_summary",
                side_effect=fake_update,
            ),
            patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
        ):
            run("第一句", user_id="u1", chat_session_id="s-sum-save")
            run("第二句", user_id="u1", chat_session_id="s-sum-save")
    finally:
        short_term_memory.max_turns = original

    assert updated == ["第一句"]


_REGEN_HISTORY = [
    {"role": "user", "content": "北京天气"},
    {"role": "assistant", "content": "旧回答"},
]


def test_regenerate_success_replaces_old_answer_atomically():
    replaced: list[list[dict]] = []

    def fake_replace(user_id, chat_session_id, messages):
        replaced.append(messages)

    with (
        patch(
            "ai_agent.agents.loop.store.load_recent_turns",
            return_value=_REGEN_HISTORY,
        ),
        patch(
            "ai_agent.agents.loop.store.replace_chat_after_last_user",
            side_effect=fake_replace,
        ),
        patch("ai_agent.agents.loop.store.append_chat") as fake_append,
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        events = list(
            run_stream("", user_id="u1", chat_session_id="s-regen", regenerate=True)
        )

    assert [e["type"] for e in events] == ["text", "done"]
    # 成功才替换：同一事务删旧写新，走 replace 而非 append
    assert len(replaced) == 1
    assert [m["role"] for m in replaced[0]] == ["assistant"]
    assert replaced[0][0]["content"] == "ok"
    fake_append.assert_not_called()


def _regenerate_failure_cases():
    def llm_error(messages, tools):
        raise LLMError("upstream timeout")
        yield  # pragma: no cover

    def empty_result(messages, tools):
        yield ("result", None)

    return [llm_error, empty_result]


@pytest.mark.parametrize("failing", _regenerate_failure_cases())
def test_regenerate_failure_keeps_old_answer(failing):
    replaced: list[list[dict]] = []
    appended: list[list[dict]] = []

    with (
        patch(
            "ai_agent.agents.loop.store.load_recent_turns",
            return_value=_REGEN_HISTORY,
        ),
        patch(
            "ai_agent.agents.loop.store.replace_chat_after_last_user",
            side_effect=lambda *_a: replaced.append(_a[2]),
        ),
        patch(
            "ai_agent.agents.loop.store.append_chat",
            side_effect=lambda *_a, **_k: appended.append(_a[2]),
        ),
        patch("ai_agent.agents.loop.capture_long_term") as capture,
        patch("ai_agent.agents.loop.stream_chat", side_effect=failing),
    ):
        events = list(
            run_stream("", user_id="u1", chat_session_id="s-regen-fail", regenerate=True)
        )

    # 失败时旧回答原样保留：不删库、不写库、不抽取记忆、不污染热缓存
    assert events[-1]["type"] == "error"
    assert replaced == []
    assert appended == []
    capture.assert_not_called()
    assert short_term_memory.load("s-regen-fail") is None
