import contextvars
from unittest.mock import patch

import pytest

from ai_agent.agents.loop import _dedupe_seam, run, run_loop, run_stream
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
        patch("ai_agent.agents.loop.store.create_checkpoint"),
        patch("ai_agent.agents.loop.store.update_checkpoint"),
        patch("ai_agent.agents.loop.store.mark_checkpoint"),
        patch("ai_agent.agents.loop.store.delete_checkpoint"),
        patch("ai_agent.agents.loop.summarize_session", side_effect=_fake_summarize),
        patch("ai_agent.agents.loop.capture_long_term"),
    ):
        yield
    short_term_memory.clear()


def _ok_stream(messages, tools, **kwargs):
    yield ("text", "ok")
    yield ("result", {"content": "ok", "tool_calls": []})


def test_loop_returns_model_text_when_no_tool():
    def fake_stream(messages, tools, **kwargs):
        yield ("text", "你好")
        yield ("result", {"content": "你好", "tool_calls": []})

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        assert run("你好") == "你好"


def test_loop_runs_weather_tool():
    calls = {"n": 0}

    def fake_stream(messages, tools, **kwargs):
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

    def fake_stream(messages, tools, **kwargs):
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

    def fake_stream(messages, tools, **kwargs):
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

    def fake_stream(messages, tools, **kwargs):
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
    def fake_stream(messages, tools, **kwargs):
        yield ("text", "你好")
        yield ("result", {"content": "你好", "tool_calls": []})

    with patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream):
        gen = run_stream("你好", user_id="u1", chat_session_id="s3")
        first = next(gen)  # run_started
        assert first["type"] == "run_started"
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

    def fake_stream(messages, tools, **kwargs):
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

    def fake_stream(messages, tools, **kwargs):
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

    def fake_stream(messages, tools, **kwargs):
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

    def fake_stream(messages, tools, **kwargs):
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

    assert [e["type"] for e in events] == ["run_started", "text", "done"]
    # 成功才替换：同一事务删旧写新，走 replace 而非 append
    assert len(replaced) == 1
    assert [m["role"] for m in replaced[0]] == ["assistant"]
    assert replaced[0][0]["content"] == "ok"
    fake_append.assert_not_called()


def _regenerate_failure_cases():
    def llm_error(messages, tools, **kwargs):
        raise LLMError("upstream timeout")
        yield  # pragma: no cover

    def empty_result(messages, tools, **kwargs):
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


def test_run_stream_checkpoint_lifecycle_success():
    calls: list[tuple[str, tuple]] = []

    def record(kind: str):
        def _fn(*a, **k):
            # messages 是后续会被继续 append 的同一列表对象，拍快照留存
            # （列表参数可能在任意位置，逐个拷贝）
            calls.append(
                (kind, tuple(list(x) if isinstance(x, list) else x for x in a))
            )

        return _fn

    with (
        patch("ai_agent.agents.loop.store.create_checkpoint", side_effect=record("create")),
        patch("ai_agent.agents.loop.store.update_checkpoint", side_effect=record("update")),
        patch(
            "ai_agent.agents.loop.store.delete_checkpoint",
            side_effect=lambda *a, **k: calls.append(("delete", a)),
        ),
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        run("你好", user_id="u1", chat_session_id="s-cp-ok")

    # C0 一次 + C1 一次（无工具）→ completed 删行
    assert [kind for kind, _ in calls] == ["create", "update", "delete"]
    create_args, update_args = calls[0][1], calls[1][1]
    # create: (run_id, user_id, session_id, messages)
    assert create_args[1] == "u1"
    assert create_args[2] == "s-cp-ok"
    assert create_args[3][-1]["role"] == "user"
    # update: (run_id, messages, step)，messages 以最终 assistant 结尾
    assert update_args[2] == 1
    assert update_args[1][-1]["role"] == "assistant"
    assert update_args[1][-1]["content"] == "ok"


def test_run_stream_checkpoint_updates_per_tool_step():
    updates: list[int] = []
    calls = {"n": 0}

    def with_tool(messages, tools, **kwargs):
        # 真实 stream_chat 一次调用只返回一个 result；
        # 用最后一条消息区分第一次（模型请求工具）和第二次（工具结果已给模型）
        calls["n"] += 1
        if messages[-1].get("role") != "tool":
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "get_weather",
                            "arguments": {"city": "北京"},
                        }
                    ],
                },
            )
            return
        yield ("text", "20度")
        yield ("result", {"content": "20度", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.store.create_checkpoint"),
        patch(
            "ai_agent.agents.loop.store.update_checkpoint",
            side_effect=lambda rid, msgs, step: updates.append(step),
        ),
        patch("ai_agent.agents.loop.store.delete_checkpoint"),
        patch("ai_agent.agents.loop.store.append_chat"),
        patch("ai_agent.agents.loop.stream_chat", side_effect=with_tool),
    ):
        run("北京天气", user_id="u1", chat_session_id="s-cp-tool")

    assert calls["n"] == 2
    # C1（模型输出，带 tool_calls）→ C2（工具结果）→ C1（第二轮模型输出）
    assert updates == [1, 2, 3]


def test_run_stream_checkpoint_failed_on_llm_error():
    marked: list[str] = []

    def failing(messages, tools, **kwargs):
        raise LLMError("boom")
        yield  # pragma: no cover

    with (
        patch("ai_agent.agents.loop.store.create_checkpoint"),
        patch("ai_agent.agents.loop.store.update_checkpoint"),
        patch(
            "ai_agent.agents.loop.store.mark_checkpoint",
            side_effect=lambda rid, status: marked.append(status),
        ),
        patch("ai_agent.agents.loop.store.delete_checkpoint"),
        patch("ai_agent.agents.loop.stream_chat", side_effect=failing),
    ):
        events = list(
            run_stream("你好", user_id="u1", chat_session_id="s-cp-fail")
        )

    assert events[-1]["type"] == "error"
    assert marked == ["failed"]


def test_run_stream_checkpoint_interrupted_on_client_disconnect():
    marked: list[str] = []

    def partial(messages, tools, **kwargs):
        yield ("text", "部分")
        yield ("text", "回答")
        yield ("result", {"content": "部分回答", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.store.create_checkpoint"),
        patch("ai_agent.agents.loop.store.update_checkpoint"),
        patch(
            "ai_agent.agents.loop.store.mark_checkpoint",
            side_effect=lambda rid, status: marked.append(status),
        ),
        patch("ai_agent.agents.loop.store.delete_checkpoint"),
        patch("ai_agent.agents.loop.store.append_chat"),
        patch("ai_agent.agents.loop.stream_chat", side_effect=partial),
    ):
        gen = run_stream("你好", user_id="u1", chat_session_id="s-cp-int")
        next(gen)  # run_started
        next(gen)  # 收到第一个流片段后挂起
        gen.close()  # 模拟 SSE 客户端断开（GeneratorExit）

    assert marked == ["interrupted"]


def test_run_stream_interrupt_persists_partial_assistant():
    """中断时把已吐出的半截回答写入历史，刷新后还能看见。"""
    appended: list[list[dict]] = []

    def partial(messages, tools, **kwargs):
        yield ("text", "部分")
        yield ("text", "回答")
        yield ("result", {"content": "部分回答", "tool_calls": []})

    with (
        patch(
            "ai_agent.agents.loop.store.append_chat",
            side_effect=lambda *_a: appended.append(_a[2]),
        ),
        patch("ai_agent.agents.loop.stream_chat", side_effect=partial),
    ):
        gen = run_stream("你好", user_id="u1", chat_session_id="s-partial")
        next(gen)  # run_started
        next(gen)  # 第一个 text
        gen.close()

    assert len(appended) == 1
    roles = [m["role"] for m in appended[0]]
    assert roles == ["user", "assistant"]
    assert appended[0][-1]["content"] == "部分"


def test_run_stream_regenerate_interrupt_keeps_old_answer():
    """重新生成中途暂停：不覆盖库里的旧完整回答。"""
    replaced: list[list[dict]] = []
    appended: list[list[dict]] = []

    def partial(messages, tools, **kwargs):
        yield ("text", "新")
        yield ("result", {"content": "新回答", "tool_calls": []})

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
            side_effect=lambda *_a: appended.append(_a[2]),
        ),
        patch("ai_agent.agents.loop.stream_chat", side_effect=partial),
    ):
        gen = run_stream(
            "", user_id="u1", chat_session_id="s-regen-int", regenerate=True
        )
        next(gen)  # run_started
        next(gen)  # text
        gen.close()

    assert replaced == []
    assert appended == []


def test_run_loop_without_run_id_skips_checkpoint():
    with (
        patch("ai_agent.agents.loop.store.create_checkpoint") as cp_create,
        patch("ai_agent.agents.loop.store.update_checkpoint") as cp_update,
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        list(run_loop([{"role": "user", "content": "你好"}], "u1"))

    cp_create.assert_not_called()
    cp_update.assert_not_called()


def test_run_stream_resume_replaces_partial_history():
    """恢复成功：走替换路径落库，不重复 append。"""
    claimed = {
        "user_id": "u1",
        "chat_session_id": "s-resume",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "北京天气"},
            {"role": "assistant", "content": "让我查一下"},
        ],
    }
    replaced, appended = [], []
    with (
        patch(
            "ai_agent.agents.loop.store.claim_resumable_checkpoint",
            return_value=claimed,
        ),
        patch("ai_agent.agents.loop.store.get_session_summary", return_value=""),
        patch(
            "ai_agent.agents.loop.store.replace_chat_after_last_user",
            side_effect=lambda *a: replaced.append(a),
        ),
        patch(
            "ai_agent.agents.loop.store.append_chat",
            side_effect=lambda *a: appended.append(a),
        ),
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        events = list(
            run_stream(
                "", user_id="u1", chat_session_id="whatever", resume_run_id="old-run"
            )
        )

    assert [e["type"] for e in events][0] == "run_started"
    assert events[-1]["type"] == "done"
    assert not appended  # 替换而非追加
    # prefill 续写：半截"让我查一下"作为种子送给模型，新内容接在其后，
    # 合并成一条完整回答落库
    assert [e.get("content") for e in events if e["type"] == "text"] == [
        "让我查一下",
        "ok",
    ]
    assert [m["role"] for m in replaced[0][2]] == ["assistant"]
    assert replaced[0][2][0]["content"] == "让我查一下ok"


def test_run_stream_resume_prefills_partial_assistant():
    """prefill 续写（默认）：半截 assistant 留在上下文尾部送给模型，
    且种子消息带消息级 partial=true（服务端原生从断点续写）。"""
    seen: list[list[dict]] = []

    def capture(messages, tools, **kwargs):
        seen.append([dict(m) for m in messages])
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    claimed = {
        "user_id": "u1",
        "chat_session_id": "s-resume",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "讲个故事"},
            {"role": "assistant", "content": "从前"},
        ],
    }
    with (
        patch(
            "ai_agent.agents.loop.store.claim_resumable_checkpoint",
            return_value=claimed,
        ),
        patch("ai_agent.agents.loop.store.get_session_summary", return_value=""),
        patch("ai_agent.agents.loop.store.replace_chat_after_last_user"),
        patch("ai_agent.agents.loop.stream_chat", side_effect=capture),
    ):
        list(
            run_stream(
                "", user_id="u1", chat_session_id="s-resume", resume_run_id="old-run"
            )
        )

    assert [m["role"] for m in seen[0]] == ["system", "user", "assistant"]
    assert seen[0][-1]["content"] == "从前"
    assert seen[0][-1]["partial"] is True


def test_run_stream_resume_continuation_merges_into_seed():
    """续写结果与半截种子合并成一条完整回答落库，前端只看到增量。"""
    replaced = []

    def capture(messages, tools, **kwargs):
        yield ("text", "接着写的内容")
        yield ("result", {"content": "接着写的内容", "tool_calls": []})

    claimed = {
        "user_id": "u1",
        "chat_session_id": "s-resume",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "讲个故事"},
            {"role": "assistant", "content": "从前"},
        ],
    }
    with (
        patch(
            "ai_agent.agents.loop.store.claim_resumable_checkpoint",
            return_value=claimed,
        ),
        patch("ai_agent.agents.loop.store.get_session_summary", return_value=""),
        patch(
            "ai_agent.agents.loop.store.replace_chat_after_last_user",
            side_effect=lambda *a: replaced.append(a[2]),
        ),
        patch("ai_agent.agents.loop.stream_chat", side_effect=capture),
    ):
        events = list(
            run_stream(
                "", user_id="u1", chat_session_id="s-resume", resume_run_id="old-run"
            )
        )

    # 历史是合并后的完整回答（一条 assistant，半截 + 续写，partial 标记已摘除）
    assert [m["role"] for m in replaced[0]] == ["assistant"]
    assert replaced[0][0]["content"] == "从前接着写的内容"
    assert "partial" not in replaced[0][0]
    # 前端只看到续写的增量（半截由回放事件负责）
    assert [e.get("content") for e in events if e["type"] == "text"] == [
        "从前",
        "接着写的内容",
    ]


def test_run_stream_resume_interrupted_mid_continuation_stashes_into_seed():
    """续写期间再次中断：增量接进半截种子，快照保持单条半截消息。"""
    snapshots: list[list[dict]] = []

    def partial(messages, tools, **kwargs):
        yield ("text", "，天气晴朗")
        yield ("result", {"content": "，天气晴朗", "tool_calls": []})

    claimed = {
        "user_id": "u1",
        "chat_session_id": "s-resume",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "北京天气"},
            {"role": "assistant", "content": "今天"},
        ],
    }
    with (
        patch(
            "ai_agent.agents.loop.store.claim_resumable_checkpoint",
            return_value=claimed,
        ),
        patch("ai_agent.agents.loop.store.get_session_summary", return_value=""),
        patch(
            "ai_agent.agents.loop.store.update_checkpoint",
            side_effect=lambda _rid, msgs, _step: snapshots.append(
                [dict(m) for m in msgs]
            ),
        ),
        patch("ai_agent.agents.loop.store.mark_checkpoint"),
        patch("ai_agent.agents.loop.store.delete_checkpoint"),
        patch("ai_agent.agents.loop.stream_chat", side_effect=partial),
    ):
        gen = run_stream(
            "", user_id="u1", chat_session_id="s-resume", resume_run_id="old-run"
        )
        next(gen)  # run_started
        next(gen)  # 回放事件（半截种子文本）
        next(gen)  # 续写的第一个片段，此刻挂在 run_loop 的 yield 上
        gen.close()  # 模拟 SSE 客户端断开

    assert snapshots, "中断时应把半截写进快照"
    assert snapshots[-1] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "北京天气"},
        {"role": "assistant", "content": "今天，天气晴朗", "partial": True},
    ]


def test_run_stream_resume_fallback_drops_incomplete_assistant():
    """resume_prefill=False 回退路径：半截 assistant 不送给模型，
    整段重生成（适用于不支持 assistant 结尾上下文的兼容 API）。"""
    seen: list[list[str]] = []

    def capture(messages, tools, **kwargs):
        seen.append([str(m.get("role")) for m in messages])
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    claimed = {
        "user_id": "u1",
        "chat_session_id": "s-resume",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "讲个故事"},
            {"role": "assistant", "content": "从前"},
        ],
    }
    with (
        patch(
            "ai_agent.agents.loop.store.claim_resumable_checkpoint",
            return_value=claimed,
        ),
        patch("ai_agent.agents.loop.store.get_session_summary", return_value=""),
        patch("ai_agent.agents.loop.store.replace_chat_after_last_user"),
        patch("ai_agent.agents.loop.stream_chat", side_effect=capture),
        patch("ai_agent.agents.loop.settings.resume_prefill", False),
    ):
        list(
            run_stream(
                "", user_id="u1", chat_session_id="s-resume", resume_run_id="old-run"
            )
        )

    assert seen == [["system", "user"]]


def test_run_stream_resume_claim_miss_yields_error():
    """checkpoint 被别的请求抢走 / 状态不符：发 error 且不落库。"""
    with (
        patch(
            "ai_agent.agents.loop.store.claim_resumable_checkpoint",
            return_value=None,
        ),
        patch("ai_agent.agents.loop.store.append_chat") as append_chat,
    ):
        events = list(
            run_stream(
                "", user_id="u1", chat_session_id="s", resume_run_id="gone"
            )
        )
    assert events == [
        {"type": "error", "message": "对话无法恢复，请重新发送消息"}
    ]
    append_chat.assert_not_called()


def test_run_stream_create_conflict_yields_error():
    """DB 级互斥兜底：C0 写入与并发请求冲突时直接以 error 结束，
    不发 run_started、不跑模型、不落库。"""
    with (
        patch("ai_agent.agents.loop.store.create_checkpoint", return_value=False),
        patch("ai_agent.agents.loop.store.append_chat") as append_chat,
        patch("ai_agent.agents.loop.stream_chat") as stream_chat,
    ):
        events = list(run_stream("你好", user_id="u1", chat_session_id="s-conflict"))

    assert events == [{"type": "error", "message": "该会话有正在进行的回复"}]
    append_chat.assert_not_called()
    stream_chat.assert_not_called()


def test_run_stream_resume_replays_tail_and_keeps_tool_round():
    """带工具的 run 中断在文本轮：prefill 续写保留快照里的 assistant/tool
    轮，半截作为种子送给模型，续写内容合并成一条完整回答落库。"""
    claimed = {
        "user_id": "u1",
        "chat_session_id": "s-resume-tool",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "北京天气"},
            {
                "role": "assistant",
                "content": "让我查一下",
                "tool_calls": [{"id": "c1", "name": "get_weather", "arguments": {}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "get_weather",
                "content": "20度",
            },
            {"role": "assistant", "content": "半截回答"},  # prefill 种子
        ],
    }
    replaced, appended = [], []
    with (
        patch(
            "ai_agent.agents.loop.store.claim_resumable_checkpoint",
            return_value=claimed,
        ),
        patch("ai_agent.agents.loop.store.get_session_summary", return_value=""),
        patch(
            "ai_agent.agents.loop.store.replace_chat_after_last_user",
            side_effect=lambda *a: replaced.append(a[2]),
        ),
        patch(
            "ai_agent.agents.loop.store.append_chat",
            side_effect=lambda *a: appended.append(a[2]),
        ),
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        events = list(
            run_stream(
                "", user_id="u1", chat_session_id="whatever", resume_run_id="old-run"
            )
        )

    types = [e["type"] for e in events]
    assert types == ["run_started", "text", "text", "text", "done"]
    assert events[1]["content"] == "让我查一下"  # 回放已完成轮的文本
    assert events[2]["content"] == "半截回答"  # 回放 prefill 种子（前端替换语义）
    assert events[3]["content"] == "ok"  # 续写的新内容
    assert not appended  # 替换而非追加
    # 工具轮保留进历史，半截 + 续写合并成一条完整回答
    assert [m["role"] for m in replaced[0]] == ["assistant", "tool", "assistant"]
    assert [m["content"] for m in replaced[0]] == ["让我查一下", "20度", "半截回答ok"]


def test_dedupe_seam_removes_repeated_sentence():
    """模型续写复述了最后一句时，接缝去重把它裁掉。"""
    assert _dedupe_seam("今天天气很好，阳光明媚", "阳光明媚，适合出门") == "，适合出门"
    # 短前缀巧合（<6 字符）不误删
    assert _dedupe_seam("答案是", "是的，没错") == "是的，没错"
    # 完全无重叠
    assert _dedupe_seam("从前有座山", "山里有座庙") == "山里有座庙"


def test_run_loop_use_web_search_false_excludes_tool():
    """「智能搜索」开关关闭时，web_search 从工具列表剔除（不是靠 prompt 约定）。"""
    fake_schemas = [
        {"function": {"name": "get_weather"}},
        {"function": {"name": "web_search"}},
        {"function": {"name": "memory_search"}},
    ]
    seen_tools = []

    def capture(messages, tools, **kwargs):
        seen_tools.append([t["function"]["name"] for t in tools])
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.tool_schemas", return_value=fake_schemas),
        patch("ai_agent.agents.loop.stream_chat", side_effect=capture),
        patch("ai_agent.agents.loop.store.append_chat"),
    ):
        list(run_loop([{"role": "user", "content": "你好"}], use_web_search=False))
        list(run_loop([{"role": "user", "content": "你好"}]))

    assert seen_tools[0] == ["get_weather", "memory_search"]
    assert seen_tools[1] == ["get_weather", "web_search", "memory_search"]


def test_run_stream_resume_restores_use_web_search_flag():
    """断点续跑沿用原 run 的「智能搜索」开关：checkpoint 里存的 False
    即使在 run_stream 入参默认 True 的情况下也要生效。"""
    claimed = {
        "user_id": "u1",
        "chat_session_id": "s-resume",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "你好"},
        ],
        "use_web_search": False,
    }
    seen_tools = []

    def capture(messages, tools, **kwargs):
        seen_tools.append([t["function"]["name"] for t in tools])
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    with (
        patch(
            "ai_agent.agents.loop.store.claim_resumable_checkpoint",
            return_value=claimed,
        ),
        patch("ai_agent.agents.loop.store.get_session_summary", return_value=""),
        patch(
            "ai_agent.agents.loop.store.replace_chat_after_last_user",
            side_effect=lambda *a: None,
        ),
        patch(
            "ai_agent.agents.loop.tool_schemas",
            return_value=[
                {"function": {"name": "get_weather"}},
                {"function": {"name": "web_search"}},
            ],
        ),
        patch("ai_agent.agents.loop.stream_chat", side_effect=capture),
    ):
        list(
            run_stream(
                "",
                user_id="u1",
                chat_session_id="whatever",
                resume_run_id="old-run",
            )
        )

    assert seen_tools == [["get_weather"]]
