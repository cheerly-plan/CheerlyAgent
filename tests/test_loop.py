from ai_agent.agents import turn as turn_module
import contextvars
from unittest.mock import patch

import pytest

from ai_agent.agents.loop import (
    _dedupe_seam,
    _must_call_kb_tool,
    run,
    run_loop,
    run_stream,
)
from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.memory.short_term import SUMMARY_USER_PREFIX, short_term_memory
from ai_agent.storage.store import MemoryItem


def _fake_summarize(previous, overflow):
    bits = [m.get("content") or "" for m in overflow if m.get("role") == "user"]
    prefix = f"{previous};" if previous else ""
    return prefix + ",".join(bits) if bits else previous


@pytest.fixture(autouse=True)
def _isolate_loop_memory():
    short_term_memory.clear()
    with (
        patch("ai_agent.agents.loop.store.list_profile", return_value=[]),
        patch("ai_agent.agents.loop.store.load_all_turns", return_value=[]),
        patch("ai_agent.agents.loop.store.get_session_summary", return_value=""),
        patch("ai_agent.agents.loop.store.update_session_summary"),
        patch("ai_agent.agents.loop.store.append_chat"),
        patch("ai_agent.agents.loop.store.create_checkpoint"),
        patch("ai_agent.agents.loop.store.update_checkpoint"),
        patch("ai_agent.agents.loop.store.mark_checkpoint"),
        patch("ai_agent.agents.loop.store.delete_checkpoint"),
        patch("ai_agent.agents.turn.summarize_session", side_effect=_fake_summarize),
        patch("ai_agent.agents.turn.capture_long_term"),
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

    from ai_agent.tools.misc import weather as wt

    with (
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
        patch.object(
            wt,
            "_geocode",
            return_value={
                "name": "北京",
                "admin1": "北京市",
                "country": "中国",
                "latitude": 39.9,
                "longitude": 116.4,
            },
        ),
        patch.object(
            wt,
            "_fetch_forecast",
            return_value={
                "current": {
                    "temperature_2m": 20.5,
                    "apparent_temperature": 19.0,
                    "relative_humidity_2m": 45,
                    "weather_code": 2,
                    "wind_speed_10m": 3.2,
                },
                "daily": {
                    "time": ["2026-10-25"],
                    "weather_code": [2],
                    "temperature_2m_max": [22.0],
                    "temperature_2m_min": [12.0],
                    "precipitation_probability_max": [10],
                },
            },
        ),
    ):
        result = run("北京天气怎么样？")

    assert calls["n"] == 2
    assert "北京" in result
    assert "气温 20.5°C" in result


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


def test_run_stream_followup_persists_user():
    """追问落库必须以本轮 user 问句为起点，刷新会话不会丢记录。"""
    appended: list[list[dict]] = []

    def fake_append(user_id, chat_session_id, messages):
        appended.append(list(messages))

    with (
        patch("ai_agent.agents.loop.store.append_chat", side_effect=fake_append),
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        run("上一问", user_id="u1", chat_session_id="s-followup")
        run("这一问", user_id="u1", chat_session_id="s-followup")

    assert len(appended) == 2
    assert any(m.get("role") == "user" and m.get("content") == "上一问" for m in appended[0])
    assert any(m.get("role") == "user" and m.get("content") == "这一问" for m in appended[1])


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
        patch("ai_agent.agents.loop.store.load_all_turns", return_value=history),
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
        patch("ai_agent.agents.loop.store.load_all_turns", return_value=history),
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
        patch("ai_agent.agents.loop.store.load_all_turns", return_value=poison),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
    ):
        run("第二句", user_id="u1", chat_session_id="s-hot")

    assert "不该出现" not in seen[1]
    assert seen[1] == ["第一句", "ok", "第二句"]


def test_run_stream_captures_long_term_after_reply():
    with (
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
        patch("ai_agent.agents.turn.capture_long_term") as capture,
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
            "ai_agent.agents.loop.store.load_all_turns",
            side_effect=RuntimeError("db down"),
        ),
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        assert run("你好", user_id="u1", chat_session_id="s-load-fail") == "ok"


def test_run_stream_requests_chat_window():
    with (
        patch(
            "ai_agent.agents.loop.store.load_all_turns", return_value=[]
        ) as load_all_turns,
        patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream),
    ):
        run("hi", user_id="u1", chat_session_id="s-lim")
    load_all_turns.assert_called_once_with("u1", "s-lim")


def test_loop_isolates_short_term_by_user():
    with patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream):
        run("给a", user_id="a", chat_session_id="same")
        run("给b", user_id="b", chat_session_id="same")
    assert short_term_memory.load("same", user_id="a").messages[-2]["content"] == "给a"
    assert short_term_memory.load("same", user_id="b").messages[-2]["content"] == "给b"


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
            "ai_agent.agents.loop.store.load_all_turns",
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
            "ai_agent.agents.loop.store.load_all_turns",
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
        patch("ai_agent.agents.turn.capture_long_term") as capture,
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
            "ai_agent.agents.loop.store.load_all_turns",
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


def test_run_loop_use_kb_false_excludes_kb_search():
    """「知识库」开关关闭时 kb_search 不挂载；打开则恢复。两者互不干扰。"""
    fake_schemas = [
        {"function": {"name": "get_weather"}},
        {"function": {"name": "kb_search"}},
        {"function": {"name": "kg_search"}},
        {"function": {"name": "web_search"}},
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
        list(run_loop([{"role": "user", "content": "你好"}], use_kb=False))
        list(run_loop([{"role": "user", "content": "你好"}], use_kb=True))
        # 两个开关同时关：各剔各的
        list(
            run_loop(
                [{"role": "user", "content": "你好"}],
                use_kb=False,
                use_web_search=False,
            )
        )

    # kg_search（图谱检索）同属知识库能力，随 kb_search 一起摘掉
    assert seen_tools[0] == ["get_weather", "web_search"]
    assert seen_tools[1] == [
        "get_weather",
        "kb_search",
        "kg_search",
        "web_search",
    ]
    assert seen_tools[2] == ["get_weather"]


def test_run_loop_requires_kb_tool_when_question_names_subject():
    """问句点名库内主体时，第一步只留知识库工具并强制调用；查完再放开。"""
    fake_schemas = [
        {"function": {"name": "get_weather"}},
        {"function": {"name": "kb_search"}},
        {"function": {"name": "kg_search"}},
        {"function": {"name": "web_search"}},
    ]
    seen: list[tuple[str, list[str]]] = []

    def capture(messages, tools, tool_choice="auto"):
        seen.append((tool_choice, [t["function"]["name"] for t in tools]))
        if tool_choice == "required":
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "kb_search",
                            "arguments": {"query": "变动原因"},
                        }
                    ],
                },
            )
            return
        yield ("text", "根据报告")
        yield ("result", {"content": "根据报告", "tool_calls": []})

    def subjects(query, file_ids=None):
        if "长安汽车" in query:
            return {"长安汽车", "中国黄金"}
        return set()

    with (
        patch("ai_agent.agents.loop.tool_schemas", return_value=fake_schemas),
        patch("ai_agent.agents.loop.stream_chat", side_effect=capture),
        patch("ai_agent.agents.loop.execute_tool", return_value="报告原文"),
        patch("ai_agent.kb.retrieval.mentioned_subjects", side_effect=subjects),
    ):
        events = list(
            run_loop(
                [
                    {
                        "role": "user",
                        "content": "长安汽车和中国黄金的主要会计数据变动原因",
                    }
                ]
            )
        )

    assert seen[0] == ("required", ["kb_search", "kg_search"])
    assert seen[1][0] == "auto"
    assert seen[1][1] == ["get_weather", "kb_search", "kg_search", "web_search"]
    assert events[-1]["type"] == "done"
    assert any(e.get("type") == "tool_start" for e in events)


def test_run_loop_leaves_tool_choice_auto_without_subject():
    """没点名主体的追问不强制检索，工具列表保持完整。"""
    fake_schemas = [
        {"function": {"name": "kb_search"}},
        {"function": {"name": "web_search"}},
    ]
    seen: list[tuple[str, list[str]]] = []

    def capture(messages, tools, tool_choice="auto"):
        seen.append((tool_choice, [t["function"]["name"] for t in tools]))
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.tool_schemas", return_value=fake_schemas),
        patch("ai_agent.agents.loop.stream_chat", side_effect=capture),
        patch(
            "ai_agent.kb.retrieval.mentioned_subjects",
            return_value=set(),
        ),
        patch("ai_agent.kb.retrieval.mentioned_entities", return_value=[]),
    ):
        list(
            run_loop(
                [
                    {"role": "user", "content": "中国银行营业额"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "old",
                                "name": "kb_search",
                                "arguments": {"query": "营业收入"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "old",
                        "name": "kb_search",
                        "content": "营业收入 1",
                    },
                    {"role": "assistant", "content": "1"},
                    {"role": "user", "content": "净利润分别是多少"},
                ]
            )
        )

    assert seen == [("auto", ["kb_search", "web_search"])]


def test_must_call_kb_tool_records_entities_when_subject_also_matches(caplog):
    """点了公司又点了实体时，两样都记进日志。不能因为主体命中就跳过实体检测。"""
    tools = [{"function": {"name": "kb_search"}}, {"function": {"name": "kg_search"}}]
    messages = [{"role": "user", "content": "长安汽车的诺安成长持股多少"}]
    with (
        patch("ai_agent.kb.retrieval.mentioned_subjects", return_value={"长安汽车"}),
        patch(
            "ai_agent.kb.retrieval.mentioned_entities",
            return_value=["诺安成长混合型证券投资基金"],
        ) as entities,
    ):
        with caplog.at_level("INFO"):
            assert _must_call_kb_tool(messages, tools) is True
    entities.assert_called_once()
    assert "subjects=长安汽车" in caplog.text
    assert "entities=诺安成长混合型证券投资基金" in caplog.text


def test_report_generation_skips_forced_kb_retrieval():
    """生成研报直接走 generate_report，不必先被强制查一轮 kb_search。"""
    tools = [{"function": {"name": "kb_search"}}, {"function": {"name": "kg_search"}}]
    messages = [
        {"role": "user", "content": "生成中国银行2026Q3财报深度分析PDF研报"}
    ]
    with (
        patch("ai_agent.kb.retrieval.mentioned_subjects", return_value={"中国银行"}),
        patch("ai_agent.kb.retrieval.mentioned_entities", return_value=[]),
    ):
        assert _must_call_kb_tool(messages, tools) is False


def test_question_about_a_report_still_forces_retrieval():
    """「研报里说了什么」是检索题，不能被生成类例外误放行。"""
    tools = [{"function": {"name": "kb_search"}}]
    messages = [{"role": "user", "content": "中国银行的研报里营收是多少"}]
    with (
        patch("ai_agent.kb.retrieval.mentioned_subjects", return_value={"中国银行"}),
        patch("ai_agent.kb.retrieval.mentioned_entities", return_value=[]),
    ):
        assert _must_call_kb_tool(messages, tools) is True


def test_must_call_kb_tool_forces_followup_without_subject():
    """追问省略主体（「那毛利率呢」），上一轮问句点过库内主体 → 同样强制检索。"""
    tools = [{"function": {"name": "kb_search"}}, {"function": {"name": "kg_search"}}]
    messages = [
        {"role": "user", "content": "中科飞测的营业收入是多少"},
        {"role": "assistant", "content": "营业收入为 1.2 亿元。"},
        {"role": "user", "content": "那毛利率呢"},
    ]

    def subjects(query, file_ids=None):
        return {"中科飞测"} if "中科飞测" in query else set()

    with (
        patch("ai_agent.kb.retrieval.mentioned_subjects", side_effect=subjects),
        patch("ai_agent.kb.retrieval.mentioned_entities", return_value=[]),
    ):
        assert _must_call_kb_tool(messages, tools) is True


def test_must_call_kb_tool_followup_skips_when_flag_off():
    """回看兜底开关关掉（kb_followup_force_enabled=False）→ 退回旧行为不强制。"""
    tools = [{"function": {"name": "kb_search"}}]
    messages = [
        {"role": "user", "content": "中科飞测的营业收入是多少"},
        {"role": "assistant", "content": "答"},
        {"role": "user", "content": "那毛利率呢"},
    ]

    def subjects(query, file_ids=None):
        return {"中科飞测"} if "中科飞测" in query else set()

    with (
        patch("ai_agent.agents.loop.settings.kb_followup_force_enabled", False),
        patch("ai_agent.kb.retrieval.mentioned_subjects", side_effect=subjects),
        patch("ai_agent.kb.retrieval.mentioned_entities", return_value=[]),
    ):
        assert _must_call_kb_tool(messages, tools) is False


def test_must_call_kb_tool_followup_not_forced_without_history_subject():
    """历史问句也没点过主体的追问：保持不强制，别把闲聊追问也锁进知识库。"""
    tools = [{"function": {"name": "kb_search"}}]
    messages = [
        {"role": "user", "content": "今天天气如何"},
        {"role": "assistant", "content": "晴"},
        {"role": "user", "content": "那明天呢"},
    ]
    with (
        patch("ai_agent.kb.retrieval.mentioned_subjects", return_value=set()),
        patch("ai_agent.kb.retrieval.mentioned_entities", return_value=[]),
    ):
        assert _must_call_kb_tool(messages, tools) is False


def test_run_loop_requires_kb_tool_on_followup_question():
    """run_loop 集成：追问省略主体时，第一步只留知识库工具并强制调用；查完放开。"""
    fake_schemas = [
        {"function": {"name": "get_weather"}},
        {"function": {"name": "kb_search"}},
        {"function": {"name": "kg_search"}},
    ]
    seen: list[tuple[str, list[str]]] = []

    def capture(messages, tools, tool_choice="auto"):
        seen.append((tool_choice, [t["function"]["name"] for t in tools]))
        if tool_choice == "required":
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "kb_search",
                            "arguments": {"query": "毛利率"},
                        }
                    ],
                },
            )
            return
        yield ("text", "根据报告")
        yield ("result", {"content": "根据报告", "tool_calls": []})

    def subjects(query, file_ids=None):
        return {"中科飞测"} if "中科飞测" in query else set()

    messages = [
        {"role": "user", "content": "中科飞测的营业收入是多少"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "old", "name": "kb_search", "arguments": {"query": "营业收入"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "old",
            "name": "kb_search",
            "content": "营业收入 1",
        },
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "那毛利率呢"},
    ]
    with (
        patch("ai_agent.agents.loop.tool_schemas", return_value=fake_schemas),
        patch("ai_agent.agents.loop.stream_chat", side_effect=capture),
        patch("ai_agent.agents.loop.execute_tool", return_value="毛利率 40%"),
        patch("ai_agent.kb.retrieval.mentioned_subjects", side_effect=subjects),
    ):
        events = list(run_loop(messages))

    assert seen[0] == ("required", ["kb_search", "kg_search"])
    assert seen[1][0] == "auto"
    assert seen[1][1] == ["get_weather", "kb_search", "kg_search"]
    assert events[-1]["type"] == "done"
    assert any(e.get("type") == "tool_start" for e in events)


def test_run_loop_requires_kb_tool_when_question_names_entity():
    """问句点名图谱实体（不是文件主体）时，同样强制调用知识库工具。"""
    fake_schemas = [
        {"function": {"name": "kb_search"}},
        {"function": {"name": "kg_search"}},
        {"function": {"name": "web_search"}},
    ]
    seen: list[tuple[str, list[str]]] = []

    def capture(messages, tools, tool_choice="auto"):
        seen.append((tool_choice, [t["function"]["name"] for t in tools]))
        if tool_choice == "required":
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "kg_search",
                            "arguments": {"query": "金仙"},
                        }
                    ],
                },
            )
            return
        yield ("text", "金仙任董事")
        yield ("result", {"content": "金仙任董事", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.tool_schemas", return_value=fake_schemas),
        patch("ai_agent.agents.loop.stream_chat", side_effect=capture),
        patch("ai_agent.agents.loop.execute_tool", return_value="命中实体：金仙"),
        patch("ai_agent.kb.retrieval.mentioned_subjects", return_value=set()),
        patch("ai_agent.kb.retrieval.mentioned_entities", return_value=["金仙"]),
        patch("ai_agent.tools.kb.kg_search.prefetch_neighborhood", return_value=""),
    ):
        events = list(run_loop([{"role": "user", "content": "金仙的任职"}]))

    assert seen[0] == ("required", ["kb_search", "kg_search"])
    assert seen[1][0] == "auto"
    assert events[-1]["type"] == "done"


def test_run_loop_injects_graph_neighborhood_before_tools():
    """关系问句在选工具前把图谱邻域放进上下文。科目问句不放。"""
    seen: list[list[dict]] = []

    def capture(messages, tools, tool_choice="auto"):
        seen.append(list(messages))
        yield ("text", "汇金持股")
        yield ("result", {"content": "汇金持股", "tool_calls": []})

    block = "【图谱邻域】\n命中实体：中央汇金(org)"
    with (
        patch("ai_agent.agents.loop.stream_chat", side_effect=capture),
        patch("ai_agent.agents.loop.tool_schemas", return_value=[]),
        patch(
            "ai_agent.tools.kb.kg_search.prefetch_neighborhood",
            return_value=block,
        ) as prefetch,
    ):
        list(run_loop([{"role": "user", "content": "中国银行的股东有哪些"}]))

    assert prefetch.called
    assert seen[0][-1]["content"] == block

    seen.clear()
    with (
        patch("ai_agent.agents.loop.stream_chat", side_effect=capture),
        patch("ai_agent.agents.loop.tool_schemas", return_value=[]),
        patch(
            "ai_agent.tools.kb.kg_search.prefetch_neighborhood",
            return_value="",
        ),
    ):
        list(run_loop([{"role": "user", "content": "营业收入是多少"}]))

    assert seen[0][-1]["content"] == "营业收入是多少"


def test_run_loop_does_not_force_tool_after_this_question_was_retrieved():
    """这句问话已经有工具结果时不再强制，避免续跑时没法写答案。"""
    fake_schemas = [
        {"function": {"name": "kb_search"}},
        {"function": {"name": "kg_search"}},
    ]
    seen: list[str] = []

    def capture(messages, tools, tool_choice="auto"):
        seen.append(tool_choice)
        yield ("text", "根据报告")
        yield ("result", {"content": "根据报告", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.tool_schemas", return_value=fake_schemas),
        patch("ai_agent.agents.loop.stream_chat", side_effect=capture),
        patch(
            "ai_agent.kb.retrieval.mentioned_subjects",
            return_value={"长安汽车"},
        ),
    ):
        list(
            run_loop(
                [
                    {"role": "user", "content": "长安汽车的变动原因"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "name": "kb_search",
                                "arguments": {"query": "变动原因"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "c1",
                        "name": "kb_search",
                        "content": "报告原文",
                    },
                ]
            )
        )

    assert seen == ["auto"]


def test_run_stream_resume_restores_use_kb_flag():
    """断点续跑沿用原 run 的「知识库」开关，与智能搜索同一套机制。"""
    claimed = {
        "user_id": "u1",
        "chat_session_id": "s-resume-kb",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "你好"},
        ],
        "use_web_search": True,
        "use_kb": False,
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
                {"function": {"name": "kb_search"}},
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
                use_kb=True,  # 前端现在开着，也要被 checkpoint 里的 False 覆盖
            )
        )

    assert seen_tools == [["web_search"]]


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


def test_run_loop_budget_exhaustion_forces_final_answer():
    """预算用尽：不再直接报错，剥掉工具强制模型基于已检索内容收尾。

    旧行为是 yield「已超过最大轮数」——用户手里只有模型说「我即将检索」
    的半截话，检索已经花了的成本全部白费。收尾必须给最终回答。
    """
    from ai_agent.agents.budget import Budget

    tool_args: list[list] = []

    def stream(messages, tools, **kwargs):
        tool_args.append(list(tools))
        if tools:  # 正常轮：模型反复请求工具
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"c{len(messages)}",
                            "name": "get_weather",
                            "arguments": {"city": "杭州"},
                        }
                    ],
                },
            )
        else:  # 收尾轮：没有工具可用，只能回答
            yield "text", "根据检索结果，今天晴，20 度。"
            yield (
                "result",
                {"content": "根据检索结果，今天晴，20 度。", "tool_calls": []},
            )

    with (
        patch("ai_agent.agents.loop.stream_chat", side_effect=stream),
        patch("ai_agent.agents.loop.execute_tool", return_value="晴 20 度"),
    ):
        events = list(
            run_loop(
                [{"role": "user", "content": "循环"}],
                "u1",
                budget=Budget(max_steps=2),
            )
        )

    assert [e for e in events if e["type"] == "error"] == []
    done = events[-1]
    assert done["type"] == "done"
    # 收尾那次调用不再带工具
    assert tool_args[-1] == []
    assert any(
        e.get("type") == "text" and "今天晴" in str(e.get("content"))
        for e in events
    )


def test_run_loop_retries_when_final_answer_empty():
    """工具跑完但第二轮正文为空：再逼一轮不带工具，把回答补出来。"""
    calls = {"n": 0}
    tool_args: list[list] = []

    def stream(messages, tools, **kwargs):
        calls["n"] += 1
        tool_args.append(list(tools))
        if calls["n"] == 1:
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "get_weather",
                            "arguments": {"city": "杭州"},
                        }
                    ],
                },
            )
            return
        if tools:
            yield ("result", {"content": "", "tool_calls": []})
            return
        yield ("text", "杭州今天晴。")
        yield ("result", {"content": "杭州今天晴。", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.stream_chat", side_effect=stream),
        patch("ai_agent.agents.loop.execute_tool", return_value="晴 20 度"),
    ):
        events = list(
            run_loop(
                [{"role": "user", "content": "杭州天气"}],
                "u1",
            )
        )

    assert [e for e in events if e["type"] == "error"] == []
    assert events[-1]["type"] == "done"
    assert tool_args[-1] == []
    assert any(
        e.get("type") == "text" and "杭州今天晴" in str(e.get("content"))
        for e in events
    )
    assert calls["n"] == 3


def test_run_loop_dedupes_identical_tool_calls():
    """完全相同的 (工具, 参数) 只执行一次，重复请求收到提示而非结果。

    实测本地模型会原样重发同一个 kg_search 六次，每次都返回相同的
    实体列表，把预算轮数烧光后没有轮次留给最终回答。
    """
    from ai_agent.agents.budget import Budget

    executed: list[str] = []

    def stream(messages, tools, **kwargs):
        # 模拟真实死循环：模型前两轮原样重发同一个 kg_search，
        # 收到「已调用过」的提示后才老实作答
        tool_rounds = sum(1 for m in messages if m.get("role") == "tool")
        if tool_rounds >= 2:
            yield "text", "答"
            yield ("result", {"content": "答", "tool_calls": []})
        else:
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"c{len(messages)}",
                            "name": "kg_search",
                            "arguments": {"query": "同一句话"},
                        }
                    ],
                },
            )

    def execute(name, arguments):
        executed.append(arguments["query"])
        return "命中实体：甲公司"

    messages: list = [{"role": "user", "content": "查"}]
    with (
        patch("ai_agent.agents.loop.stream_chat", side_effect=stream),
        patch("ai_agent.agents.loop.execute_tool", side_effect=execute),
    ):
        events = list(run_loop(messages, "u1", budget=Budget(max_steps=4)))

    assert executed == ["同一句话"]  # 第二次同参数调用被短路，没有执行
    # 上下文里留了提示，模型知道为什么没拿到新结果
    hints = [
        m
        for m in messages
        if m.get("role") == "tool" and "完全相同的参数" in str(m.get("content"))
    ]
    assert len(hints) == 1
    done = events[-1]
    assert done["type"] == "done"


def test_run_loop_dedupes_call_with_default_args_omitted():
    """省略默认值的参数与写全默认值的参数算同一次调用。

    实测模型靠 {"query": X} / {"query": X, "top_k": 5} / {"depth": 2, ...}
    这类写法差异把同一个 kg_search 调了 9 次，把决策轮次烧光。
    """
    from ai_agent.agents.budget import Budget

    executed: list[dict] = []

    def stream(messages, tools, **kwargs):
        tool_rounds = sum(1 for m in messages if m.get("role") == "tool")
        if tool_rounds >= 2:
            yield "text", "答"
            yield ("result", {"content": "答", "tool_calls": []})
        else:
            # 第二轮补上 top_k=5（schema 默认值），语义与第一轮完全相同
            args = {"query": "硝酸铵", "files": ["硝酸铵相关.pdf"]} if tool_rounds == 0 else {
                "query": "硝酸铵",
                "files": ["硝酸铵相关.pdf"],
                "top_k": 5,
            }
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {"id": f"c{tool_rounds}", "name": "kg_search", "arguments": args}
                    ],
                },
            )

    def execute(name, arguments):
        executed.append(arguments)
        return "命中实体：湖南南岭"

    messages: list = [{"role": "user", "content": "查"}]
    with (
        patch("ai_agent.agents.loop.stream_chat", side_effect=stream),
        patch("ai_agent.agents.loop.execute_tool", side_effect=execute),
    ):
        list(run_loop(messages, "u1", budget=Budget(max_steps=4)))

    assert len(executed) == 1  # 补全默认值后判定为重复，第二次没执行
    assert any(
        m.get("role") == "tool" and "完全相同的参数" in str(m.get("content"))
        for m in messages
    )


def test_run_loop_flags_repeated_tool_output():
    """参数不同但结果一字不差时，追加提示把模型推向换词或收尾。"""
    from ai_agent.agents.budget import Budget

    def stream(messages, tools, **kwargs):
        tool_rounds = sum(1 for m in messages if m.get("role") == "tool")
        if tool_rounds >= 2:
            yield "text", "答"
            yield ("result", {"content": "答", "tool_calls": []})
        else:
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"c{tool_rounds}",
                            "name": "kg_search",
                            "arguments": {"query": f"第{tool_rounds}种问法"},
                        }
                    ],
                },
            )

    def execute(name, arguments):
        # 图谱写死返回同一个中心实体：换检索词也是这份输出
        return "命中实体：湖南南岭民用爆破器材股份有限公司"

    messages: list = [{"role": "user", "content": "查"}]
    with (
        patch("ai_agent.agents.loop.stream_chat", side_effect=stream),
        patch("ai_agent.agents.loop.execute_tool", side_effect=execute),
    ):
        list(run_loop(messages, "u1", budget=Budget(max_steps=4)))

    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert "已调用过" not in str(tool_msgs[1].get("content"))  # 参数确实不同
    assert "返回过完全相同的结果" in str(tool_msgs[1]["content"])


def test_run_loop_budget_completion_defaults_to_settings():
    """不传 budget 时回落到 settings.max_turns，行为与旧版一致。"""
    with patch("ai_agent.agents.loop.stream_chat", side_effect=_ok_stream):
        events = list(run_loop([{"role": "user", "content": "你好"}], "u1"))

    assert events[-1] == {"type": "done"}
    assert settings.max_turns >= 1



def test_interrupted_turn_skips_long_term_capture():
    """中断/失败的半截回答不进长期记忆（完整回合才抽取）。"""
    from ai_agent.agents.loop import _capture_turn  # noqa: F401  确认可导入

    def failing(messages, tools, **kwargs):
        yield ("text", "半截")
        raise LLMError("boom")

    with (
        patch("ai_agent.agents.loop.stream_chat", side_effect=failing),
        patch("ai_agent.agents.turn.capture_long_term") as capture,
    ):
        events = list(
            run_stream("问题", user_id="u1", chat_session_id="s-interrupt")
        )

    assert events[-1]["type"] == "error"
    capture.assert_not_called()


def test_fit_context_drops_oldest_turns_keeps_summary_and_last():
    """超预算裁剪：丢最旧整轮，保留 system/摘要对/最后一个 turn，原列表不动。"""
    from ai_agent.agents.loop import _fit_context

    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": SUMMARY_USER_PREFIX + "摘要"},
        {"role": "assistant", "content": "已了解"},
        {"role": "user", "content": "A" * 500},
        {"role": "assistant", "content": "B" * 500},
        {"role": "user", "content": "C" * 500},
        {"role": "assistant", "content": "D" * 20},
    ]
    import copy

    snapshot = copy.deepcopy(messages)

    # 预算只够容纳 system+摘要+C/D 两轮，A/B 轮被丢
    fitted = _fit_context(messages, 1100)

    assert messages == snapshot  # 原列表不被修改
    roles = [m["role"] for m in fitted]
    assert fitted[0]["content"] == "SYS"
    assert fitted[1]["content"].startswith(SUMMARY_USER_PREFIX)
    assert fitted[-1]["content"] == "D" * 20
    assert "A" * 500 not in [m.get("content") for m in fitted]
    assert roles.count("user") >= 1  # 至少保留当前回合的 user
    # 不超预算时原样返回（同一对象，零开销路径）
    assert _fit_context(messages, 10**9) is messages
    # budget<=0 关闭功能
    assert _fit_context(messages, 0) is messages


def test_fit_context_never_orphans_tool_messages():
    """裁剪按整轮丢，assistant(tool_calls) 与 tool 结果不会被拆开。"""
    from ai_agent.agents.loop import _fit_context

    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U1" * 400},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "kb_search", "arguments": {"q": "x"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "kb_search", "content": "T" * 400},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
    ]
    fitted = _fit_context(messages, 900)

    tool_msgs = [m for m in fitted if m.get("role") == "tool"]
    calls = {
        c["id"]
        for m in fitted
        for c in (m.get("tool_calls") or [])
    }
    for msg in tool_msgs:
        assert msg["tool_call_id"] in calls  # 没有孤儿 tool 结果
    assert fitted[-1]["content"] == "A2"  # 当前回合保留


def test_permission_denied_is_not_reported_as_retryable():
    """403/401 不该提示「稍后重试」——重试永远不会成功，只会让用户反复点。"""
    from ai_agent.agents.loop import _llm_error_message

    for code in (401, 403):
        denied = _llm_error_message(LLMError("AccessDenied.Unpurchased", status_code=code))
        assert "稍后重试" not in denied
        assert "未开通" in denied or "无权限" in denied

    assert "繁忙" in _llm_error_message(LLMError("429", status_code=429))
    # 没有状态码（超时、连接断开）仍是可重试的坦白说法
    assert "稍后重试" in _llm_error_message(LLMError("timeout"))


def test_run_loop_emits_permission_message_for_403():
    """上游 403 时 SSE 下发的是「权限」提示，而不是通用的稍后重试。"""

    def _denied(messages, tools, **kwargs):
        raise LLMError("Error code: 403 - AccessDenied.Unpurchased", status_code=403)
        yield  # pragma: no cover

    with patch("ai_agent.agents.loop.stream_chat", _denied):
        events = list(
            run_loop(
                [{"role": "user", "content": "hi"}],
                user_id="u1",
                run_id="r1",
                use_web_search=False,
            )
        )

    errors = [e for e in events if e.get("type") == "error"]
    assert len(errors) == 1
    assert "未开通" in errors[0]["message"] or "无权限" in errors[0]["message"]
    assert "稍后重试" not in errors[0]["message"]


def test_mid_stream_error_is_reported_instead_of_killing_the_stream():
    """流建立后才出错（非 LLMError）也要转成 error 事件，不能让 SSE 直接断掉。"""

    def _boom(messages, tools, **kwargs):
        yield ("text", "半截回答")
        raise ConnectionResetError("peer reset")

    with patch("ai_agent.agents.loop.stream_chat", _boom):
        events = list(
            run_loop(
                [{"role": "user", "content": "hi"}],
                user_id="u1",
                run_id="r1",
                use_web_search=False,
            )
        )

    assert {"type": "text", "content": "半截回答"} in events
    errors = [e for e in events if e.get("type") == "error"]
    assert len(errors) == 1
    assert "中断" in errors[0]["message"]


def test_execute_tool_clamps_out_of_range_args():
    """模型传 depth=3（schema 上限 2）时，夹回边界重试，而不是整次失败。"""
    from ai_agent.tools.registry import _clamp_out_of_range_args
    from ai_agent.tools.kb.kg_search import KgSearchArgs

    clamped = _clamp_out_of_range_args(
        KgSearchArgs, {"query": "安克创新", "depth": 3, "top_k": 10}
    )
    assert clamped["depth"] == 2
    assert clamped["top_k"] == 10  # 未越界，保持原值


def test_clamp_only_touches_out_of_range_numeric_fields():
    from ai_agent.tools.registry import _clamp_out_of_range_args
    from ai_agent.tools.kb.kg_search import KgSearchArgs

    clamped = _clamp_out_of_range_args(
        KgSearchArgs, {"query": "安克创新", "depth": 0, "top_k": 99}
    )
    assert clamped["depth"] == 1
    assert clamped["top_k"] == 10
    assert clamped["query"] == "安克创新"  # 非数值字段不动



def test_report_job_id_lands_on_final_assistant_message():
    """研报任务 id 要随回答落库：刷新后前端凭它续上卡片、把下载链接长回来。"""
    from ai_agent.tools.progress import emit_event

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
                            "id": "call_job",
                            "name": "generate_report",
                            "arguments": {"request": "生成中国银行 2026Q3 研报"},
                        }
                    ],
                },
            )
            return
        yield ("text", "已提交，稍后回来看结果")
        yield ("result", {"content": "已提交，稍后回来看结果", "tool_calls": []})

    def fake_execute(_name, _arguments):
        emit_event(
            {
                "type": "report_job",
                "job_id": "job-xyz",
                "poll_url": "/v1/report/jobs/job-xyz",
            }
        )
        return "已提交研报生成任务：job_id=job-xyz"

    with (
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
        patch("ai_agent.agents.loop.execute_tool", side_effect=fake_execute),
        patch("ai_agent.agents.loop.store.append_chat") as append,
    ):
        run("生成中国银行 2026Q3 研报", chat_session_id="s-job")

    turn = append.call_args.args[2]
    assert turn[-1]["role"] == "assistant"
    assert turn[-1]["meta"]["report_job_id"] == "job-xyz"
