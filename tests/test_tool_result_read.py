"""通用工具结果读取：老化行指路 + 按 tool_call_id 取回全文。"""

import pytest

from ai_agent.agents.loop import run_loop
from ai_agent.agents.messages import age_tool_results
from ai_agent.storage.db import user_scope
from ai_agent.tools.base import ToolOutput
from ai_agent.tools.context import session_scope, tool_result_index, tool_results_scope
from ai_agent.tools.misc.tool_result import read_tool_result
from ai_agent.tools.registry import TOOLS_MAP
from ai_agent.tools.result_pool import ToolResultPool, tool_result_pool


@pytest.fixture(autouse=True)
def _clear_tool_result_pool():
    tool_result_pool.clear()
    yield
    tool_result_pool.clear()


def _tool_message(call_id: str, name: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


def test_tool_result_index_collects_original_content():
    messages = [
        _tool_message("call1", "web_search", "完整联网结果"),
        {"role": "assistant", "content": "答"},
    ]
    index = tool_result_index(messages)
    assert index["call1"] == {"name": "web_search", "content": "完整联网结果"}


def test_read_tool_result_returns_full_text_and_paginates():
    content = "0123456789" * 100
    with tool_results_scope(tool_result_index([_tool_message("call1", "web_search", content)])):
        first = read_tool_result("call1", max_chars=200)
        assert "[web_search · call1]" in first
        assert "已读 200/1000" in first
        second = read_tool_result("call1", offset=200, max_chars=200)
        assert "已读 400/1000" in second
        assert "0123456789" in first and "0123456789" in second


def test_read_tool_result_handles_unknown_and_empty_scope():
    assert "没有找到" in read_tool_result("missing")
    with tool_results_scope({}):
        assert "没有找到" in read_tool_result("missing")


def test_aged_stub_points_to_generic_tool_for_non_kb_result():
    messages = [
        {"role": "user", "content": "问1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call1", "name": "web_search", "arguments": {"query": "q"}}
            ],
        },
        _tool_message("call1", "web_search", "联网结果" + "字" * 900),
        {"role": "user", "content": "问2"},
    ]
    stub = age_tool_results(messages, keep_turns=1)[2]["content"]
    assert 'read_tool_result(tool_call_id="call1")' in stub
    assert "kb_read(" not in stub


def test_aged_kb_stub_keeps_both_generic_and_kb_hints():
    messages = [
        {"role": "user", "content": "问1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call1", "name": "kb_search", "arguments": {"query": "q"}}
            ],
        },
        _tool_message("call1", "kb_search", "[某公司 · 第10页]\n" + "数" * 900),
        {"role": "user", "content": "问2"},
    ]
    stub = age_tool_results(messages, keep_turns=1)[2]["content"]
    assert 'read_tool_result(tool_call_id="call1")' in stub
    assert "kb_read(location=" in stub


def test_registry_exposes_generic_reader():
    assert "read_tool_result" in TOOLS_MAP


def test_run_loop_sets_tool_result_scope_before_dispatch():
    seen: dict[str, str] = {}

    def fake_stream(messages, tools, tool_choice="auto"):
        if not seen:
            seen["phase"] = "tool"
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "read1",
                            "name": "read_tool_result",
                            "arguments": {"tool_call_id": "old1"},
                        }
                    ],
                },
            )
            return
        yield ("text", "completed")
        yield ("result", {"content": "completed", "tool_calls": []})

    messages = [
        {"role": "user", "content": "之前查一下"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "old1", "name": "web_search", "arguments": {"query": "旧问题"}}
            ],
        },
        _tool_message("old1", "web_search", "OLD-FULL-TEXT"),
        {"role": "user", "content": "把刚才的原文给我"},
    ]

    import ai_agent.agents.loop as loop

    original = loop.stream_chat
    loop.stream_chat = fake_stream
    try:
        events = list(run_loop(messages, use_kb=False))
    finally:
        loop.stream_chat = original

    assert any(event.get("type") == "done" for event in events)
    returned = [
        msg["content"]
        for msg in messages
        if msg.get("role") == "tool" and msg.get("tool_call_id") == "read1"
    ]
    assert returned and "OLD-FULL-TEXT" in returned[0]


def test_read_tool_result_falls_back_to_process_pool():
    with tool_results_scope({}), user_scope("u1"), session_scope("s1"):
        tool_result_pool.put("u1", "s1", "pool1", "web_search", "POOL-FULL")
        out = read_tool_result("pool1")
    assert "POOL-FULL" in out
    assert "[web_search · pool1]" in out


def test_tool_result_pool_ttl_and_lru():
    now = [0.0]
    pool = ToolResultPool(max_entries=2, ttl_seconds=10, clock=lambda: now[0])
    pool.put("u1", "s1", "c1", "web_search", "one")
    pool.put("u1", "s1", "c2", "web_search", "two")
    now[0] = 5
    assert pool.get("u1", "s1", "c1")["content"] == "one"
    pool.put("u1", "s1", "c3", "web_search", "three")
    assert pool.get("u1", "s1", "c2") is None  # LRU 最早被逐出
    now[0] = 20
    assert pool.get("u1", "s1", "c3") is None  # TTL 过期


def test_tool_result_pool_truncates_large_content():
    pool = ToolResultPool(max_content_chars=8)
    pool.put("u1", "s1", "big", "web_search", "0123456789")
    item = pool.get("u1", "s1", "big")
    assert item["content"] == "01234567"
    assert item["truncated"] is True


def test_run_loop_puts_raw_output_in_pool():
    seen: dict[str, str] = {}

    def fake_stream(messages, tools, tool_choice="auto"):
        if not seen:
            seen["phase"] = "tool"
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_pool",
                            "name": "web_search",
                            "arguments": {"query": "q"},
                        }
                    ],
                },
            )
            return
        yield ("text", "done")
        yield ("result", {"content": "done", "tool_calls": []})

    def fake_execute(name, arguments):
        return ToolOutput(
            "CLIPPED",
            raw="FULL-RAW-CONTENT",
            meta={"tool": name, "raw_chars": 15},
        )

    import ai_agent.agents.loop as loop

    original_stream = loop.stream_chat
    original_execute = loop.execute_tool
    loop.stream_chat = fake_stream
    loop.execute_tool = fake_execute
    try:
        list(
            run_loop(
                [{"role": "user", "content": "查一下"}],
                "u1",
                chat_session_id="s1",
                use_kb=False,
            )
        )
    finally:
        loop.stream_chat = original_stream
        loop.execute_tool = original_execute

    pooled = tool_result_pool.get("u1", "s1", "call_pool")
    assert pooled is not None
    assert pooled["content"] == "FULL-RAW-CONTENT"
    assert pooled["meta"]["raw_chars"] == 15


def test_aged_stub_uses_citations_from_meta():
    messages = [
        {"role": "user", "content": "问1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "kb_search", "arguments": {"query": "q"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "kb_search",
            "content": "正文" * 500,
            "meta": {
                "tool": "kb_search",
                "raw_chars": 5000,
                "citations": [
                    {"filename": "绿的谐波2026Q3.pdf", "page": 12, "kind": "table"}
                ],
            },
        },
        {"role": "user", "content": "问2"},
    ]
    stub = age_tool_results(messages, keep_turns=1)[2]["content"]
    assert "绿的谐波2026Q3.pdf · 第12页" in stub
    assert "原文已省略" in stub


def test_read_tool_result_falls_back_to_db(monkeypatch):
    from ai_agent.storage.store import store

    monkeypatch.setattr(
        store,
        "load_tool_result",
        lambda user, session, call_id: {
            "name": "web_search",
            "content": "DB-FULL-TEXT",
            "meta": {},
        },
    )
    with tool_results_scope({}), user_scope("u1"), session_scope("s1"):
        out = read_tool_result("db1")
    assert "DB-FULL-TEXT" in out
    assert "[web_search · db1]" in out
