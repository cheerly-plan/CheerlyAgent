"""成对裁剪工具调用：不能留下孤儿 tool 消息。"""

from ai_agent.memory.prune import prune_tool_pairs
from ai_agent.memory.short_term import messages_tokens
from ai_agent.tools.result_pool import ToolResultPool


def _call(call_id: str, name: str, args: dict | None = None) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "name": name, "arguments": args or {}}],
    }


def _tool(call_id: str, name: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
        "meta": {},
    }


def _assert_paired(messages):
    calls = {
        str(call.get("id"))
        for msg in messages
        for call in (msg.get("tool_calls") or [])
    }
    tools = {
        str(msg.get("tool_call_id"))
        for msg in messages
        if msg.get("role") == "tool"
    }
    assert tools <= calls


def test_consumed_placeholder_pair_is_removed():
    messages = [
        {"role": "user", "content": "问1"},
        _call("c1", "web_search"),
        _tool("c1", "web_search", "〔工具结果已归档，原文已省略〕"),
        {"role": "user", "content": "问2"},
        _call("r1", "read_tool_result", {"tool_call_id": "c1"}),
        _tool("r1", "read_tool_result", "完整原文"),
        {"role": "user", "content": "问3"},
    ]
    pruned = prune_tool_pairs(
        messages, messages_tokens(messages) - 1, consumed_only=True
    )
    _assert_paired(pruned)
    ids = {m.get("tool_call_id") for m in pruned if m.get("role") == "tool"}
    assert "c1" not in ids
    assert "r1" in ids
    assert "完整原文" in [m.get("content") for m in pruned]


def test_low_value_lru_pair_removed_but_current_turn_kept():
    messages = [
        {"role": "user", "content": "问1"},
        _call("old", "web_search"),
        _tool("old", "web_search", "旧" * 500),
        {"role": "user", "content": "问2"},
        _call("cur", "web_search"),
        _tool("cur", "web_search", "当前" * 500),
        {"role": "user", "content": "问3"},
    ]
    pruned = prune_tool_pairs(messages, messages_tokens(messages) - 100)
    _assert_paired(pruned)
    ids = {m.get("tool_call_id") for m in pruned if m.get("role") == "tool"}
    assert "old" not in ids
    assert "cur" in ids
    assert "当前" * 500 in [m.get("content") for m in pruned]


def test_pool_evict_calls_removes_only_named_snapshot():
    pool = ToolResultPool()
    pool.put("u", "s", "a", "web_search", "A")
    pool.put("u", "s", "b", "web_search", "B")
    assert pool.evict_calls("u", "s", {"a"}) == 1
    assert pool.get("u", "s", "a") is None
    assert pool.get("u", "s", "b")["content"] == "B"
