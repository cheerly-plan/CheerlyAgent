"""上下文裁剪的协议级不变量。

无论走老化、成对删除、整轮丢弃还是兜底裁剪，送给模型的消息都必须满足
OpenAI 兼容协议的两个硬约束：

1. 每个 ``assistant(tool_calls)`` 的 call id 都有对应的 ``tool`` 消息；
2. 没有 ``tool`` 消息指向不存在的 call id。

违反任何一条都会被网关直接 400，而这类回归很容易在新增裁剪手段时引入。
"""

import pytest

from ai_agent.agents.messages import _fit_context, age_tool_results
from ai_agent.memory.prune import prune_tool_pairs
from ai_agent.memory.short_term import fit_turns, messages_tokens


def _assert_protocol(messages):
    calls = {
        str(call.get("id"))
        for msg in messages
        for call in (msg.get("tool_calls") or [])
    }
    tool_ids = {
        str(msg.get("tool_call_id"))
        for msg in messages
        if msg.get("role") == "tool"
    }
    assert tool_ids <= calls, f"孤儿 tool 消息: {tool_ids - calls}"


def _tool_turn(n: int, chars: int = 400) -> list[dict]:
    return [
        {"role": "user", "content": f"问题{n}"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": f"call{n}", "name": "kb_search", "arguments": {"query": "q"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call{n}",
            "name": "kb_search",
            "content": f"[文件{n} · 第{n}页]\n" + "数" * chars,
            "meta": {"tool": "kb_search", "raw_chars": chars},
        },
        {"role": "assistant", "content": f"回答{n}"},
    ]


def _conversation(turns: int, chars: int = 400) -> list[dict]:
    messages = [{"role": "system", "content": "SYS"}]
    for n in range(1, turns + 1):
        messages.extend(_tool_turn(n, chars))
    return messages


@pytest.mark.parametrize("turns", [1, 2, 3, 6, 12])
@pytest.mark.parametrize("budget", [300, 1200, 5000, 40000])
def test_fit_context_never_orphans_tool_messages(turns, budget):
    messages = _conversation(turns)
    fitted = _fit_context(messages, budget, soft_limit=budget)
    _assert_protocol(fitted)


def test_fit_context_respects_budget_after_aging_and_pruning():
    messages = _conversation(8, chars=600)
    budget = 2000
    fitted = _fit_context(messages, budget, soft_limit=budget)
    assert messages_tokens(fitted) <= budget
    _assert_protocol(fitted)


def test_fit_context_keeps_last_turn_intact():
    messages = _conversation(6, chars=600)
    fitted = _fit_context(messages, 1500, soft_limit=1500)
    last_user = max(
        index for index, msg in enumerate(fitted) if msg.get("role") == "user"
    )
    tail = fitted[last_user:]
    assert any(msg.get("role") == "tool" for msg in tail), "当前轮工具结果被裁掉了"
    call_ids = {
        str(call.get("id"))
        for msg in tail
        for call in (msg.get("tool_calls") or [])
    }
    assert {str(m.get("tool_call_id")) for m in tail if m.get("role") == "tool"} <= call_ids


def test_aging_then_prune_keeps_protocol():
    messages = _conversation(6)
    aged = age_tool_results(messages, target_tokens=messages_tokens(messages) // 2)
    pruned = prune_tool_pairs(aged, messages_tokens(aged) // 2)
    _assert_protocol(pruned)
    assert messages_tokens(pruned) <= messages_tokens(aged)


def test_fit_turns_drops_whole_turns_only():
    messages = _conversation(6)
    kept, dropped = fit_turns(messages, messages_tokens(messages) // 2)
    kept_msgs = [msg for turn in kept for msg in turn]
    _assert_protocol(kept_msgs)
    # 被丢的必须是完整轮次：每个被丢的 tool_call 和它的 tool 结果同在 dropped
    dropped_calls = {
        str(call.get("id"))
        for msg in dropped
        for call in (msg.get("tool_calls") or [])
    }
    dropped_tools = {
        str(msg.get("tool_call_id")) for msg in dropped if msg.get("role") == "tool"
    }
    assert dropped_tools <= dropped_calls
    # 被保留的轮次里也不允许出现半截
    kept_calls = {
        str(call.get("id"))
        for msg in kept_msgs
        for call in (msg.get("tool_calls") or [])
    }
    kept_tools = {
        str(msg.get("tool_call_id"))
        for msg in kept_msgs
        if msg.get("role") == "tool"
    }
    assert kept_tools <= kept_calls


def test_empty_conversation_is_untouched():
    messages = [{"role": "system", "content": "SYS"}]
    assert _fit_context(messages, 1000) == messages
