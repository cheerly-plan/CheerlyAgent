"""成对裁剪工具调用：只删 assistant(tool_calls) + 对应 tool 消息。

为什么必须成对：OpenAI 兼容协议要求每个 tool_call 都有对应的 tool 消息。
只删占位会留下孤儿 tool_call，下一次请求直接 400。

两种触发：
1. consumed_only：该 call_id 已经被 read_tool_result 读过，且当前消息是
   老化占位——占位凭证已消费，可以连同 assistant 调用一起删。
2. 预算不足时的低价值 LRU：优先删最旧、无 citations 的工具对，给当前
   回合腾空间；当前轮永不删除。
"""

from __future__ import annotations

from ai_agent.llm.tokenizer import count_messages_tokens
from ai_agent.llm.types import ChatMessage

_AGED_MARK = "原文已省略"


def _tool_call_ids(msg: ChatMessage) -> list[str]:
    return [
        str(call.get("id") or "")
        for call in (msg.get("tool_calls") or [])
        if call.get("id")
    ]


def consumed_call_ids(messages: list[ChatMessage]) -> set[str]:
    """找出已经被 read_tool_result 读过的原始 call_id。"""
    out: set[str] = set()
    for msg in messages:
        for call in msg.get("tool_calls") or []:
            if call.get("name") != "read_tool_result":
                continue
            args = call.get("arguments") or {}
            target = str(args.get("tool_call_id") or "").strip()
            if target:
                out.add(target)
    return out


def _find_parent(messages: list[ChatMessage], call_id: str) -> int | None:
    for index, msg in enumerate(messages):
        if call_id in _tool_call_ids(msg):
            return index
    return None


def _remove_pair(
    messages: list[ChatMessage], call_id: str
) -> list[ChatMessage] | None:
    parent = _find_parent(messages, call_id)
    if parent is None:
        return None
    tool_indexes = [
        index
        for index, msg in enumerate(messages)
        if msg.get("role") == "tool" and str(msg.get("tool_call_id") or "") == call_id
    ]
    if not tool_indexes:
        return None
    out = list(messages)
    for index in reversed(tool_indexes):
        out.pop(index)
    # 重新找 parent：上面的 pop 可能改变下标
    parent = _find_parent(out, call_id)
    if parent is None:
        return out
    assistant = dict(out[parent])
    calls = [
        call
        for call in (assistant.get("tool_calls") or [])
        if str(call.get("id") or "") != call_id
    ]
    if calls:
        assistant["tool_calls"] = calls
        out[parent] = assistant
    elif not str(assistant.get("content") or "").strip():
        out.pop(parent)
    else:
        assistant.pop("tool_calls", None)
        out[parent] = assistant
    return out


def prune_tool_pairs(
    messages: list[ChatMessage],
    target_tokens: int,
    *,
    consumed_only: bool = False,
) -> list[ChatMessage]:
    """按预算成对删除工具调用；当前轮（最后一条 user 之后）不动。"""
    if target_tokens <= 0 or count_messages_tokens(messages) <= target_tokens:
        return messages
    last_user = -1
    for index, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user = index
    if last_user < 0:
        return messages

    consumed = consumed_call_ids(messages) if consumed_only else set()
    candidates: list[tuple[int, int, str]] = []
    for index, msg in enumerate(messages):
        if index > last_user or msg.get("role") != "tool":
            continue
        call_id = str(msg.get("tool_call_id") or "").strip()
        if not call_id:
            continue
        content = str(msg.get("content") or "")
        if consumed_only:
            if call_id not in consumed or _AGED_MARK not in content:
                continue
        # read_tool_result 自己的结果不参与低价值 LRU，避免把刚读回的原文删掉
        if msg.get("name") == "read_tool_result":
            continue
        meta = msg.get("meta") if isinstance(msg.get("meta"), dict) else {}
        cited = bool(meta.get("citations"))
        # 老的、无 citations 的优先；同优先级按出现顺序从旧到新
        candidates.append((1 if cited else 0, index, call_id))

    if not candidates:
        return messages
    candidates.sort(key=lambda item: (item[0], item[1]))
    out = messages
    for _cited, _index, call_id in candidates:
        if count_messages_tokens(out) <= target_tokens:
            break
        new = _remove_pair(out, call_id)
        if new is not None and len(new) < len(out):
            out = new
    return out


def removed_tool_call_ids(
    before: list[ChatMessage], after: list[ChatMessage]
) -> set[str]:
    old = {
        str(msg.get("tool_call_id") or "")
        for msg in before
        if msg.get("role") == "tool" and msg.get("tool_call_id")
    }
    new = {
        str(msg.get("tool_call_id") or "")
        for msg in after
        if msg.get("role") == "tool" and msg.get("tool_call_id")
    }
    return old - new
