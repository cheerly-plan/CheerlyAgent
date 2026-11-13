"""把一条会话的消息按用户提问切开，展开后面的 agent 过程。"""

from __future__ import annotations

import json
from typing import Any

from ai_agent.api.schemas import InspectStep, InspectTurn


def _arguments(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _tool_calls(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    calls: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        calls.append(
            {
                "id": str(item.get("id") or ""),
                "name": str(item.get("name") or ""),
                "arguments": _arguments(item.get("arguments")),
            }
        )
    return calls


def _tokens(msg: Any) -> tuple[int | None, int | None]:
    prompt = msg.get("prompt_tokens")
    completion = msg.get("completion_tokens")
    return (
        int(prompt) if prompt is not None else None,
        int(completion) if completion is not None else None,
    )


def sum_tokens(values: list[int | None]) -> int | None:
    """只要有一条采集到就求和；全都没有返回 None（老数据不显示成 0）。"""
    real = [v for v in values if v is not None]
    return sum(real) if real else None


def turns_from_messages(messages: list[Any]) -> list[InspectTurn]:
    """用户消息开一轮。其后的 assistant / tool 消息按发生顺序摊成步骤。"""
    turns: list[InspectTurn] = []
    steps: list[InspectStep] = []
    calls: dict[str, dict[str, Any]] = {}
    turn_prompts: list[int | None] = []
    turn_completions: list[int | None] = []
    question = ""
    opened = False

    def close() -> None:
        nonlocal opened, steps, question, calls, turn_prompts, turn_completions
        if not opened:
            return
        turns.append(
            InspectTurn(
                index=len(turns),
                question=question,
                steps=steps,
                prompt_tokens=sum_tokens(turn_prompts),
                completion_tokens=sum_tokens(turn_completions),
            )
        )
        steps = []
        calls = {}
        turn_prompts = []
        turn_completions = []
        question = ""
        opened = False

    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content") or "")
        if role == "user":
            close()
            question = content
            opened = True
            continue
        if not opened:
            continue
        if role == "assistant":
            prompt_tokens, completion_tokens = _tokens(msg)
            if prompt_tokens is not None:
                turn_prompts.append(prompt_tokens)
            if completion_tokens is not None:
                turn_completions.append(completion_tokens)
            if content.strip():
                steps.append(
                    InspectStep(
                        kind="llm",
                        content=content,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                )
            for call in _tool_calls(msg.get("tool_calls")):
                if call["id"]:
                    calls[call["id"]] = call
                steps.append(
                    InspectStep(
                        kind="tool",
                        name=call["name"],
                        tool_call_id=call["id"],
                        arguments=call["arguments"],
                    )
                )
        elif role == "tool":
            call_id = str(msg.get("tool_call_id") or "")
            matched = calls.get(call_id)
            name = str(msg.get("name") or "")
            if matched:
                name = name or str(matched["name"])
            steps.append(
                InspectStep(
                    kind="tool_result",
                    name=name,
                    tool_call_id=call_id,
                    content=content,
                    arguments=matched["arguments"] if matched else {},
                )
            )
    close()
    return turns
