"""消息列表的裁剪与断点缝合。

从 loop.py 拆出(S5)。流式输出会中断：连接断开、客户端刷新、模型半途改主意。
续跑时要拿新生成的内容接回旧内容，接缝处模型偶尔复述最后一句，需要按最小
重叠去重；上下文超预算时按整轮丢弃，不能切碎一轮。
"""

from __future__ import annotations

import hashlib
import json

from ai_agent.agents.turn import _checkpoint_update
from ai_agent.core.logging import get_logger
from ai_agent.core.text import clip_text
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.short_term import (
    assemble,
    fit_turns,
    flatten,
    messages_chars,
    take_summary,
    take_system,
)
from ai_agent.tools.registry import TOOLS_MAP

logger = get_logger(__name__)


# 接缝去重的最小重叠长度：模型续写偶尔会复述最后一句，通常整句 ≥3 字；
# 更短的重叠大概率是巧合（如"的"+"的话"），不能误删
_SEAM_MIN_OVERLAP = 3


def _canonical_args(name: str, arguments: dict) -> str:
    """把工具参数按 args_model 补全默认值再序列化，用于「重复调用」判定。

    模型会省略可省略的字段：`{"query": X}` 与 `{"query": X, "top_k": 5}`
    语义完全一样，但 JSON 字面不同。只比字面的话，这种「换个写法再查一次」
    会绕过去重——实测同一句问题下，模型靠 depth/top_k 的写法差异把同一个
    kg_search 调了 9 次，把决策轮次烧光。
    """
    tool = TOOLS_MAP.get(name)
    canon: object = arguments
    if tool is not None:
        try:
            canon = tool.args_model.model_validate(arguments).model_dump(mode="json")
        except Exception:
            # 参数本身不合法（执行时也会报错），退回字面比较即可
            canon = arguments
    return json.dumps(canon, sort_keys=True, ensure_ascii=False, default=str)


def _output_digest(output: str) -> str:
    return hashlib.sha1(output.strip().encode("utf-8")).hexdigest()


def _drop_empty_assistant(messages: list[ChatMessage]) -> None:
    """空正文收尾前摘掉刚写入的空气泡，避免上下文里连续两条 assistant。"""
    last = messages[-1] if messages else None
    if (
        last
        and last.get("role") == "assistant"
        and not last.get("tool_calls")
        and not str(last.get("content") or "").strip()
    ):
        messages.pop()


def _stash_partial_assistant(
    messages: list[ChatMessage],
    streamed: str,
    run_id: str | None,
    step: int,
) -> None:
    """流被掐断或失败时把已吐出的半截 assistant 写进 messages，
    刷新后历史还能看到，checkpoint 也能据此恢复。"""
    if not streamed:
        return
    last = messages[-1] if messages else None
    if last and last.get("role") == "assistant" and not last.get("tool_calls"):
        # prefill 续写场景：last 就是送给模型的半截种子，本次流的增量
        # 必须接在旧内容后面，整体覆盖会丢掉断点前的文本
        last["content"] = str(last.get("content") or "") + streamed
    else:
        messages.append({"role": "assistant", "content": streamed})
    if run_id:
        _checkpoint_update(run_id, messages, step + 1)


def _dedupe_seam(head: str, tail: str) -> str:
    """去掉 tail 开头与 head 结尾的重叠段（模型续写时偶尔复述最后一句）。
    只处理较长的重叠，短前缀巧合不误删。"""
    limit = min(len(head), len(tail))
    for k in range(limit, _SEAM_MIN_OVERLAP - 1, -1):
        if head.endswith(tail[:k]):
            return tail[k:]
    return tail


def _pop_incomplete_assistant(messages: list[ChatMessage]) -> list[ChatMessage]:
    """（resume_prefill=False 的回退路径）半截 assistant 只给前端展示，
    不送给模型：从上一轮完整消息接着跑，整段替换半截回答。"""
    if (
        messages
        and messages[-1].get("role") == "assistant"
        and not messages[-1].get("tool_calls")
    ):
        return messages[:-1]
    return messages


_SYSTEM_CLIP_MARK = "\n[系统提示词动态部分已省略，请按需用工具查询]\n"


def _enforce_budget(fitted: list[ChatMessage], max_chars: int) -> list[ChatMessage]:
    """整轮丢完仍超预算时的兜底，绝不把超限请求直接发给模型。

    system 的动态块（画像 / 知识库能力清单 / 日期）拼在尾部，原先完全不受
    预算约束；单轮里多次工具调用也可能把最后一个 turn 撑到超预算。按顺序
    降级：裁 system 尾部 → 压缩当前回合的 tool 结果（保留 tool_call_id，
    不产生孤儿消息）。fitted 是 assemble 产出的副本，改动不碰原始 messages。
    """
    if messages_chars(fitted) <= max_chars:
        return fitted
    for idx, msg in enumerate(fitted):
        if messages_chars(fitted) <= max_chars:
            break
        if msg.get("role") != "system":
            continue
        content = str(msg.get("content") or "")
        # 短 system 换成省略标记反而更长，直接跳过，让 tool 结果去吸收超额
        if len(content) <= len(_SYSTEM_CLIP_MARK):
            continue
        excess = messages_chars(fitted) - max_chars
        keep = max(0, len(content) - excess - len(_SYSTEM_CLIP_MARK))
        fitted[idx] = {**msg, "content": content[:keep] + _SYSTEM_CLIP_MARK}
    for idx, msg in enumerate(fitted):
        if messages_chars(fitted) <= max_chars:
            break
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content") or "")
        excess = messages_chars(fitted) - max_chars
        fitted[idx] = {
            **msg,
            "content": clip_text(content, max(0, len(content) - excess)),
        }
    if messages_chars(fitted) > max_chars:
        logger.error(
            "上下文裁剪后仍超预算 chars=%d budget=%d",
            messages_chars(fitted),
            max_chars,
        )
    return fitted


_AGED_MARK = "〔本结果 {chars} 字原文已省略{note}；需要细节请重新调用该工具〕"


def age_tool_results(
    messages: list[ChatMessage], keep_turns: int | None = None
) -> list[ChatMessage]:
    """把**更早轮次**的工具原文换成一行引用，只留最近 ``keep_turns`` 轮原文。

    当前这轮的工具结果必须完整：模型正在读它、要直接引用里面的数字。
    真正该精简的是历史轮次——实测 5 轮窗口里工具输出占 96%，而越早的
    检索结果与当前问题越远，却仍以原文占着位置。

    换掉的只是送模型的那份副本：落库、checkpoint、前端引用都不动。省略行里
    带上当时的调用参数，模型需要细节时照着再查一次即可（一次检索 0.3 秒，
    比每轮背几万字便宜）。
    """
    from ai_agent.core.config import settings

    keep = settings.tool_result_full_turns if keep_turns is None else keep_turns
    if keep is None or keep < 0:
        return messages
    # 从后往前数 user 消息，定位「轮次」边界
    user_positions = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not user_positions:
        return messages
    # keep=0 的语义是「只保留当前轮」，边界要落在**最后一条 user 消息**上；
    # 直接用 len(messages) 会把当前轮正在用的工具结果也压掉。
    if keep and len(user_positions) <= keep:
        return messages
    boundary = user_positions[-keep] if keep else user_positions[-1]
    calls: dict[str, str] = {}
    for msg in messages:
        for call in msg.get("tool_calls") or []:
            args = call.get("arguments") or {}
            brief = json.dumps(args, ensure_ascii=False)
            calls[call.get("id", "")] = f"{call.get('name')}({clip_text(brief, 60)})"
    out: list[ChatMessage] = []
    changed = False
    for index, msg in enumerate(messages):
        if index >= boundary or msg.get("role") != "tool":
            out.append(msg)
            continue
        content = str(msg.get("content") or "")
        if len(content) <= 200:
            out.append(msg)          # 短的没必要换，标记本身可能更长
            continue
        head = content.splitlines()[0].strip()
        note = calls.get(msg.get("tool_call_id", ""), "")
        stub = _AGED_MARK.format(
            chars=len(content), note=f"，检索：{note}" if note else ""
        )
        out.append({**msg, "content": f"{clip_text(head, 80)}\n{stub}"})
        changed = True
    # 没有可老化的内容时返回原对象：调用方依赖「预算充裕就原样返回」的语义
    return out if changed else messages


def _fit_context(messages: list[ChatMessage], max_chars: int) -> list[ChatMessage]:
    """超预算时对送模型的副本做裁剪：从最旧的完整轮次开始整轮丢弃。

    只动副本，原列表原封不动——落库、checkpoint 快照都依赖原始 messages。
    整轮（user + 其后的 assistant/tool）一起丢，保证 assistant(tool_calls)
    和它的 tool 结果永远成对，不会因裁剪出现孤儿 tool 消息（会被 API 400）。
    被丢的轮次由收尾的 compress() 用同一个预算并入会话摘要，不再静默蒸发。
    """
    if max_chars <= 0:
        return messages
    # 先给历史轮次的工具原文"老化"成一行引用：这是常规动作，不是超预算才做。
    # 当前轮永远保留原文（模型正在读它），省的是早就用不上的旧检索结果。
    messages = age_tool_results(messages)
    if messages_chars(messages) <= max_chars:
        return messages
    system, rest = take_system(messages)
    summary, _ = take_summary(rest)
    turns, dropped = fit_turns(messages, max_chars)
    sys_text = str(system[0].get("content") or "") if system else ""
    fitted = assemble(sys_text, summary, flatten(turns))
    if not sys_text and fitted and fitted[0].get("role") == "system":
        fitted = fitted[1:]
    fitted = _enforce_budget(fitted, max_chars)
    logger.info(
        "上下文超预算已裁剪 chars=%d budget=%d kept_turns=%d dropped_msgs=%d",
        messages_chars(messages),
        max_chars,
        len(turns),
        len(dropped),
    )
    return fitted


def _keep_prefill_tail(messages: list[ChatMessage]) -> bool:
    """（prefill 续写路径）半截 assistant 留在上下文尾部作为续写种子，
    并打上消息级 partial 标记：服务端据此从断点接着生成，不整段重开。
    返回是否存在可续写的半截。"""
    last = messages[-1] if messages else None
    if (
        last is not None
        and last.get("role") == "assistant"
        and not last.get("tool_calls")
        and last.get("content")
    ):
        last["content"] = str(last["content"]).rstrip()
        last["partial"] = True
        return True
    return False
