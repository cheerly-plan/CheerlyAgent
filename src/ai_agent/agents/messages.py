"""消息列表的裁剪与断点缝合。

从 loop.py 拆出(S5)。流式输出会中断：连接断开、客户端刷新、模型半途改主意。
续跑时要拿新生成的内容接回旧内容，接缝处模型偶尔复述最后一句，需要按最小
重叠去重；上下文超预算时按整轮丢弃，不能切碎一轮。
"""

from __future__ import annotations

import hashlib
import json

from ai_agent.agents.turn import _checkpoint_update
from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.short_term import (
    assemble,
    flatten,
    iter_turns,
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


def _previous_turns(messages: list[ChatMessage]) -> list[str]:
    """当前 user 之前最近 N 轮的「问句 + 最终回答」，时间正序。

    从当前 user 的前一条开始往前扫。不能从 messages[-1] 开始——那会把
    「当前这句」当成上一轮塞进去，改写上下文里就没有主体（公司名）。
    倒序扫描先遇到「回答」再遇到「问句」，所以每轮暂存成 (user, answer)，
    最后按时间正序摊平。
    """
    turns: list[tuple[str, str]] = []
    pending_answer = ""
    last = len(messages) - 1
    end = last - 1 if last >= 0 and messages[last].get("role") == "user" else last
    for i in range(end, -1, -1):
        msg = messages[i]
        role = msg.get("role")
        if role == "assistant":
            if msg.get("tool_calls"):
                continue
            text = str(msg.get("content") or "").strip()
            if text and not pending_answer:
                pending_answer = text
        elif role == "user":
            text = str(msg.get("content") or "").strip()
            if text:
                turns.append((text, pending_answer))
                pending_answer = ""
        if len(turns) >= settings.kb_rewrite_history_turns:
            break
    prev_turn: list[str] = []
    for user_q, answer in reversed(turns):
        prev_turn.append(user_q)
        if answer:
            prev_turn.append(answer)
    return prev_turn


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


def _messages_chars(messages: list[ChatMessage]) -> int:
    """上下文长度的粗估（字符数）。中文场景 1 字符 ≈ 0.6~1 token，
    用字符数做预算天然偏保守，不会把上下文撑爆。"""
    total = 0
    for msg in messages:
        total += len(str(msg.get("content") or ""))
        for call in msg.get("tool_calls") or []:
            total += len(str(call.get("arguments") or ""))
    return total


def _fit_context(messages: list[ChatMessage], max_chars: int) -> list[ChatMessage]:
    """超预算时对送模型的副本做裁剪：从最旧的完整轮次开始整轮丢弃。

    只动副本，原列表原封不动——落库、checkpoint 快照都依赖原始 messages，
    裁剪绝不能影响它们。整轮（user + 其后的 assistant/tool）一起丢，
    保证 assistant(tool_calls) 和它的 tool 结果永远成对，不会因裁剪
    出现孤儿 tool 消息（会被 API 400）。摘要对保留，丢了早期轮次的
    信息至少还有摘要兜底；最后一个 turn 是当前进行中的回合，永不丢。
    """
    if max_chars <= 0 or _messages_chars(messages) <= max_chars:
        return messages
    system, rest = take_system(messages)
    summary, rest = take_summary(rest)
    turns = iter_turns(rest)
    while len(turns) > 1:
        turns.pop(0)
        if _messages_chars([*system, *flatten(turns)]) <= max_chars:
            break
    sys_text = str(system[0].get("content") or "") if system else ""
    fitted = assemble(sys_text, summary, flatten(turns))
    if not sys_text and fitted and fitted[0].get("role") == "system":
        fitted = fitted[1:]
    logger.info(
        "上下文超预算已裁剪 chars=%d budget=%d kept_turns=%d",
        _messages_chars(messages),
        max_chars,
        len(turns),
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
