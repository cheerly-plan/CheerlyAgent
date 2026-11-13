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
from ai_agent.llm import metrics
from ai_agent.llm.tokenizer import clip_text_to_tokens, count_tokens
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.prune import prune_tool_pairs
from ai_agent.memory.short_term import (
    assemble,
    fit_turns,
    flatten,
    messages_tokens,
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
_TOOL_CLIP_MARK = "\n[工具结果已省略]"


def _enforce_budget(fitted: list[ChatMessage], max_tokens: int) -> list[ChatMessage]:
    """整轮丢完仍超预算时的兜底，绝不把超限请求直接发给模型。

    system 的动态块（画像 / 知识库能力清单 / 日期）拼在尾部，原先完全不受
    预算约束；单轮里多次工具调用也可能把最后一个 turn 撑到超预算。按顺序
    降级：裁 system 尾部 → 压缩当前回合的 tool 结果（保留 tool_call_id，
    不产生孤儿消息）。fitted 是 assemble 产出的副本，改动不碰原始 messages。
    """
    if messages_tokens(fitted) <= max_tokens:
        return fitted
    for idx, msg in enumerate(fitted):
        if messages_tokens(fitted) <= max_tokens:
            break
        if msg.get("role") != "system":
            continue
        content = str(msg.get("content") or "")
        # 短 system 换成省略标记反而更长，直接跳过，让 tool 结果去吸收超额
        if count_tokens(content) <= count_tokens(_SYSTEM_CLIP_MARK):
            continue
        excess = messages_tokens(fitted) - max_tokens
        keep = max(0, count_tokens(content) - excess - count_tokens(_SYSTEM_CLIP_MARK))
        fitted[idx] = {
            **msg,
            "content": clip_text_to_tokens(content, keep, _SYSTEM_CLIP_MARK),
        }
    for idx, msg in enumerate(fitted):
        if messages_tokens(fitted) <= max_tokens:
            break
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content") or "")
        excess = messages_tokens(fitted) - max_tokens
        fitted[idx] = {
            **msg,
            "content": clip_text_to_tokens(
                content,
                max(0, count_tokens(content) - excess),
                _TOOL_CLIP_MARK,
            ),
        }
    if messages_tokens(fitted) > max_tokens:
        logger.error(
            "上下文裁剪后仍超预算 tokens=%d budget=%d",
            messages_tokens(fitted),
            max_tokens,
        )
    return fitted


_AGED_MARK = (
    "〔本结果 {chars} 字原文已省略{note}"
    "{ref_hint}{kb_hint}〕"
)
_KB_TOOLS = ("kb_search", "kg_search")


def _calls_brief(messages: list[ChatMessage]) -> dict[str, str]:
    calls: dict[str, str] = {}
    for msg in messages:
        for call in msg.get("tool_calls") or []:
            args = call.get("arguments") or {}
            brief = json.dumps(args, ensure_ascii=False)
            calls[call.get("id", "")] = f"{call.get('name')}({clip_text(brief, 60)})"
    return calls


def _aged_message(msg: ChatMessage, calls: dict[str, str]) -> ChatMessage:
    content = str(msg.get("content") or "")
    head = content.splitlines()[0].strip()
    call_id = str(msg.get("tool_call_id") or "").strip()
    raw_meta = msg.get("meta")
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    name = str(meta.get("tool") or msg.get("name") or "")
    note = calls.get(call_id, "")
    try:
        chars = int(meta.get("raw_chars") or len(content))
    except (TypeError, ValueError):
        chars = len(content)
    hint = ""
    citations = meta.get("citations") or []
    if citations:
        first = citations[0] if isinstance(citations[0], dict) else {}
        where = str(first.get("filename") or "").strip()
        page = first.get("page")
        if where and page is not None:
            where = f"{where} · 第{page}页"
        if where:
            extra = f" 等{len(citations)}处" if len(citations) > 1 else ""
            hint = f"；命中：{clip_text(where, 40)}{extra}"
    elif meta.get("preview"):
        hint = f"；开头：{clip_text(str(meta['preview']), 40)}"
    ref_hint = (
        f"；原文用 read_tool_result(tool_call_id=\"{call_id}\")"
        if call_id
        else ""
    )
    kb_hint = (
        "；知识库也可用 kb_read(location=\"文件名 · 第N页\")"
        if name in _KB_TOOLS
        else ""
    )
    stub = _AGED_MARK.format(
        chars=chars,
        note=f"，检索：{note}" if note else "",
        ref_hint=ref_hint,
        kb_hint=kb_hint,
    )
    if hint:
        stub = stub.replace("原文已省略", f"原文已省略{hint}", 1)
    return {**msg, "content": f"{clip_text(head, 80)}\n{stub}"}


def age_tool_results(
    messages: list[ChatMessage],
    keep_turns: int | None = None,
    *,
    target_tokens: int = 0,
) -> list[ChatMessage]:
    """按 token 成本/新旧/是否被引用，把历史工具原文换成元数据占位。

    - ``target_tokens > 0``：预算模式。只在总量超过目标时，按「可省 token
      数 × 新旧程度」从高到低收；有 citations 的结果降权（优先保留可溯源
      的内容）。当前轮的工具结果永不老化。
    - ``target_tokens == 0``：兼容旧调用，仍按 ``keep_turns`` 老化更早轮次；
      生产路径已改用 target_tokens。

    换掉的只是送模型副本：落库、checkpoint、前端引用都不动；原文在
    ``tool_result_pool`` / messages 里，可用 ``read_tool_result`` 读回。
    """
    if target_tokens > 0:
        total = messages_tokens(messages)
        if total <= target_tokens:
            return messages
        last_user = -1
        for index, msg in enumerate(messages):
            if msg.get("role") == "user":
                last_user = index
        if last_user < 0:
            return messages
        calls = _calls_brief(messages)
        candidates: list[tuple[float, int, ChatMessage, int]] = []
        for index, msg in enumerate(messages):
            if index > last_user or msg.get("role") != "tool":
                continue
            content = str(msg.get("content") or "")
            if len(content) <= 200:
                continue
            stub = _aged_message(msg, calls)
            savings = messages_tokens([msg]) - messages_tokens([stub])
            if savings <= 0:
                continue
            later_users = sum(
                1 for later in messages[index + 1 :] if later.get("role") == "user"
            )
            meta = msg.get("meta") if isinstance(msg.get("meta"), dict) else {}
            cited = bool(meta.get("citations"))
            score = savings * (1 + later_users * 0.25)
            if cited:
                score *= 0.5
            candidates.append((score, index, stub, savings))
        if not candidates:
            return messages
        candidates.sort(key=lambda item: item[0], reverse=True)
        out = list(messages)
        for _score, index, stub, savings in candidates:
            out[index] = stub
            total -= savings
            if total <= target_tokens:
                break
        return out

    from ai_agent.core.config import settings

    keep = settings.tool_result_full_turns if keep_turns is None else keep_turns
    if keep is None or keep < 0:
        return messages
    user_positions = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not user_positions:
        return messages
    if keep and len(user_positions) <= keep:
        return messages
    boundary = user_positions[-keep] if keep else user_positions[-1]
    calls = _calls_brief(messages)
    out: list[ChatMessage] = []
    changed = False
    for index, msg in enumerate(messages):
        if index >= boundary or msg.get("role") != "tool":
            out.append(msg)
            continue
        if len(str(msg.get("content") or "")) <= 200:
            out.append(msg)
            continue
        out.append(_aged_message(msg, calls))
        changed = True
    return out if changed else messages

def _fit_context(
    messages: list[ChatMessage],
    max_tokens: int,
    *,
    soft_limit: int = 0,
) -> list[ChatMessage]:
    """超预算时对送模型的副本做裁剪：从最旧的完整轮次开始整轮丢弃。

    只动副本，原列表原封不动——落库、checkpoint 快照都依赖原始 messages。
    整轮（user + 其后的 assistant/tool）一起丢，保证 assistant(tool_calls)
    和它的 tool 结果永远成对，不会因裁剪出现孤儿 tool 消息（会被 API 400）。
    被丢的轮次由收尾的 compress() 用同一个预算并入会话摘要，不再静默蒸发。

    ``soft_limit`` 是软水位：达到后先按「可省 token × 新旧 × 是否被引用」
    收缩工具结果，避免「刚好没到硬上限」时还保留太多旧工具原文，下一轮
    突然撞墙。
    """
    if max_tokens <= 0:
        return messages
    before = messages_tokens(messages)
    # 预算压力才老化工具结果：超过硬上限按硬上限收，超过软水位按软水位收。
    # 收哪些不再看固定轮数，而是按「可省 token × 新旧 × 是否被引用」排序。
    target = max_tokens if before > max_tokens else 0
    if target == 0 and soft_limit > 0 and before >= soft_limit:
        target = soft_limit
    if target > 0:
        if before >= soft_limit > 0:
            metrics.incr(metrics.CONTEXT_SOFT)
            logger.info(
                "上下文达到软水位 tokens=%d soft=%d；按工具结果成本开始收缩",
                before,
                soft_limit,
            )
        aged = age_tool_results(messages, target_tokens=target)
        # 先删已消费的占位对，再按低价值 LRU 删工具对；只动送模型副本。
        aged = prune_tool_pairs(aged, target, consumed_only=True)
        aged = prune_tool_pairs(aged, target)
    else:
        aged = messages
    if messages_tokens(aged) <= max_tokens:
        return aged
    messages = aged
    system, rest = take_system(messages)
    summary, _ = take_summary(rest)
    turns, dropped = fit_turns(messages, max_tokens)
    sys_text = str(system[0].get("content") or "") if system else ""
    fitted = assemble(sys_text, summary, flatten(turns))
    if not sys_text and fitted and fitted[0].get("role") == "system":
        fitted = fitted[1:]
    fitted = _enforce_budget(fitted, max_tokens)
    metrics.incr(metrics.CONTEXT_HARD)
    logger.info(
        "上下文超预算已裁剪 tokens=%d budget=%d soft=%d kept_turns=%d dropped_msgs=%d",
        messages_tokens(messages),
        max_tokens,
        soft_limit,
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
