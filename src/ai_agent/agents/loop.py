import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from ai_agent.agents.budget import Budget
from ai_agent.agents.prompts import SYSTEM_PROMPT
from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.kb.citations import Citation, collect_citations, merge_sources
from ai_agent.kb.retrieval import search_knowledge
from ai_agent.kb.scope import reset_kb_scope, set_kb_scope
from ai_agent.llm.client import stream_chat
from ai_agent.llm.types import ChatEvent, ChatMessage, ChatResult
from ai_agent.memory.facets import format_profile_block
from ai_agent.memory.long_term import capture_long_term
from ai_agent.memory.short_term import (
    assemble,
    compress,
    flatten,
    iter_turns,
    short_term_memory,
    take_summary,
    take_system,
)
from ai_agent.memory.store import store, user_scope
from ai_agent.memory.summarize import summarize_session
from ai_agent.tools.registry import execute_tool, tool_label, tool_schemas
from ai_agent.tools.kb_search import _citations_of, _format_hit_with_context

logger = get_logger(__name__)

# 接缝去重的最小重叠长度：模型续写偶尔会复述最后一句，通常整句 ≥3 字；
# 更短的重叠大概率是巧合（如"的"+"的话"），不能误删
_SEAM_MIN_OVERLAP = 3

# 流结束后的收尾工作（长期记忆抽取/摘要压缩含多次 LLM 调用）放到后台线程，
# 不再阻塞 SSE 尾部延迟；进程退出时由 lifespan 调 shutdown_turn_finalizer 等待。
_turn_finalizer = ThreadPoolExecutor(max_workers=4, thread_name_prefix="turn-finalize")


def shutdown_turn_finalizer(wait: bool = True) -> None:
    """等待在途的记忆抽取/摘要任务完成后再关停，避免重启丢长期记忆。"""
    _turn_finalizer.shutdown(wait=wait, cancel_futures=False)


def _llm_error_message(error: LLMError) -> str:
    """把上游错误翻成用户能照着做的一句话。

    403/401 重试永远不会好（模型未开通、key 无权限、账户欠费），
    对用户说「请稍后重试」只会让他反复点；必须指向真因。
    """
    if error.is_permission_denied:
        return "模型服务拒绝了本次调用（模型未开通或无权限），请联系管理员检查模型账户"
    if error.is_rate_limited:
        return "模型服务繁忙，请稍后再试"
    return "模型服务暂时不可用，请稍后重试"


def _profile_prompt(user_id: str) -> str:
    try:
        items = store.list_profile(user_id)
        logger.info("list_profile user_id=%s count=%s", user_id, len(items))
    except Exception:
        logger.exception("用户画像读取失败")
        return ""
    return format_profile_block(items)


def _system_prompt(user_id: str) -> str:
    block = _profile_prompt(user_id)
    if not block:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n{block}"


def _kb_prefetch(
    user_input: str, user_id: str, kb_file_ids: tuple[str, ...]
) -> tuple[ChatMessage | None, list[Citation]]:
    """kb 开启时先替模型查一次库，命中注入成 system 消息（always-retrieve）。

    为什么不等模型自己决定调不调 kb_search：本地模型对「这问题要不要查」
    的判断不稳定，提示词再怎么放宽也有漏网——用户把库里的原文粘进对话框，
    模型照样可能直接用通用知识作答（实测发生过）。检索是毫秒级的便宜操作，
    主动查一次的代价可以忽略；命中与否交给向量距离：闲聊时没有语义接近的
    片段，自然不会注入，不会把闲聊带偏。

    命中同时作为引用交给上层（SSE sources 事件 + 落库），和模型主动调
    kb_search 的命中在展示上完全一致。检索失败不阻断对话，只是这轮退回
    「靠模型自己调工具」的旧模式。
    """
    query = user_input.strip()
    if not query:
        return None, []
    scope_token = set_kb_scope(kb_file_ids)
    try:
        try:
            with user_scope(user_id):
                hits = search_knowledge(
                    query, top_k=settings.kb_prefetch_top_k
                )
        except Exception:
            logger.exception("kb 预检索失败，本轮退回工具检索模式")
            return None, []
    finally:
        reset_kb_scope(scope_token)
    hits = [
        hit for hit in hits if hit.distance <= settings.kb_prefetch_max_distance
    ]
    if not hits:
        return None, []
    citations = _citations_of(hits)
    blocks = [
        f"【命中 {i + 1} · 与问题的向量距离 {hit.distance:.2f}】\n"
        f"{_format_hit_with_context(hit)}"
        for i, hit in enumerate(hits)
    ]
    content = (
        "以下是系统针对用户最新问题预检索知识库得到的原文片段。回答前先读：\n"
        "与问题相关时，以这些原文为依据作答，并自然说明出处（文件名、页码）；\n"
        "片段与问题无关时忽略它们，按你自己的知识回答，不要生搬硬套。\n\n"
        + "\n\n".join(blocks)
    )
    message: ChatMessage = {
        "role": "system",
        "content": content,
        "meta": {"kb_prefetch": True},
    }
    return message, citations


def _checkpoint_create(
    run_id: str,
    user_id: str,
    chat_session_id: str,
    messages: list[ChatMessage],
    use_web_search: bool = True,
    use_kb: bool = True,
) -> bool:
    """返回 False 表示 DB 级互斥冲突（同会话已有进行中的 run）；
    其他异常不阻断对话，只失去断点续跑能力。"""
    try:
        # 旧测试桩可能返回 None（无返回值），视为写入成功
        return (
            store.create_checkpoint(
                run_id, user_id, chat_session_id, messages, use_web_search, use_kb
            )
            is not False
        )
    except Exception:
        logger.exception("checkpoint C0 写入失败 run_id=%s", run_id)
        return True


def _checkpoint_update(run_id: str, messages: list[ChatMessage], step: int) -> None:
    try:
        store.update_checkpoint(run_id, messages, step)
    except Exception:
        logger.exception("checkpoint 更新失败 run_id=%s step=%s", run_id, step)


def _checkpoint_finish(run_id: str, status: str) -> None:
    """completed 删行（完整 turn 已落 agent_chat_history，快照无用）；
    failed/interrupted 保留，供 P2 恢复。想留审计改成 mark + 定期清理。"""
    try:
        if status == "completed":
            store.delete_checkpoint(run_id)
        else:
            store.mark_checkpoint(run_id, status)
    except Exception:
        logger.exception("checkpoint 终态写入失败 run_id=%s status=%s", run_id, status)


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


def _session_state(user_id: str, chat_session_id: str) -> tuple[list[ChatMessage], str]:
    cached = short_term_memory.load(chat_session_id, user_id=user_id)
    if cached is not None:
        return cached.messages, cached.summary
    try:
        summary = store.get_session_summary(user_id, chat_session_id)
        recent = store.load_recent_turns(
            user_id, chat_session_id, turns=short_term_memory.max_turns
        )
    except Exception:
        logger.exception("会话记忆加载失败")
        return [], ""
    return recent, summary


def _persist_turn(user_id: str, chat_session_id: str, turn: list[ChatMessage]) -> None:
    try:
        store.append_chat(user_id, chat_session_id, turn)
    except Exception:
        logger.exception("会话记忆保存失败")


def _replace_last_turn(
    user_id: str, chat_session_id: str, turn: list[ChatMessage]
) -> None:
    """重新生成成功后的落库：同一事务内删旧回答、写新回答。"""
    try:
        store.replace_chat_after_last_user(user_id, chat_session_id, turn)
    except Exception:
        logger.exception("重新生成结果保存失败")


def _capture_turn(
    user_id: str, turn: list[ChatMessage], summary: str, user_input: str
) -> None:
    try:
        capture_long_term(user_id, turn, summary, user_input)
    except Exception:
        logger.exception("长期记忆抽取失败")


def _compress_and_cache(
    user_id: str,
    chat_session_id: str,
    messages: list[ChatMessage],
    summary: str,
) -> None:
    window, new_summary = messages, summary
    try:
        window, new_summary = compress(
            messages,
            keep=short_term_memory.max_turns,
            old_summary=summary,
            summarize=summarize_session,
        )
    except Exception:
        logger.exception("短期记忆压缩失败")
    if new_summary != summary:
        try:
            store.update_session_summary(user_id, chat_session_id, new_summary)
        except Exception:
            logger.exception("会话摘要保存失败")
    short_term_memory.save(
        chat_session_id, window, user_id=user_id, summary=new_summary
    )


def run_stream(
    user_input: str,
    user_id: str = "default",
    chat_session_id: str = "default",
    *,
    finalize_background: bool = False,
    regenerate: bool = False,
    resume_run_id: str | None = None,
    use_web_search: bool = True,
    use_kb: bool = True,
    kb_file_ids: tuple[str, ...] = (),
) -> Iterator[ChatEvent]:
    if resume_run_id and regenerate:
        raise ValueError("regenerate 与 resume_run_id 互斥")
    run_id = str(uuid.uuid4())
    replacing = False
    if resume_run_id:
        # 断点续跑：上下文以 checkpoint 快照为准（热缓存里没有中间 tool 消息）。
        # 旧行已被路由层预检、在这里原子认领（DELETE...RETURNING）。
        try:
            claimed = store.claim_resumable_checkpoint(resume_run_id)
        except Exception:
            logger.exception("checkpoint 认领失败 run_id=%s", resume_run_id)
            claimed = None
        if claimed is None:
            # 被另一个请求抢走 / 状态不符，走 SSE error 让前端提示
            yield {"type": "error", "message": "对话无法恢复，请重新发送消息"}
            return
        chat_session_id = claimed["chat_session_id"]
        # 断点续跑沿用原 run 的开关状态（智能搜索/知识库）：中断时工具
        # 列表是什么样，续跑就是什么样，不受前端当前开关影响
        use_web_search = bool(claimed.get("use_web_search", True))
        use_kb = bool(claimed.get("use_kb", True))
        try:
            summary = store.get_session_summary(user_id, chat_session_id)
        except Exception:
            logger.exception("会话摘要加载失败")
            summary = ""
        messages = list(claimed["messages"])
        user_input = next(
            (
                str(m.get("content") or "")
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        if settings.resume_prefill:
            # 半截留作种子并打上消息级 partial 标记（服务端从断点续写）
            _keep_prefill_tail(messages)
        else:
            messages = _pop_incomplete_assistant(messages)
        # 落库替换的边界 = 快照里最后一条 user 消息之后：尾部可能还保留着
        # 前几轮已完成的 assistant/tool（带工具的 run 中断在后续轮），
        # 它们属于本次 run 已产出内容，替换时必须写回历史，不能丢
        persisted = len(messages)
        while persisted > 0 and messages[persisted - 1].get("role") != "user":
            persisted -= 1
        # 关键：恢复成功走"替换"落库路径——中断那轮的 finally 已把半截
        # turn 写进 agent_chat_history，成功后必须原子替换，失败则保留
        replacing = True
    else:
        recent, summary = _session_state(user_id, chat_session_id)
        if regenerate:
            # 重新生成最后一条回答：仅从内存上下文里剔除旧 assistant 消息后重问，
            # 数据库里的旧回答在生成成功前不动（失败不丢）
            while recent and recent[-1].get("role") != "user":
                recent.pop()
            if not recent:
                return
            replacing = True
            user_input = str(recent[-1].get("content") or "")
            messages = assemble(_system_prompt(user_id), summary, recent)
            persisted = len(messages)
        else:
            messages = assemble(_system_prompt(user_id), summary, recent)
            persisted = len(messages)
            messages.append({"role": "user", "content": user_input})
    if not _checkpoint_create(
        run_id, user_id, chat_session_id, messages, use_web_search, use_kb
    ):
        # DB 级互斥兜底：路由层预检之后另一条并发请求抢先开跑了
        yield {"type": "error", "message": "该会话有正在进行的回复"}
        return
    # run_id 随首个事件发给前端：流异常断开后"继续生成"按钮靠它发起 resume
    yield {"type": "run_started", "run_id": run_id}
    failed = False
    completed = False
    terminal: str | None = None
    try:
        # 回放快照里已产出、且保留在替换边界之后的内容（如工具轮前的文本）：
        # 前端用替换语义接流（首个文本事件覆盖旧半截）。不回放的话，替换
        # 会把多轮场景里更早轮次的文本从气泡里冲掉。
        for msg in messages[persisted:]:
            if msg.get("role") == "assistant" and msg.get("content"):
                yield {"type": "text", "content": str(msg["content"])}
        for event in run_loop(
            messages,
            user_id,
            run_id=run_id,
            use_web_search=use_web_search,
            use_kb=use_kb,
            kb_file_ids=kb_file_ids,
        ):
            if event.get("type") == "error":
                failed = True
            if event.get("type") == "done":
                completed = True
            yield event
    except GeneratorExit:
        # SSE 客户端断开 / 服务端关闭生成器
        terminal = "interrupted"
        raise
    finally:
        if terminal is None:
            terminal = (
                "completed" if completed else ("failed" if failed else "interrupted")
            )
        _checkpoint_finish(run_id, terminal)
        turn = messages[persisted:]
        # replacing（重新生成 / 断点恢复）只有成功完成才落库：
        # 失败或中断保留库里已有内容（旧完整回答或上一轮已写入的半截）。
        skip_persist = replacing and not completed
        if not skip_persist:
            if replacing:
                _replace_last_turn(user_id, chat_session_id, turn)
            else:
                _persist_turn(user_id, chat_session_id, turn)
            # 长期记忆只从完整回合抽取：中断/失败的半截回答进记忆库，
            # 之后会被当成"用户说过的话"反复召回，污染长期画像。
            # 摘要压缩照跑——它只是窗口管理，半截留在短期上下文里
            # 让下一轮对话保持连续（历史里也看得到半截）。
            if finalize_background:
                # 先同步完成快路径（落库 + 热缓存），保证下一轮上下文连续；
                # 慢路径（记忆抽取、摘要压缩，含多次 LLM 调用）交给后台线程。
                short_term_memory.save(
                    chat_session_id, messages, user_id=user_id, summary=summary
                )
                if completed:
                    _turn_finalizer.submit(
                        _capture_turn, user_id, turn, summary, user_input
                    )
                _turn_finalizer.submit(
                    _compress_and_cache, user_id, chat_session_id, messages, summary
                )
            else:
                if completed:
                    _capture_turn(user_id, turn, summary, user_input)
                _compress_and_cache(user_id, chat_session_id, messages, summary)


def run_loop(
    messages: list[ChatMessage],
    user_id: str = "default",
    *,
    run_id: str | None = None,
    use_web_search: bool = True,
    use_kb: bool = True,
    kb_file_ids: tuple[str, ...] = (),
    budget: Budget | None = None,
) -> Iterator[ChatEvent]:
    if budget is None:
        budget = Budget.from_settings()
    # 开关直接决定工具列表：智能搜索关闭 → 剔除 web_search；
    # 知识库关闭 → 剔除 kb_search 和 kg_search（图谱检索同属知识库能力）。
    # 模型这轮就没有对应能力，而不是靠 prompt 约定「别用」
    excluded: set[str] = set()
    if not use_web_search:
        excluded.add("web_search")
    if not use_kb:
        excluded.update({"kb_search", "kg_search"})
    tools = [s for s in tool_schemas() if s["function"]["name"] not in excluded]
    step = 0
    # 本轮的知识库引用：跨多次工具调用累积、按切片去重。
    # 它既随 SSE 下发（前端即时展示），也在收尾时挂到回答消息上落库
    turn_sources: list[dict] = []
    # 预检索注入：只在新一轮提问的入口做（最后一条是 user 且其后还没有
    # 注入块）。断点续跑中断在工具轮时，注入块已在 checkpoint 快照里，
    # 最后一条是 assistant/tool，不会重复注入。
    if (
        use_kb
        and settings.kb_prefetch_enabled
        and messages
        and messages[-1].get("role") == "user"
    ):
        prefetch_msg, prefetch_cited = _kb_prefetch(
            str(messages[-1].get("content") or ""), user_id, kb_file_ids
        )
        turn_sources = merge_sources(turn_sources, prefetch_cited)
        if prefetch_msg is not None:
            messages.append(prefetch_msg)
    for turn in range(budget.max_steps):
        result: ChatResult | None = None
        streamed = ""
        try:
            # 上下文预算：超限时裁剪的是送模型的副本，messages 本体不动
            context = _fit_context(messages, settings.llm_context_char_budget)
            for kind, payload in stream_chat(context, tools):
                if kind == "text":
                    streamed += payload
                    yield {"type": "text", "content": payload}
                else:
                    result = payload
        except LLMError as e:
            # 详情只进日志；上游异常文本可能带出 URL/响应片段，不透传给客户端
            logger.error("LLM 调用失败: %s", e)
            _stash_partial_assistant(messages, streamed, run_id, step)
            yield {"type": "error", "message": _llm_error_message(e)}
            return
        except GeneratorExit:
            _stash_partial_assistant(messages, streamed, run_id, step)
            raise
        except Exception:
            # 流已经建立、读到一半才炸（连接被掐、网关返回非 JSON 等）：
            # 不兜住的话异常会穿透 SSE 生成器，前端只看到连接断掉，
            # 半个回答留在气泡里且没有任何提示
            logger.exception("模型流式输出中断 run_id=%s", run_id)
            _stash_partial_assistant(messages, streamed, run_id, step)
            yield {"type": "error", "message": "模型输出中断，请稍后重试"}
            return

        if result is None:
            _stash_partial_assistant(messages, streamed, run_id, step)
            yield {"type": "error", "message": "模型没有返回结果"}
            return
        assistant: ChatMessage = {"role": "assistant", "content": result["content"]}
        if result["tool_calls"]:
            assistant["tool_calls"] = result["tool_calls"]
        last = messages[-1] if messages else None
        if (
            last
            and last.get("role") == "assistant"
            and not last.get("tool_calls")
        ):
            # prefill 续写：上下文尾部的半截 assistant 带 partial 标记，
            # 服务端从断点续写。模型返回后摘掉标记、去重接缝、合并进同一条
            # 消息（content + 可能的 tool_calls），历史与快照里都是一条
            # 不带 partial 的完整回答
            merged = _dedupe_seam(
                str(last.get("content") or ""), str(result["content"] or "")
            )
            last.pop("partial", None)
            if result["tool_calls"]:
                last["tool_calls"] = result["tool_calls"]
            last["content"] = str(last.get("content") or "") + merged
        else:
            messages.append(assistant)
        if run_id:
            step += 1
            _checkpoint_update(run_id, messages, step)
        if not result["tool_calls"]:
            logger.info(
                "run_loop 完成 run_id=%s steps=%d budget=%d",
                run_id,
                turn + 1,
                budget.max_steps,
            )
            # 引用挂在最终回答上：它是这一轮"给出结论"的那条消息，
            # 挂在带 tool_calls 的中间消息上，前端历史里会被跳过
            if turn_sources and messages[-1].get("role") == "assistant":
                messages[-1]["meta"] = {"sources": turn_sources}
            done_event: ChatEvent = {"type": "done"}
            if turn_sources:
                done_event["sources"] = turn_sources
            yield done_event
            return
        for call in result["tool_calls"]:
            # label 为用户可读展示名，前端「正在调用 xx」直接用它渲染
            yield {
                "type": "tool_start",
                "name": call["name"],
                "label": tool_label(call["name"]),
            }
            # 范围必须在工具执行这一层同栈设置：run_stream 的生成器会被
            # 不同线程轮流 next() 推进，外层设的 contextvar 传不进来
            scope_token = set_kb_scope(kb_file_ids)
            try:
                with collect_citations() as cited:
                    with user_scope(user_id):
                        output = execute_tool(call["name"], call["arguments"])
            finally:
                reset_kb_scope(scope_token)
            turn_sources = merge_sources(turn_sources, cited)
            end_event: ChatEvent = {"type": "tool_end", "name": call["name"]}
            if turn_sources:
                end_event["sources"] = turn_sources
            yield end_event
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": output,
                }
            )
            if run_id:
                step += 1
                _checkpoint_update(run_id, messages, step)
    logger.warning(
        "run_loop 超出预算 run_id=%s steps=%d（工具循环疑似失控）",
        run_id,
        budget.max_steps,
    )
    yield {"type": "error", "message": f"已超过最大轮数 {budget.max_steps}。"}


def run(
    user_input: str, user_id: str = "default", chat_session_id: str = "default"
) -> str:
    parts: list[str] = []
    for event in run_stream(user_input, user_id, chat_session_id):
        if event["type"] == "text":
            parts.append(event.get("content") or "")
        elif event["type"] == "error":
            return event.get("message") or "ERROR"
    return "".join(parts)
