import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from ai_agent.agents.budget import Budget
from ai_agent.agents.prompts import SYSTEM_PROMPT
from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.kb.citations import collect_citations, merge_sources
from ai_agent.kb.scope import (
    reset_kb_query_hint,
    reset_kb_scope,
    set_kb_query_hint,
    set_kb_scope,
)
from ai_agent.llm.client import stream_chat
from ai_agent.llm.types import ChatEvent, ChatMessage, ChatResult
from ai_agent.memory.facets import format_profile_block
from ai_agent.memory.short_term import (
    assemble,
    short_term_memory,
)
from ai_agent.memory.store import store, user_scope
from ai_agent.tools.registry import execute_tool, tool_label, tool_schemas

from ai_agent.agents.messages import (
    _canonical_args,
    _dedupe_seam,
    _drop_empty_assistant,
    _fit_context,
    _keep_prefill_tail,
    _output_digest,
    _pop_incomplete_assistant,
    _previous_turns,
    _stash_partial_assistant,
)
from ai_agent.agents.rewrite import (
    _drop_old_kb_catalog,
    _forced_kb_search_call,
    _has_retrieval_call,
    _kb_catalog_message,
    _rewrite_query,
    _skip_forced_kb_search,
)
from ai_agent.agents.turn import (
    _capture_turn,
    _checkpoint_create,
    _checkpoint_finish,
    _checkpoint_update,
    _compress_and_cache,
    _persist_turn,
    _replace_last_turn,
    _session_state,
)

logger = get_logger(__name__)



logger = get_logger(__name__)

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
    # 本轮新追加的 user 消息：run_loop 可能删掉历史里的旧目录，下标会前移，
    # finally 里用对象身份重定位落库起点，避免把这句问话切出 turn。
    turn_anchor: ChatMessage | None = None
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
            # 先清旧目录再记落库边界，避免下标被删目录带偏
            _drop_old_kb_catalog(messages)
            persisted = len(messages)
        else:
            messages = assemble(_system_prompt(user_id), summary, recent)
            _drop_old_kb_catalog(messages)
            persisted = len(messages)
            turn_anchor = {"role": "user", "content": user_input}
            messages.append(turn_anchor)
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
        if turn_anchor is not None:
            try:
                persisted = next(
                    i for i, msg in enumerate(messages) if msg is turn_anchor
                )
            except StopIteration:
                pass
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
    # 本轮已执行过的 (工具, 参数)：完全相同的调用结果不会变，短路掉。
    # 本地模型会陷入「再查一次试试」的循环——实测同一 kg_search 原样
    # 调了 6 次，每次都返回同样的实体列表，把预算轮数烧光后没有轮次
    # 留给最终回答，用户只看到「我即将检索」就断了
    seen_calls: set[tuple[str, str]] = set()
    # 本轮各工具的已返回结果摘要：参数不同但输出一字不差，说明这条路
    # 已经走到头了（图谱永远返回同一个中心实体、向量库永远空结果）。
    # 只在下次重复时追加一句系统提示，不阻断调用
    seen_outputs: dict[tuple[str, str], int] = {}
    # 目录 + 改写：只在新一轮提问的入口做（最后一条是 user）。断点续跑
    # 中断在工具轮时，目录已在 checkpoint 快照里，最后一条是 assistant/tool，
    # 不会重复注入。改写词留给后面工具执行时替换「原话 query」。
    original_query = ""
    rewritten_query = ""
    catalog_has_files = False
    if use_kb and messages and messages[-1].get("role") == "user":
        original_query = str(messages[-1].get("content") or "").strip()
        rewritten_query = _rewrite_query(
            original_query, _previous_turns(messages)
        )
        if settings.kb_catalog_enabled:
            catalog_msg = _kb_catalog_message(original_query, rewritten_query)
            if catalog_msg is not None:
                _drop_old_kb_catalog(messages)
                messages.append(catalog_msg)
                catalog_has_files = bool(
                    (catalog_msg.get("meta") or {}).get("kb_ready_files")
                )
    # 知识库开着且有文件时，首轮倾向检索（本地模型常直接作答）。
    # 天气/闲聊除外：强行检索会把「深圳带伞」检索成「深圳市顺丰投资」。
    force_search = bool(
        use_kb
        and settings.kb_force_search
        and catalog_has_files
        and original_query
        and not _skip_forced_kb_search(original_query)
        and any(s["function"]["name"] == "kb_search" for s in tools)
    )
    for turn in range(budget.max_steps):
        result: ChatResult | None = None
        streamed = ""
        held_text: list[str] = []
        tool_choice: str | dict = "auto"
        if force_search:
            tool_choice = {
                "type": "function",
                "function": {"name": "kb_search"},
            }
        try:
            # 上下文预算：超限时裁剪的是送模型的副本，messages 本体不动
            context = _fit_context(messages, settings.llm_context_char_budget)
            for kind, payload in stream_chat(
                context, tools, tool_choice=tool_choice
            ):
                if kind == "text":
                    streamed += payload
                    if force_search:
                        held_text.append(payload)
                    else:
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
        if force_search:
            if _has_retrieval_call(result):
                for chunk in held_text:
                    yield {"type": "text", "content": chunk}
            else:
                query = rewritten_query or original_query
                logger.info("kb 模型未检索，强制 kb_search query=%r", query[:80])
                result = {
                    "content": "",
                    "tool_calls": [_forced_kb_search_call(query)],
                }
                streamed = ""
            force_search = False
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
            text = str(result.get("content") or streamed or "").strip()
            if not text:
                # 工具已经跑完但模型没吐正文（本地小模型 / 思考链吃掉 content
                # 时会出现）。再逼一轮不带工具，避免前端只剩「已调用 xx」。
                logger.warning("最终回答为空，追加一轮强制作答 run_id=%s", run_id)
                _drop_empty_assistant(messages)
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "（系统提示）刚才没有给出文字回答。"
                            "请立即基于以上已获取的信息，用简体中文给出完整回答；"
                            "不要再调用任何工具。"
                        ),
                    }
                )
                streamed = ""
                result = None
                try:
                    context = _fit_context(
                        messages, settings.llm_context_char_budget
                    )
                    for kind, payload in stream_chat(context, []):
                        if kind == "text":
                            streamed += payload
                            yield {"type": "text", "content": payload}
                        else:
                            result = payload
                except LLMError as e:
                    logger.error("空回答收尾失败: %s", e)
                    yield {"type": "error", "message": _llm_error_message(e)}
                    return
                except GeneratorExit:
                    _stash_partial_assistant(messages, streamed, run_id, step)
                    raise
                except Exception:
                    logger.exception("空回答收尾异常 run_id=%s", run_id)
                    yield {"type": "error", "message": "模型输出中断，请稍后重试"}
                    return
                retry_text = str((result or {}).get("content") or "") or streamed
                messages.append({"role": "assistant", "content": retry_text})
                if run_id:
                    step += 1
                    _checkpoint_update(run_id, messages, step)
            logger.info(
                "run_loop 完成 run_id=%s steps=%d budget=%d content_len=%d",
                run_id,
                turn + 1,
                budget.max_steps,
                len(str((messages[-1].get("content") if messages else "") or "")),
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
            dedupe_key = (call["name"], _canonical_args(call["name"], call["arguments"]))
            if dedupe_key in seen_calls:
                # 协议要求每个 tool_call 都有对应 tool 消息，用提示文本占位；
                # 不发 tool_start/tool_end 事件——前端没开始过这次调用
                logger.info(
                    "工具同参数重复调用已短路 run_id=%s tool=%s",
                    run_id,
                    call["name"],
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "content": (
                            "（系统）你已用完全相同的参数调用过该工具，结果不会变化。"
                            "请更换检索词、改用其他工具，或基于已有信息直接作答。"
                        ),
                    }
                )
                continue
            seen_calls.add(dedupe_key)
            # label 为用户可读展示名，前端「正在调用 xx」直接用它渲染
            yield {
                "type": "tool_start",
                "name": call["name"],
                "label": tool_label(call["name"]),
            }
            # 范围必须在工具执行这一层同栈设置：run_stream 的生成器会被
            # 不同线程轮流 next() 推进，外层设的 contextvar 传不进来
            scope_token = set_kb_scope(kb_file_ids)
            hint_token = set_kb_query_hint(original_query, rewritten_query)
            try:
                with collect_citations() as cited:
                    with user_scope(user_id):
                        output = execute_tool(call["name"], call["arguments"])
            finally:
                reset_kb_query_hint(hint_token)
                reset_kb_scope(scope_token)
            turn_sources = merge_sources(turn_sources, cited)
            # 参数换了、结果却一字不差：模型在「换个说法再试一次」，
            # 再试也不会变。追加一句系统提示把它推向换词/换工具/收尾
            out_key = (call["name"], _output_digest(output))
            if seen_outputs.get(out_key):
                logger.info(
                    "工具返回结果与之前完全相同 run_id=%s tool=%s 次数=%d",
                    run_id,
                    call["name"],
                    seen_outputs[out_key] + 1,
                )
                output += (
                    "\n\n（系统）该工具已经返回过完全相同的结果，换参数也查不到"
                    "新内容。请改用更具体的检索词（例如直接搜材料名、指标名、"
                    "表格标题，而不是公司名这类大词），或换用其他工具，"
                    "或基于已有信息直接作答。"
                )
            seen_outputs[out_key] = seen_outputs.get(out_key, 0) + 1
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
        "run_loop 超出预算 run_id=%s steps=%d（工具循环疑似失控），强制收尾",
        run_id,
        budget.max_steps,
    )
    # 预算耗尽不直接报错：前面几轮检索已经花了成本，用户手里只有模型
    # 说「我即将检索」的半截话。剥掉工具再问最后一次，强制它基于已
    # 拿到的结果给出回答——收尾消息用 system（前端不渲染、不污染气泡）。
    # 这次也失败才真正报错。
    messages.append(
        {
            "role": "system",
            "content": (
                "（系统提示）检索轮次已用完，不能再调用任何工具。"
                "请立即基于以上已获取的信息给出最终回答；"
                "确实缺少关键数据时，明确说明缺什么即可。"
            ),
        }
    )
    streamed = ""

    def _stash_final(content: str) -> None:
        # 收尾场景 messages 尾部是 tool 消息，_stash_partial_assistant
        # 只认「最后一条是 assistant」的 prefill 场景，这里手动落
        if content:
            messages.append({"role": "assistant", "content": content})

    try:
        context = _fit_context(messages, settings.llm_context_char_budget)
        result: ChatResult | None = None
        for kind, payload in stream_chat(context, []):
            if kind == "text":
                streamed += payload
                yield {"type": "text", "content": payload}
            else:
                result = payload
        final_content = str((result or {}).get("content") or "") or streamed
        _stash_final(final_content)
        if run_id:
            step += 1
            _checkpoint_update(run_id, messages, step)
        if turn_sources and messages[-1].get("role") == "assistant":
            messages[-1]["meta"] = {"sources": turn_sources}
        done_event: ChatEvent = {"type": "done"}
        if turn_sources:
            done_event["sources"] = turn_sources
        yield done_event
    except GeneratorExit:
        _stash_final(streamed)
        raise
    except LLMError as e:
        logger.error("预算收尾回答失败: %s", e)
        _stash_final(streamed)
        yield {"type": "error", "message": _llm_error_message(e)}
    except Exception:
        logger.exception("预算收尾回答异常 run_id=%s", run_id)
        _stash_final(streamed)
        yield {"type": "error", "message": "模型输出中断，请稍后重试"}


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
