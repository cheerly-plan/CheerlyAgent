import queue
import re
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from ai_agent.agents.budget import Budget
from ai_agent.agents.messages import (
    _canonical_args,
    _dedupe_seam,
    _drop_empty_assistant,
    _fit_context,
    _keep_prefill_tail,
    _output_digest,
    _pop_incomplete_assistant,
    _stash_partial_assistant,
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

from ai_agent.prompts.chat import (
    RUNTIME_BUDGET_EXHAUSTED,
    RUNTIME_DUP_CALL,
    RUNTIME_DUP_OUTPUT,
    RUNTIME_EMPTY_ANSWER,
    build_system_prompt,
)
# 报告类型清单是业务数据，由编排层注入系统提示词（prompts 层不反向依赖
# report 域）。skills 注册表是轻叶子（只读 SKILL.md），模块级 import 无碍。
from ai_agent.report.skills import skill_catalog
from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.core.text import clip_text
from ai_agent.kb.retrieval.citations import collect_citations, merge_sources
from ai_agent.kb.ingest.doc_profiles import resolve_profile
from ai_agent.kb.retrieval.scope import (
    extract_prev_turn,
    reset_kb_question,
    reset_kb_scope,
    reset_rewrite_context,
    set_kb_question,
    set_kb_scope,
    set_rewrite_context,
)
from ai_agent.llm.client import stream_chat
from ai_agent.llm.tokenizer import count_json_tokens
from ai_agent.llm.types import ChatEvent, ChatMessage, ChatResult
from ai_agent.memory.facets import format_profile_block
from ai_agent.memory.progressive import (
    build_memory_index,
    format_recall_block,
    full_trace,
    rerank_items,
)
from ai_agent.memory.short_term import (
    assemble,
    short_term_memory,
)
from ai_agent.storage.store import store, user_scope
from ai_agent.tools.progress import (
    reset_event_emitter,
    reset_progress,
    set_event_emitter,
    set_progress,
)
from ai_agent.tools.context import (
    session_scope,
    tool_result_index,
    tool_results_scope,
)
from ai_agent.tools.registry import execute_tool, tool_schemas
from ai_agent.tools.result_pool import tool_result_pool

# 会长时间阻塞的工具：丢到工作线程，边跑边推 tool_progress。generate_report
# 现在只提交后台任务（秒回），留在集合里是为了统一 user_scope / 引用收集的调用路径。
_PROGRESS_TOOLS = frozenset({"generate_report"})
# 知识库能力清单按 doc_type 分组，库里类型多了会线性撑大 system；只列文档数
# 最多的几类，其余让模型用 kb_list_files 现查。整体还有 system_dynamic_max_chars 兜底。
_KB_CAP_MAX_DOC_TYPES = 8
_DYNAMIC_CLIP_MARK = "\n[系统动态信息已省略 {n} 字符，完整清单请用 kb_list_files 查询]\n"
_tool_progress_pool = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="tool-progress"
)

logger = get_logger(__name__)

# 问句点名了库内主体或图谱实体时，第一步只留这两个工具并强制调用。
# 天气、记忆、联网留在后面的轮次，避免模型用它们应付「必须调一个工具」。
_KB_RETRIEVAL_TOOLS = frozenset({"kb_search", "kg_search", "kb_read"})

# 「生成一份研报」走独立流水线：工具自己会检索、抽取、计算、校验。
# 这类请求不必先被强制查一轮 kb_search，否则既多烧一次调用，又把原文片段
# 塞进上下文、诱导模型自己撰写正文而不去调 generate_report。
# 必须同时有「生成类动词」和「研报类名词」，避免误伤「研报里说了什么」这种检索题。
_REPORT_VERB = re.compile(r"生成|出一份|做一份|写一份|制作|导出|来一份|给我一份")
_REPORT_NOUN = re.compile(r"研报|财报深度分析|分析报告|PDF", re.IGNORECASE)

# 流结束后的收尾工作（长期记忆抽取/摘要压缩含多次 LLM 调用）放到后台线程，
# 不再阻塞 SSE 尾部延迟；进程退出时由 lifespan 调 shutdown_turn_finalizer 等待。
_turn_finalizer = ThreadPoolExecutor(max_workers=4, thread_name_prefix="turn-finalize")


def shutdown_turn_finalizer(wait: bool = True) -> None:
    """关闭 turn-finalize / 工具进度线程池。

    wait=True 会等在途记忆抽取跑完（生产停机）；热重载必须 wait=False，
    且 lifespan 还需 os._exit 跳过 ThreadPoolExecutor 的 atexit join，
    否则长工具仍会钉死旧 worker。
    """
    _turn_finalizer.shutdown(wait=wait, cancel_futures=not wait)
    _tool_progress_pool.shutdown(wait=wait, cancel_futures=not wait)


def _run_tool_with_progress(
    name: str,
    arguments: dict,
    *,
    user_id: str,
    kb_file_ids: list[str] | None,
    messages: list[ChatMessage],
    chat_session_id: str = "",
) -> Iterator[dict]:
    """在工作线程跑工具，主循环把进度转成 tool_progress；末尾 yield _tool_result。"""
    # 队列里可能是进度文案（str），也可能是工具抛出的结构化事件（dict）：
    # 后者原样下发给前端（如研报后台任务的 job_id），不塞进进度文案。
    progress_q: queue.Queue[str | dict | None] = queue.Queue()
    prev_turn = extract_prev_turn(messages, settings.kb_rewrite_history_turns)
    question = _latest_user_text(messages)

    def _emit(message: str) -> None:
        progress_q.put(message)

    def _emit_event(event: dict) -> None:
        progress_q.put(dict(event))

    def _worker() -> tuple[str, list[dict]]:
        scope_token = set_kb_scope(kb_file_ids)
        rewrite_token = set_rewrite_context(prev_turn)
        question_token = set_kb_question(question)
        progress_token = set_progress(_emit)
        event_token = set_event_emitter(_emit_event)
        try:
            with tool_results_scope(tool_result_index(messages)):
                with session_scope(chat_session_id):
                    with collect_citations() as cited:
                        with user_scope(user_id):
                            output = execute_tool(name, arguments)
            return output, list(cited)
        finally:
            reset_event_emitter(event_token)
            reset_progress(progress_token)
            reset_kb_scope(scope_token)
            reset_rewrite_context(rewrite_token)
            reset_kb_question(question_token)
            progress_q.put(None)

    future = _tool_progress_pool.submit(_worker)
    while True:
        try:
            item = progress_q.get(timeout=0.4)
        except queue.Empty:
            if future.done():
                break
            continue
        if item is None:
            break
        if isinstance(item, dict):
            yield item  # 结构化事件：type 由工具给出，缺 type 的在 emit_event 已挡掉
            continue
        yield {"type": "tool_progress", "name": name, "message": item}
    while True:
        try:
            item = progress_q.get_nowait()
        except queue.Empty:
            break
        if item is None:
            continue
        if isinstance(item, dict):
            yield item
            continue
        yield {"type": "tool_progress", "name": name, "message": item}
    output, cited = future.result()
    yield {"type": "_tool_result", "content": output, "sources": cited}


def _attach_sources(
    messages: list[ChatMessage], turn_sources: list[dict], report_job_id: str = ""
) -> None:
    """引用与研报任务 id 挂在最终回答上。

    挂在带 tool_calls 的中间消息上，历史回放会跳过。``report_job_id`` 随消息
    落库，刷新页面后前端还能凭它继续轮询任务、把下载链接长回来。
    """
    if not messages or messages[-1].get("role") != "assistant":
        return
    meta = dict(messages[-1].get("meta") or {})
    if turn_sources:
        meta["sources"] = turn_sources
    if report_job_id:
        meta["report_job_id"] = report_job_id
    if meta:
        messages[-1]["meta"] = meta


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


def _profile_prompt(user_id: str, *, user_input: str = "") -> str:
    try:
        items = store.list_profile(user_id)
        logger.info("list_profile user_id=%s count=%s", user_id, len(items))
    except Exception:
        logger.exception("用户画像读取失败")
        return ""

    # 渐进式记忆恒定启用：注入「索引 + 与当前问句相关的条目」，被省略的
    # 画像由 expand_memory 按需展开。任何异常 fail-open 回退到全量画像。
    try:
        block, trace = build_memory_index(
            items,
            user_input=user_input,
            top_k=settings.memory_index_top_k,
            max_chars=settings.memory_index_max_chars,
            expand_hint=True,
        )
    except Exception:
        logger.exception("渐进式记忆块构建失败，回退全量画像")
        block = format_profile_block(items)
        trace = full_trace(items, block, reason="fallback_error", mode="fallback")
    if settings.memory_trace_enabled:
        trace.emit(logger)
    return block


def _recall_block(user_id: str, user_input: str) -> str:
    """第 3 步：按当前问句自动召回情景记忆摘要，不依赖模型主动调工具。

    没有问句时不召回（空查询的向量结果没有意义）。召回会多一次本地
    embedding 与向量查询，任何异常都 fail-open 跳过，不让记忆链路拖垮对话。
    """
    if not user_input.strip():
        return ""
    try:
        # 向量先取宽 3 倍，再按「相关性 + 新鲜度 + 重要性」收窄到 top_k：
        # 纯向量排序只看语义相似，刚发生、被反复用到的近况会被淹没。
        hits = store.search_episode(
            user_input,
            k=max(settings.memory_auto_recall_top_k * 3, 3),
            user_id=user_id,
        )
        hits = rerank_items(
            hits, user_input, top_k=settings.memory_auto_recall_top_k
        )
        block = format_recall_block(
            hits, max_chars=settings.memory_auto_recall_max_chars
        )
        logger.info("memory_recall hits=%d chars=%d", len(hits), len(block))
        return block
    except Exception:
        logger.exception("情景记忆自动召回失败，本轮跳过")
        return ""


def _kb_capability_block(kb_file_ids: tuple[str, ...] = ()) -> str:
    """拼接知识库能力声明，注入系统提示词。

    静态提示词只讲「怎么选工具」（关系→kg_search、数字→kb_search），具体
    图谱覆盖哪些实体/关系由本块按库内实际文档类型动态生成——加公告、年报
    等新类型时无需改提示词，只要 kb/skills 或 doc_profiles 覆盖里配好即可，提示词不再只针对季报。
    这样模型按问题选对应类型的关系，而不是被硬编码的季报语义带偏。
    """
    try:
        rows = store.list_kb_files()
    except Exception:
        logger.exception("知识库文件清单读取失败")
        return ""
    ready = [r for r in rows if r.get("status") == "ready"]
    if kb_file_ids:
        want = {str(f) for f in kb_file_ids}
        ready = [r for r in ready if str(r.get("file_id")) in want]
    if not ready:
        return ""

    # 按 doc_type 分组；空 doc_type 回落到 general（恰好能暴露漏标类型的问题）
    groups: dict[str, list[dict]] = {}
    for r in ready:
        dt = str(r.get("doc_type") or "general").strip() or "general"
        groups.setdefault(dt, []).append(r)

    lines = [
        "【知识库能力】知识库图谱存的是实体之间的关系（「谁和谁有什么关系」）。"
        "各文档类型在图谱中覆盖的实体与关系如下，不在其中的关系请走 kb_search 读原文：",
    ]
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if len(ranked) > _KB_CAP_MAX_DOC_TYPES:
        lines.append(
            f"（文档类型共 {len(ranked)} 类，此处只列文档数最多的 "
            f"{_KB_CAP_MAX_DOC_TYPES} 类；完整清单用 kb_list_files 查）"
        )
        ranked = ranked[:_KB_CAP_MAX_DOC_TYPES]
    for dt, members in ranked:
        profile = resolve_profile(dt)
        scope = "、".join(profile.entity_scope) if profile.entity_scope else "（通用抽取，无限定范围）"
        preds = "、".join(profile.predicates) if profile.predicates else "（通用抽取，无限定谓词）"
        count = len(members)
        lines.append(f"- {profile.label}（{count} 份）")
        lines.append(f"  实体范围={scope}")
        lines.append(f"  关系={preds}")
    lines.append(
        "注：具体数字、报表科目、原文表述不在图谱里，请用 kb_search 查；"
        "关系题同时需要某个科目金额时，kg_search 与 kb_search 两个都调。"
    )
    period_line = _period_inventory(ready)
    if period_line:
        lines.append(period_line)
    return "\n".join(lines)


def _period_inventory(ready: list[dict]) -> str:
    """库里实际有哪些报告期。模型容易按训练截止日判断某年财报还没披露。"""
    from ai_agent.kb.ingest.meta import period_rank

    counts: dict[str, int] = {}
    for row in ready:
        period = str(row.get("period") or "").strip()
        if period:
            counts[period] = counts.get(period, 0) + 1
    if not counts:
        return ""
    ordered = sorted(counts, key=period_rank, reverse=True)
    parts = [f"{period}（{counts[period]} 份）" for period in ordered]
    return (
        "库内已有报告期：" + "、".join(parts) + "。"
        "这些是已经入库的文件。某年有没有资料，以这份清单、kb_list_files 和检索结果抬头为准，"
        "不要按训练知识里的披露进度推断。"
        "表格中更早的年份是对比列，不是文件的报告期。"
    )


def _system_prompt(
    user_id: str,
    *,
    use_kb: bool = True,
    use_memory: bool = True,
    kb_file_ids: tuple[str, ...] = (),
    user_input: str = "",
) -> str:
    # 提示词里的检索纪律/引用规范按本轮是否挂了知识库同步裁剪：工具列表
    # 已经在下面按开关剔掉了 kb_search/kg_search，提示词再讲一遍「必须先
    # 调用 kb_search」只会让模型去调一个不存在的工具。
    core = build_system_prompt(
        use_kb=use_kb,
        use_memory=use_memory,
        report_catalog=skill_catalog(),
    )
    # 只注入文档类型能覆盖的实体和关系。文件清单不进提示词，模型用
    # kb_list_files 现查；检索范围由问句里的主体和文档路由决定。
    # 无问句时保持原来的调用形式：存量测试会把 _profile_prompt 换成只收
    # 位置参数的桩，多传一个关键字参数会把它们打挂；顺带让这条路径与
    # 改动前逐字一致。只有真的拿到问句才把相关性查询传下去。
    block = (
        _profile_prompt(user_id, user_input=user_input)
        if use_memory and user_input
        else (_profile_prompt(user_id) if use_memory else "")
    )
    cap_block = _kb_capability_block(kb_file_ids) if use_kb else ""
    recall = _recall_block(user_id, user_input) if use_memory else ""
    tail = "\n\n".join(p for p in (_today_line(), block, recall, cap_block) if p)
    if not tail:
        return core
    # 动态块原先完全不受上下文预算约束：画像条数、库内文档类型数、报告类型
    # 数都会让它线性增长。这里按行保头保尾裁到硬上限，并留一句显式提示，
    # 让模型知道清单不完整、需要时自己查，而不是拿着半截清单硬答。
    tail = clip_text(tail, settings.system_dynamic_max_chars, marker=_DYNAMIC_CLIP_MARK)
    return f"{core}\n\n{tail}"


def _today_line() -> str:
    from datetime import date

    return (
        f"【当前日期】今天是 {date.today().isoformat()}。"
        "此日之前的报告期都是已披露文件，不要当成尚未发布的未来数据。"
    )


def _fitted_context(
    messages: list[ChatMessage], tools: list[dict] | None = None
) -> list[ChatMessage]:
    """按「硬上限 - 输出预留 - 工具 schema」和软水位裁剪送模型副本。"""
    usable, soft = settings.context_token_limits()
    if tools:
        tool_tokens = count_json_tokens(tools)
        usable = max(1, usable - tool_tokens)
        soft = max(0, soft - tool_tokens)
    return _fit_context(messages, usable, soft_limit=soft)


def _remember_tool_result(
    *,
    user_id: str,
    chat_session_id: str,
    call_id: str,
    name: str,
    arguments: dict,
    output: object,
    cited: list,
) -> dict:
    """把裁剪前全文放进池子，并生成可持久化的 tool 消息 meta。"""
    raw = str(getattr(output, "raw", output))
    meta = dict(getattr(output, "meta", {}) or {})
    meta.setdefault("tool", name)
    meta.setdefault("arguments", arguments)
    meta.setdefault("raw_chars", len(raw))
    if "preview" not in meta:
        lines = raw.splitlines()
        meta["preview"] = (lines[0] if lines else "").strip()[:120]
    if cited:
        meta["citations"] = [
            {
                "filename": item.filename,
                "page": item.page,
                "page_end": item.page_end,
                "kind": item.kind,
                "section": item.section,
            }
            for item in cited[:5]
        ]
    tool_result_pool.put(
        user_id,
        chat_session_id,
        call_id,
        name,
        raw,
        meta,
    )
    # 持久化裁剪前原文：池子淘汰/换 worker/重启后 read_tool_result 仍能读到。
    # 写失败不影响工具结果返回，池子仍可服务当前进程。
    try:
        store.save_raw_tool_result(
            user_id,
            chat_session_id,
            call_id,
            name,
            raw,
            meta,
        )
    except Exception:
        logger.warning("工具原文持久化失败 call_id=%s", call_id, exc_info=True)
    return meta


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
    memory_enabled: bool = True,
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
            summary_until_seq = 0
        except Exception:
            logger.exception("会话摘要加载失败")
            summary = ""
            summary_until_seq = 0
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
        recent, summary, summary_until_seq = _session_state(
            user_id, chat_session_id
        )
        if regenerate:
            # 重新生成最后一条回答：仅从内存上下文里剔除旧 assistant 消息后重问，
            # 数据库里的旧回答在生成成功前不动（失败不丢）
            while recent and recent[-1].get("role") != "user":
                recent.pop()
            if not recent:
                return
            replacing = True
            user_input = str(recent[-1].get("content") or "")
            messages = assemble(
                _system_prompt(
                    user_id,
                    use_kb=use_kb,
                    use_memory=memory_enabled,
                    kb_file_ids=kb_file_ids,
                    user_input=user_input,
                ),
                summary,
                recent
            )
            persisted = len(messages)
        else:
            messages = assemble(
                _system_prompt(
                    user_id,
                    use_kb=use_kb,
                    use_memory=memory_enabled,
                    kb_file_ids=kb_file_ids,
                    user_input=user_input,
                ),
                summary,
                recent
            )
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
            chat_session_id=chat_session_id,
            use_web_search=use_web_search,
            use_kb=use_kb,
            kb_file_ids=kb_file_ids,
            use_memory=memory_enabled,
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
                # 先存全量窗口，避免后台摘要还没跑完时下一轮只看到被裁掉
                # 但尚未进摘要的轮次；_compress_and_cache 完成后再存压缩窗口。
                short_term_memory.save(
                    chat_session_id,
                    messages,
                    user_id=user_id,
                    summary=summary,
                    summary_until_seq=summary_until_seq,
                    trim=False,
                )
                if completed:
                    _turn_finalizer.submit(
                        _capture_turn,
                        user_id,
                        turn,
                        summary,
                        user_input,
                        chat_session_id,
                        memory_enabled=memory_enabled,
                    )
                _turn_finalizer.submit(
                    _compress_and_cache,
                    user_id,
                    chat_session_id,
                    messages,
                    summary,
                    summary_until_seq,
                )
            else:
                if completed:
                    _capture_turn(
                        user_id,
                        turn,
                        summary,
                        user_input,
                        chat_session_id,
                        memory_enabled=memory_enabled,
                    )
                _compress_and_cache(
                    user_id,
                    chat_session_id,
                    messages,
                    summary,
                    summary_until_seq,
                )


def _latest_user_text(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def _question_already_retrieved(messages: list[ChatMessage]) -> bool:
    """当前这句问话之后已经有工具结果，就不再强制检索。"""
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "tool":
            return True
        if role == "user":
            return False
    return False


def _followup_subjects(
    messages: list[ChatMessage],
    file_ids: list[uuid.UUID] | None,
) -> set[str]:
    """当前句省略主体时，回看此前几轮用户问话点过的库内主体。

    追问常把主体留在上一轮（「那毛利率呢」「去年呢」）。强制检索门只看
    当前句字面会把这类问句放过去，模型不查库、拿上一轮的检索片段或训练
    记忆直接作答。回看窗口与查询改写共用 kb_rewrite_history_turns，命中
    任意一轮即认定这句仍是知识库追问。读库失败 fail-open 返回空（不强制，
    与字面检测的放行语义一致）。只回看主体；人名/子公司等实体追问更少见，
    不在这次兜底范围里。
    """
    from ai_agent.kb.retrieval import mentioned_subjects

    cur: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            cur = i
            break
    if cur is None:
        return set()
    turns = 0
    for i in range(cur - 1, -1, -1):
        if messages[i].get("role") != "user":
            continue
        text = str(messages[i].get("content") or "").strip()
        if not text:
            continue
        turns += 1
        if turns > settings.kb_rewrite_history_turns:
            break
        try:
            hits = mentioned_subjects(text, file_ids=file_ids)
        except Exception:
            logger.warning("追问主体回看检测失败，按不强制处理", exc_info=True)
            return set()
        if hits:
            logger.info(
                "追问未点名主体，回看此前问句命中 subjects=%s",
                "、".join(sorted(hits)),
            )
            return hits
    return set()


_KG_PREFETCH_MARK = "【图谱邻域】"


def _kg_prefetch_present(messages: list[ChatMessage]) -> bool:
    """这句问话之后已经注入过邻域，续跑时不要再查一次。"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return False
        if msg.get("role") == "system" and str(msg.get("content") or "").startswith(
            _KG_PREFETCH_MARK
        ):
            return True
    return False


def _attach_kg_prefetch(
    messages: list[ChatMessage], kb_file_ids: tuple[str, ...]
) -> None:
    """关系问句在选工具前放入图谱邻域。科目金额问句不放。"""
    if _kg_prefetch_present(messages):
        return
    text = _latest_user_text(messages).strip()
    if not text:
        return
    from ai_agent.tools.kb.kg_search import prefetch_neighborhood

    scope_token = set_kb_scope(kb_file_ids)
    try:
        block = prefetch_neighborhood(text)
    finally:
        reset_kb_scope(scope_token)
    if not block:
        return
    logger.info("图谱预检索已注入 chars=%d", len(block))
    messages.append({"role": "system", "content": block})


def _must_call_kb_tool(
    messages: list[ChatMessage],
    tools: list[dict],
    kb_file_ids: tuple[str, ...] = (),
) -> bool:
    """问句点名了文件主体或图谱实体时，这一步必须调用知识库工具。

    主体（公司名、简称）和图谱实体（人名、子公司、产品）都会触发。
    只强制「必须查」，选 kb_search 还是 kg_search 仍由模型按【知识库能力】决定。
    当前句字面点名主体/实体就强制；追问省略了主体时，回看前几轮用户问句
    是否点过主体（见 _followup_subjects，可用 kb_followup_force_enabled 关闭）。
    这句之后已经有工具结果（续跑、检索完准备作答）不再强制。读库失败则放行。
    """
    names = {t.get("function", {}).get("name") for t in tools}
    if not (_KB_RETRIEVAL_TOOLS & names):
        return False
    if _question_already_retrieved(messages):
        return False
    text = _latest_user_text(messages).strip()
    if not text:
        return False
    if _REPORT_VERB.search(text) and _REPORT_NOUN.search(text):
        return False
    from ai_agent.kb.retrieval import mentioned_entities, mentioned_subjects

    file_ids: list[uuid.UUID] | None = None
    if kb_file_ids:
        parsed: list[uuid.UUID] = []
        for raw in kb_file_ids:
            try:
                parsed.append(uuid.UUID(str(raw)))
            except ValueError:
                continue
        file_ids = parsed or None
    try:
        subjects = mentioned_subjects(text, file_ids=file_ids)
    except Exception:
        logger.exception("主体提及检测失败")
        subjects = set()
    # 主体命中已经足够强制检索，但实体仍要查：问的是公司 A 里的实体 B 时，
    # 日志里得看得到 B，不能因为 A 命中就把实体检测跳过。
    entities = mentioned_entities(text, file_ids=file_ids)
    if not subjects and not entities and settings.kb_followup_force_enabled:
        # 追问兜底：这句没点名主体，但前几轮问句点过库内主体——主体多半被
        # 省略或写成了指代。字面检测会放行，模型便不查库直接作答；这里把
        # 这句仍按知识库追问强制检索（具体主体回看逻辑见 _followup_subjects）。
        subjects = _followup_subjects(messages, file_ids)
    if subjects or entities:
        logger.info(
            "问句命中知识库，强制调用工具 subjects=%s entities=%s",
            "、".join(sorted(subjects)),
            "、".join(entities),
        )
    return bool(subjects or entities)


def run_loop(
    messages: list[ChatMessage],
    user_id: str = "default",
    *,
    run_id: str | None = None,
    chat_session_id: str = "",
    use_web_search: bool = True,
    use_kb: bool = True,
    kb_file_ids: tuple[str, ...] = (),
    use_memory: bool = True,
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
        # 研报积木与入口都依赖知识库表格抽取，知识库关掉时一起下线。
        # report_compare / report_validate 是纯计算与校验，不触知识库，保留。
        excluded.update({"kb_search", "kg_search", "generate_report", "report_extract_indicators"})
    if not use_memory:
        excluded.update({"search_memory", "write_memory", "expand_memory"})
    tools = [s for s in tool_schemas() if s["function"]["name"] not in excluded]
    if use_kb:
        _attach_kg_prefetch(messages, kb_file_ids)
    step = 0
    # 本轮的知识库引用：跨多次工具调用累积、按切片去重。
    # 它既随 SSE 下发（前端即时展示），也在收尾时挂到回答消息上落库
    turn_sources: list[dict] = []
    # 本轮提交的研报后台任务：挂在回答消息的 meta 上，刷新后卡片能续上
    report_job_id = ""
    # 本轮已执行过的 (工具, 参数)：完全相同的调用结果不会变，短路掉。
    # 本地模型会陷入「再查一次试试」的循环——实测同一 kg_search 原样
    # 调了 6 次，每次都返回同样的实体列表，把预算轮数烧光后没有轮次
    # 留给最终回答，用户只看到「我即将检索」就断了
    seen_calls: set[tuple[str, str]] = set()
    # 本轮各工具的已返回结果摘要：参数不同但输出一字不差，说明这条路
    # 已经走到头了（图谱永远返回同一个中心实体、向量库永远空结果）。
    # 只在下次重复时追加一句系统提示，不阻断调用
    seen_outputs: dict[tuple[str, str], int] = {}
    for turn in range(budget.max_steps):
        result: ChatResult | None = None
        streamed = ""
        tool_choice: str | dict = "auto"
        step_tools = tools
        # 点名了库内主体、而这句还没查过：这一步必须调 kb_search 或 kg_search。
        # 工具返回后的下一步不再强制，模型可以继续查，也可以写答案。
        if _must_call_kb_tool(messages, tools, kb_file_ids):
            tool_choice = "required"
            step_tools = [
                s for s in tools if s["function"]["name"] in _KB_RETRIEVAL_TOOLS
            ]
        try:
            # 上下文预算：超限时裁剪的是送模型的副本，messages 本体不动
            context = _fitted_context(messages, step_tools)
            for kind, payload in stream_chat(
                context, step_tools, tool_choice=tool_choice
            ):
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
            text = str(result.get("content") or streamed or "").strip()
            if not text:
                # 工具已经跑完但模型没吐正文（本地小模型 / 思考链吃掉 content
                # 时会出现）。再逼一轮不带工具，避免气泡里只有光标。
                logger.warning("最终回答为空，追加一轮强制作答 run_id=%s", run_id)
                _drop_empty_assistant(messages)
                messages.append(
                    {"role": "system", "content": RUNTIME_EMPTY_ANSWER}
                )
                streamed = ""
                result = None
                try:
                    context = _fitted_context(messages, step_tools)
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
            _attach_sources(messages, turn_sources, report_job_id)
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
                        "content": RUNTIME_DUP_CALL,
                    }
                )
                continue
            seen_calls.add(dedupe_key)
            # 前端只用来清掉调用前的半截字，展示名不再下发
            yield {"type": "tool_start", "name": call["name"]}
            # 范围与查询改写上下文必须在工具执行这一层同栈设置：run_stream 的
            # 生成器会被不同线程轮流 next() 推进，外层设的 contextvar 传不进来
            if call["name"] in _PROGRESS_TOOLS:
                output = ""
                cited = []
                for event in _run_tool_with_progress(
                    call["name"],
                    call["arguments"],
                    user_id=user_id,
                    kb_file_ids=kb_file_ids,
                    messages=messages,
                    chat_session_id=chat_session_id,
                ):
                    if event.get("type") == "_tool_result":
                        output = event.get("content") or ""
                        cited = list(event.get("sources") or [])
                        continue
                    if event.get("type") == "report_job":
                        report_job_id = str(event.get("job_id") or "")
                    yield event  # type: ignore[misc]
            else:
                scope_token = set_kb_scope(kb_file_ids)
                rewrite_token = set_rewrite_context(
                    extract_prev_turn(messages, settings.kb_rewrite_history_turns)
                )
                question_token = set_kb_question(_latest_user_text(messages))
                try:
                    with tool_results_scope(tool_result_index(messages)):
                        with session_scope(chat_session_id):
                            with collect_citations() as cited:
                                with user_scope(user_id):
                                    output = execute_tool(
                                        call["name"], call["arguments"]
                                    )
                finally:
                    reset_kb_scope(scope_token)
                    reset_rewrite_context(rewrite_token)
                    reset_kb_question(question_token)
            turn_sources = merge_sources(turn_sources, cited)
            tool_meta = _remember_tool_result(
                user_id=user_id,
                chat_session_id=chat_session_id,
                call_id=str(call.get("id") or ""),
                name=call["name"],
                arguments=call["arguments"],
                output=output,
                cited=cited,
            )
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
                output += "\n\n" + RUNTIME_DUP_OUTPUT
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
                    "meta": tool_meta,
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
    messages.append({"role": "system", "content": RUNTIME_BUDGET_EXHAUSTED})
    streamed = ""

    def _stash_final(content: str) -> None:
        # 收尾场景 messages 尾部是 tool 消息，_stash_partial_assistant
        # 只认「最后一条是 assistant」的 prefill 场景，这里手动落
        if content:
            messages.append({"role": "assistant", "content": content})

    try:
        context = _fitted_context(messages, step_tools)
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
        _attach_sources(messages, turn_sources, report_job_id)
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
