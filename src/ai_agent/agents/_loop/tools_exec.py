"""工具执行与结果回收：带进度的调用、结果留存、来源挂载。

由原 agents/loop.py 拆分而来，逻辑未改动。
"""

from __future__ import annotations

from collections.abc import Iterator
import queue
from ai_agent.core.config import settings
import time
from ai_agent.tools.result_pool import tool_result_pool
from ai_agent.kb.retrieval.citations import collect_citations
from ai_agent.kb.retrieval.scope import extract_prev_turn, reset_kb_question, reset_kb_scope, reset_rewrite_context, set_kb_question, set_kb_scope, set_rewrite_context
from ai_agent.llm.types import ChatMessage
from ai_agent.prompts.chat import RUNTIME_TOOL_TIMEOUT
from ai_agent.storage.store import store, user_scope
from ai_agent.tools.context import session_scope, tool_result_index, tool_results_scope
from ai_agent.tools.progress import reset_event_emitter, reset_progress, set_event_emitter, set_progress

from ai_agent.core.logging import get_logger

from ai_agent.agents._loop.state import (
    _tool_progress_pool,
)

logger = get_logger(__name__)


# 测试 monkeypatch 打在 ``ai_agent.agents.loop`` 上；内部调用一律经
# _public() 取当前属性，否则补丁会被本地绑定绕过（静默失效）。
def _public():
    import ai_agent.agents.loop as public

    return public


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
    question = _public()._latest_user_text(messages)

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
                            # 进度型长任务：本次不设总时长，交给下面的看门狗
                            # 按「多久没进度」判活，别被 60 秒短预算打断
                            output = _public().execute_tool(name, arguments, timeout=0)
            return output, list(cited)
        finally:
            reset_event_emitter(event_token)
            reset_progress(progress_token)
            reset_kb_scope(scope_token)
            reset_rewrite_context(rewrite_token)
            reset_kb_question(question_token)
            progress_q.put(None)

    future = _tool_progress_pool.submit(_worker)
    # 长任务看门狗：有进度就算活着，长时间一句进度都没有才判超时。
    # execute_tool 内部已有总时长兜底，这里补的是「工具迟迟不返回时
    # 前端有个明确交代」，不让 SSE 一直空转。
    progress_budget = max(
        0.0, float(settings.tool_execute_progress_timeout_seconds)
    )
    last_progress = time.monotonic()
    while True:
        try:
            item = progress_q.get(timeout=0.4)
        except queue.Empty:
            if future.done():
                break
            if progress_budget > 0 and time.monotonic() - last_progress > progress_budget:
                _public().logger.warning(
                    "工具长时间无进度 name=%s budget=%.0fs，按超时收尾",
                    name,
                    progress_budget,
                )
                # 线程收不回来（挂死的调用），只放弃等待；daemon 线程不阻塞退出
                yield {
                    "type": "_tool_result",
                    "content": RUNTIME_TOOL_TIMEOUT.format(seconds=progress_budget),
                    "sources": [],
                }
                return
            continue
        last_progress = time.monotonic()
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
        _public().logger.warning("工具原文持久化失败 call_id=%s", call_id, exc_info=True)
    return meta

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

# 会长时间阻塞的工具：丢到工作线程，边跑边推 tool_progress。generate_report
# 现在只提交后台任务（秒回），留在集合里是为了统一 user_scope / 引用收集的调用路径。
_PROGRESS_TOOLS = frozenset({"generate_report"})
