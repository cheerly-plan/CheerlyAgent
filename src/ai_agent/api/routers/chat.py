import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Iterator

from ai_agent.agents.loop import run_stream
from ai_agent.api.dependencies import get_current_user
from ai_agent.api.schemas import (
    ChatMemoryMessage,
    ChatMemoryResponse,
    ChatRequest,
    ResumeRequest,
    SafeId,
)
from ai_agent.core.config import settings
from ai_agent.core.exceptions import AgentError
from ai_agent.core.logging import get_logger
from ai_agent.storage.db import UserRecord
from ai_agent.storage.store import store
from ai_agent.tasks.title import start_title_job
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/v1", tags=["chat"])

logger = get_logger(__name__)

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# 聊天流并发上限：每个进行中的 SSE 占用一个线程池线程并消耗 LLM 配额，
# 超出直接拒绝，避免打满线程池/烧穿 token 额度。
_CHAT_SLOTS = threading.BoundedSemaphore(settings.max_concurrent_chats)

# 哨兵：next(gen, _CLOSED) 在生成器耗尽时返回，用于把 for 循环改写成 while
_CLOSED = object()


def _sse_frame(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _consume_task_exception(task: asyncio.Task) -> None:
    """客户端断开后仍在途的 next()：消费异常，避免 asyncio 未取回异常告警。"""
    if not task.cancelled():
        error = task.exception()
        if error is not None:
            logger.warning("聊天流在途事件异常：%s", error)


def _stream_response(sync_events: Iterator[str]) -> StreamingResponse:
    """把同步 SSE 生成器包成 async 生成器。

    Starlette 在客户端断开时只取消响应任务，从不 close body_iterator；
    若此刻同步生成器正阻塞在 LLM 调用里，它会永远挂起在 yield 上，
    run_stream 的 finally（checkpoint 终态 + 半截 turn 落库）不会执行，
    checkpoint 卡死在 running，互斥和恢复全部失效。这里在 async 生成器
    的 finally 里显式 close，保证收尾一定执行。

    另外在返回响应前先拉取首个事件：客户端在首块到达前断开时，
    Starlette 根本不会启动 async 生成器；若同步生成器也从未启动，
    其 finally（并发槽释放）将永远不会执行，槽泄漏累积后聊天接口
    会被永久打满 429。预拉一次保证生成器必然启动过，其 finally
    由 close 线程 / GC 兜底执行。

    心跳：研报深度层单次工具调用可跑十几分钟，期间既没有 text 也没有
    tool_progress，反代/网关的 idle 计时器会把 SSE 掐断。这里不是靠业务
    层定时发进度（会污染进度文案），而是在传输层用标准 SSE 注释帧
    ``: keepalive`` 保活：EventSource 自动忽略注释行，前端 data 事件
    契约逐字不变，字节流却能让网关把 idle timer 归零。
    """

    # 生成器异常退出时自身 finally 已释放并发槽，异常直接上抛
    first = next(sync_events, _CLOSED)
    heartbeat = max(0.0, float(settings.sse_heartbeat_seconds))

    async def stream() -> AsyncIterator[str]:
        next_task: asyncio.Task | None = None
        try:
            if first is not _CLOSED:
                yield first
            while True:
                if next_task is None:
                    next_task = asyncio.create_task(
                        run_in_threadpool(next, sync_events, _CLOSED)
                    )
                if heartbeat > 0:
                    # 超时不取消 next_task：底层线程可能正阻塞在 LLM/工具里，
                    # 取消只会留下孤儿线程。发完注释帧继续等同一个 task。
                    done, _ = await asyncio.wait({next_task}, timeout=heartbeat)
                    if not done:
                        yield ": keepalive\n\n"
                        continue
                    chunk = next_task.result()
                else:
                    # 关闭心跳时逐字恢复旧行为：不能对刚创建的 task 直接
                    # 调 .result()（还没跑完会抛 InvalidStateError）。
                    chunk = await next_task
                next_task = None
                if chunk is _CLOSED:
                    return
                yield chunk
        finally:
            if next_task is not None and not next_task.done():
                next_task.add_done_callback(_consume_task_exception)
            # 不能在这里 await：取消作用域内任何 await 都会立刻再抛
            # CancelledError。放到独立线程里等在途的 LLM 调用返回后关闭
            threading.Thread(
                target=_close_sync_generator, args=(sync_events,), daemon=True
            ).start()

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


def _close_sync_generator(gen: Iterator[str]) -> None:
    # 生成器还在执行（LLM 调用中）时 close 抛 ValueError（already
    # executing）：等它挂起到 yield 再关，最多等 120 秒（LLM 超时上限）
    deadline = time.monotonic() + 120
    while True:
        try:
            gen.close()
            return
        except ValueError:
            if time.monotonic() >= deadline:
                logger.warning("聊天流生成器关闭超时，交由僵尸 checkpoint 清理兜底")
                return
            time.sleep(0.2)


def chat_sse_events(
    user_id: str,
    session_id: str,
    content: str,
    use_web_search: bool = True,
    use_kb: bool = False,
    use_report: bool = True,
    kb_file_ids: tuple[str, ...] = (),
    memory_enabled: bool = True,
) -> Iterator[str]:
    # 并发槽已在路由处理器里 acquire（超限返回 429）；这里只负责释放
    try:
        answer_parts: list[str] = []
        for event in run_stream(
            content,
            user_id=user_id,
            chat_session_id=session_id,
            finalize_background=True,
            use_web_search=use_web_search,
            use_kb=use_kb,
            use_report=use_report,
            kb_file_ids=kb_file_ids,
            memory_enabled=memory_enabled,
        ):
            if event.get("type") == "text":
                answer_parts.append(event.get("content") or "")
            yield _sse_frame(event)

        # 标题任务后台执行并直接写库；前端通过 GET /v1/session/title 轮询结果，
        # SSE 流在 done 后立即结束，不再为标题阻塞最长 22 秒。
        start_title_job(user_id, session_id, content, "".join(answer_parts))
    finally:
        _CHAT_SLOTS.release()


def _stream(
    user_id: str,
    session_id: str,
    content: str,
    use_web_search: bool,
    use_kb: bool = False,
    use_report: bool = True,
    kb_file_ids: tuple[str, ...] = (),
    memory_enabled: bool = True,
) -> StreamingResponse:
    return _stream_response(
        chat_sse_events(
            user_id,
            session_id,
            content,
            use_web_search,
            use_kb,
            use_report,
            kb_file_ids,
            memory_enabled,
        )
    )


def _acquire_chat_slot() -> None:
    """并发槽必须在返回 StreamingResponse 前获取，超限才能真正响应 429。"""
    if not _CHAT_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="当前对话并发已达上限，请稍后再试")


@router.post("/chat")
def chat(
    req: ChatRequest, user: UserRecord = Depends(get_current_user)
) -> StreamingResponse:
    if not req.regenerate and not req.message.strip():
        raise HTTPException(status_code=422, detail="消息不能为空")
    if store.has_running_checkpoint(user.user_id, req.chat_session_id):
        raise HTTPException(status_code=409, detail="该会话有正在进行的回复")
    # 新回合 / 重新生成覆盖该会话未完成的断点，避免旧 interrupted 行
    # 在新回复完成后仍被 pending_run 当成「继续生成」
    try:
        store.discard_resumable_checkpoints(user.user_id, req.chat_session_id)
    except Exception:
        logger.exception("丢弃可恢复 checkpoint 失败")

    # 内置 default 会话无需预先创建，其余会话必须真实存在，防止垃圾历史堆积
    if req.chat_session_id != "default":
        try:
            session = store.get_session(user.user_id, req.chat_session_id)
        except Exception as e:
            raise AgentError("会话查询失败") from e
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        try:
            store.touch_session(user.user_id, req.chat_session_id)
        except Exception:
            logger.exception("刷新会话活跃时间失败")
    _acquire_chat_slot()
    if req.regenerate:
        return _stream_regenerate(
            user.user_id,
            req.chat_session_id,
            req.web_search,
            req.use_kb,
            req.use_report,
            tuple(req.kb_file_ids),
            memory_enabled=user.memory_enabled,
        )
    return _stream(
        user.user_id,
        req.chat_session_id,
        req.message,
        req.web_search,
        req.use_kb,
        req.use_report,
        tuple(req.kb_file_ids),
        memory_enabled=user.memory_enabled,
    )


def _stream_regenerate(
    user_id: str,
    session_id: str,
    use_web_search: bool,
    use_kb: bool = False,
    use_report: bool = True,
    kb_file_ids: tuple[str, ...] = (),
    memory_enabled: bool = True,
) -> StreamingResponse:
    """重新生成最后一条回答：不重复保存用户消息，末尾旧回答先从历史剔除。"""

    def events() -> Iterator[str]:
        try:
            answer_parts: list[str] = []
            for event in run_stream(
                "",
                user_id=user_id,
                chat_session_id=session_id,
                finalize_background=True,
                regenerate=True,
                use_web_search=use_web_search,
                use_kb=use_kb,
                use_report=use_report,
                kb_file_ids=kb_file_ids,
                memory_enabled=memory_enabled,
            ):
                if event.get("type") == "text":
                    answer_parts.append(event.get("content") or "")
                yield _sse_frame(event)
            # 首轮流被暂停/中断时标题任务不会触发，这里补一次。
            # start_title_job 内部有守卫：仅当标题仍是"新会话"才生成，
            # 用户改过名或已生成过会直接跳过，不会覆盖。
            start_title_job(
                user_id,
                session_id,
                _last_user_question(user_id, session_id),
                "".join(answer_parts),
            )
        finally:
            _CHAT_SLOTS.release()

    return _stream_response(events())


def _last_user_question(user_id: str, session_id: str) -> str:
    try:
        tail = store.load_recent_turns(user_id, session_id, turns=1)
    except Exception:
        logger.warning("读取最后一条用户消息失败 session_id=%s", session_id)
        return ""
    return next(
        (str(m.get("content") or "") for m in tail if m.get("role") == "user"),
        "",
    )


@router.post("/chat/resume")
def chat_resume(
    req: ResumeRequest, user: UserRecord = Depends(get_current_user)
) -> StreamingResponse:
    try:
        cp = store.load_resumable_checkpoint(req.run_id)
    except Exception as e:
        raise AgentError("checkpoint 查询失败") from e
    # 归属校验：不是自己的 run 一律 404（不泄露存在性）
    if (
        cp is None
        or cp["user_id"] != user.user_id
        or cp["chat_session_id"] != req.chat_session_id
    ):
        raise HTTPException(status_code=404, detail="没有可恢复的对话")
    # session 互斥：该会话已有进行中的 run，拒绝，防止两条 run 交错写历史
    if store.has_running_checkpoint(user.user_id, cp["chat_session_id"]):
        raise HTTPException(status_code=409, detail="该会话有正在进行的回复")
    _acquire_chat_slot()
    return _stream_resume(
        user.user_id,
        req.run_id,
        cp["chat_session_id"],
        memory_enabled=user.memory_enabled,
    )


def _stream_resume(
    user_id: str,
    resume_run_id: str,
    session_id: str,
    memory_enabled: bool = True,
) -> StreamingResponse:
    def events() -> Iterator[str]:
        try:
            answer_parts: list[str] = []
            saw_answer = False
            for event in run_stream(
                "",
                user_id=user_id,
                chat_session_id="default",  # 占位，run_stream 会从 checkpoint 覆盖
                finalize_background=True,
                resume_run_id=resume_run_id,
                memory_enabled=memory_enabled,
            ):
                if event.get("type") == "text":
                    answer_parts.append(event.get("content") or "")
                    saw_answer = True
                elif event.get("type") == "done":
                    saw_answer = True
                yield _sse_frame(event)
            # 首轮流被中断时标题任务不会触发，恢复成功后这里补一次。
            # 仅当本轮有实际输出才触发：claim 失败等 error-only 流
            # 不该为空回答生成标题。start_title_job 内部守卫保证不覆盖
            # 已生成的标题。
            if saw_answer:
                start_title_job(
                    user_id,
                    session_id,
                    _last_user_question(user_id, session_id),
                    "".join(answer_parts),
                )
        finally:
            _CHAT_SLOTS.release()

    return _stream_response(events())


@router.get("/chat/pending_run")
def chat_pending_run(
    user: UserRecord = Depends(get_current_user),
    chat_session_id: SafeId = Query("default"),
) -> dict:
    try:
        run_id = store.latest_resumable_run(user.user_id, chat_session_id)
    except Exception as e:
        raise AgentError("checkpoint 查询失败") from e
    return {"run_id": run_id}


@router.get("/chat/memory", response_model_exclude_none=True)
def chat_memory(
    user: UserRecord = Depends(get_current_user),
    chat_session_id: SafeId = Query("default"),
    limit: int = Query(20, ge=1, le=100),
    before_seq: int | None = Query(None, ge=1),
) -> ChatMemoryResponse:
    try:
        page = store.load_chat_page(
            user.user_id, chat_session_id, limit=limit, before_seq=before_seq
        )
    except Exception as e:
        raise AgentError("会话记忆加载失败") from e
    return ChatMemoryResponse(
        user_id=user.user_id,
        chat_session_id=chat_session_id,
        messages=[ChatMemoryMessage.model_validate(msg) for msg in page.messages],
        has_more=page.has_more,
        next_before_seq=page.next_before_seq,
    )
