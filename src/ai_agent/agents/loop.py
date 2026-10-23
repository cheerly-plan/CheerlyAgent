from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from ai_agent.agents.prompts import SYSTEM_PROMPT
from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import stream_chat
from ai_agent.llm.types import ChatEvent, ChatMessage, ChatResult
from ai_agent.memory.long_term import capture_long_term
from ai_agent.memory.facets import format_profile_block
from ai_agent.memory.short_term import assemble, compress, short_term_memory
from ai_agent.memory.store import store, user_scope
from ai_agent.memory.summarize import summarize_session
from ai_agent.tools.registry import execute, schemas

logger = get_logger(__name__)

# 流结束后的收尾工作（长期记忆抽取/摘要压缩含多次 LLM 调用）放到后台线程，
# 不再阻塞 SSE 尾部延迟；进程退出时由 lifespan 调 shutdown_turn_finalizer 等待。
_turn_finalizer = ThreadPoolExecutor(max_workers=4, thread_name_prefix="turn-finalize")


def shutdown_turn_finalizer(wait: bool = True) -> None:
    """等待在途的记忆抽取/摘要任务完成后再关停，避免重启丢长期记忆。"""
    _turn_finalizer.shutdown(wait=wait, cancel_futures=False)


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


def _session_state(
    user_id: str, chat_session_id: str
) -> tuple[list[ChatMessage], str]:
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


def _persist_turn(
    user_id: str, chat_session_id: str, turn: list[ChatMessage]
) -> None:
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
) -> Iterator[ChatEvent]:
    recent, summary = _session_state(user_id, chat_session_id)
    replacing = False
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
    failed = False
    try:
        for event in run_loop(messages, user_id):
            if event.get("type") == "error":
                failed = True
            yield event
    finally:
        turn = messages[persisted:]
        if replacing and (failed or not any(m.get("role") == "assistant" for m in turn)):
            # 重新生成失败（LLM 超时/报错/空结果）：什么都不写，
            # 库里的旧回答和热缓存原样保留，用户可再点一次
            return
        if replacing:
            _replace_last_turn(user_id, chat_session_id, turn)
        if finalize_background:
            # 先同步完成快路径（落库 + 热缓存），保证下一轮上下文连续；
            # 慢路径（记忆抽取、摘要压缩，含多次 LLM 调用）交给后台线程。
            if not replacing:
                _persist_turn(user_id, chat_session_id, turn)
            short_term_memory.save(
                chat_session_id, messages, user_id=user_id, summary=summary
            )
            _turn_finalizer.submit(
                _capture_turn, user_id, turn, summary, user_input
            )
            _turn_finalizer.submit(
                _compress_and_cache, user_id, chat_session_id, messages, summary
            )
        else:
            if not replacing:
                _persist_turn(user_id, chat_session_id, turn)
            _capture_turn(user_id, turn, summary, user_input)
            _compress_and_cache(user_id, chat_session_id, messages, summary)


def run_loop(
    messages: list[ChatMessage], user_id: str = "default"
) -> Iterator[ChatEvent]:
    for step in range(settings.max_turns):
        result: ChatResult | None = None
        try:
            for kind, payload in stream_chat(messages, schemas()):
                if kind == "text":
                    yield {"type": "text", "content": payload}
                else:
                    result = payload
        except LLMError as e:
            # 详情只进日志；上游异常文本可能带出 URL/响应片段，不透传给客户端
            logger.error("LLM 调用失败: %s", e)
            yield {"type": "error", "message": "模型服务暂时不可用，请稍后重试"}
            return

        if result is None:
            yield {"type": "error", "message": "模型没有返回结果"}
            return

        logger.info("step %s tool_calls: %s", step, result["tool_calls"])
        assistant: ChatMessage = {"role": "assistant", "content": result["content"]}
        if result["tool_calls"]:
            assistant["tool_calls"] = result["tool_calls"]
        messages.append(assistant)
        if not result["tool_calls"]:
            yield {"type": "done"}
            return
        for call in result["tool_calls"]:
            yield {"type": "tool_start", "name": call["name"]}
            with user_scope(user_id):
                output = execute(call["name"], call["arguments"])
            yield {"type": "tool_end", "name": call["name"]}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": output,
                }
            )
    yield {"type": "error", "message": f"已超过最大轮数 {settings.max_turns}。"}


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
