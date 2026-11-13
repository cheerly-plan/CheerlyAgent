"""一轮对话的持久化与收尾。

从 loop.py 拆出(S5)。负责短期记忆的读写、断点（checkpoint）的创建/更新/完成，
以及流结束后的后台收尾（长期记忆抽取、摘要压缩）——后者放到线程池，不阻塞
SSE 尾部延迟。
"""

from __future__ import annotations

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.llm.types import ChatMessage
from ai_agent.memory.long_term import capture_long_term
from ai_agent.memory.prune import prune_tool_pairs, removed_tool_call_ids
from ai_agent.memory.short_term import compress, short_term_memory
from ai_agent.tools.result_pool import tool_result_pool
from ai_agent.storage.store import store
from ai_agent.memory.summarize import summarize_session

logger = get_logger(__name__)



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


def _session_state(
    user_id: str, chat_session_id: str
) -> tuple[list[ChatMessage], str, int]:
    cached = short_term_memory.load(chat_session_id, user_id=user_id)
    if cached is not None:
        return cached.messages, cached.summary, cached.summary_until_seq
    try:
        summary, cursor = store.get_session_state(user_id, chat_session_id)
        recent = store.load_all_turns(
            user_id, chat_session_id, after_seq=cursor
        )
    except Exception:
        logger.exception("会话记忆加载失败")
        return [], "", 0
    return recent, summary, cursor


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
    user_id: str,
    turn: list[ChatMessage],
    summary: str,
    user_input: str,
    chat_session_id: str = "",
    memory_enabled: bool = True,
) -> None:
    if not memory_enabled:
        return
    try:
        capture_long_term(
            user_id,
            turn,
            summary,
            user_input,
            chat_session_id=chat_session_id,
            memory_enabled=memory_enabled,
        )
    except Exception:
        logger.exception("长期记忆抽取失败")


def _compress_and_cache(
    user_id: str,
    chat_session_id: str,
    messages: list[ChatMessage],
    summary: str,
    summary_until_seq: int = 0,
) -> None:
    window, new_summary, new_cursor = messages, summary, summary_until_seq
    try:
        window, new_summary, new_cursor = compress(
            messages,
            old_summary=summary,
            summarize=summarize_session,
            # 与 _fit_context 同一个口径：被 token 预算挤掉的轮次也要进摘要，
            # 否则它们只存在于「送模型副本」的裁剪里，下一轮就彻底找不回。
            max_tokens=settings.context_message_token_budget(),
            old_cursor=summary_until_seq,
        )
    except Exception:
        logger.exception("短期记忆压缩失败")
    # 摘要/裁剪后不再进入活跃窗口的工具对，连同池快照一起淘汰；
    # 原文仍在 agent_chat_history，read_tool_result 可 DB 兜底。
    target = settings.context_message_token_budget()
    window = prune_tool_pairs(window, target, consumed_only=True)
    window = prune_tool_pairs(window, target)
    removed_ids = removed_tool_call_ids(messages, window)
    tool_result_pool.evict_calls(user_id, chat_session_id, removed_ids)

    if new_summary != summary or new_cursor != summary_until_seq:
        try:
            store.update_session_summary(
                user_id,
                chat_session_id,
                new_summary,
                summary_until_seq=new_cursor if new_cursor > 0 else None,
            )
        except Exception:
            logger.exception("会话摘要保存失败")
    short_term_memory.save(
        chat_session_id,
        window,
        user_id=user_id,
        summary=new_summary,
        summary_until_seq=new_cursor,
    )
