"""Agent 主循环。

本文件是对外入口，实现在 ``_loop/`` 子包（依赖方向单一、无环）：

    state       共享可变单例（工具进度线程池）
    prompts     系统提示词拼装（画像/回忆/知识库能力/报告期清单）
    followup    追问与预检索判定（主体延续、图谱预取、强制检索）
    tools_exec  工具执行与结果回收
    loop        主循环编排 run_loop / run_stream / run

**兼容层**：拆分前本模块里的所有符号（含从别处 import 进来的）都在这里
原样保留，所以 ``from ai_agent.agents.loop import X`` 与
``patch("ai_agent.agents.loop.store.list_profile")`` 这类写法都不变。

补丁之所以仍然生效：子模块内部对这些符号的调用一律写成 ``_public().X(...)``，
取的是本模块**当前**的属性；若直接本地绑定调用，补丁会被绕过而静默失效。
"""

import queue
import re
import time
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
    RUNTIME_TOOL_TIMEOUT,
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

from ai_agent.agents._loop.state import (
    _tool_progress_pool,
)

from ai_agent.agents._loop.prompts import (
    _llm_error_message,
    _profile_prompt,
    _recall_block,
    _kb_capability_block,
    _period_inventory,
    _system_prompt,
    _today_line,
    _KB_CAP_MAX_DOC_TYPES,
    _DYNAMIC_CLIP_MARK,
)

from ai_agent.agents._loop.followup import (
    _latest_user_text,
    _question_already_retrieved,
    _followup_subjects,
    _kg_prefetch_present,
    _attach_kg_prefetch,
    _must_call_kb_tool,
    _KB_RETRIEVAL_TOOLS,
    _KG_PREFETCH_MARK,
    _REPORT_VERB,
    _REPORT_NOUN,
)

from ai_agent.agents._loop.tools_exec import (
    _run_tool_with_progress,
    _remember_tool_result,
    _attach_sources,
    _PROGRESS_TOOLS,
)

from ai_agent.agents._loop.loop import (
    shutdown_turn_finalizer,
    _fitted_context,
    run_stream,
    run_loop,
    run,
    _turn_finalizer,
)

logger = get_logger(__name__)

__all__ = [
    "Budget",
    "ChatEvent",
    "ChatMessage",
    "ChatResult",
    "Iterator",
    "LLMError",
    "RUNTIME_BUDGET_EXHAUSTED",
    "RUNTIME_DUP_CALL",
    "RUNTIME_DUP_OUTPUT",
    "RUNTIME_EMPTY_ANSWER",
    "RUNTIME_TOOL_TIMEOUT",
    "ThreadPoolExecutor",
    "_DYNAMIC_CLIP_MARK",
    "_KB_CAP_MAX_DOC_TYPES",
    "_KB_RETRIEVAL_TOOLS",
    "_KG_PREFETCH_MARK",
    "_PROGRESS_TOOLS",
    "_REPORT_NOUN",
    "_REPORT_VERB",
    "_attach_kg_prefetch",
    "_attach_sources",
    "_canonical_args",
    "_capture_turn",
    "_checkpoint_create",
    "_checkpoint_finish",
    "_checkpoint_update",
    "_compress_and_cache",
    "_dedupe_seam",
    "_drop_empty_assistant",
    "_fit_context",
    "_fitted_context",
    "_followup_subjects",
    "_kb_capability_block",
    "_keep_prefill_tail",
    "_kg_prefetch_present",
    "_latest_user_text",
    "_llm_error_message",
    "_must_call_kb_tool",
    "_output_digest",
    "_period_inventory",
    "_persist_turn",
    "_pop_incomplete_assistant",
    "_profile_prompt",
    "_question_already_retrieved",
    "_recall_block",
    "_remember_tool_result",
    "_replace_last_turn",
    "_run_tool_with_progress",
    "_session_state",
    "_stash_partial_assistant",
    "_system_prompt",
    "_today_line",
    "_tool_progress_pool",
    "_turn_finalizer",
    "assemble",
    "build_memory_index",
    "build_system_prompt",
    "clip_text",
    "collect_citations",
    "count_json_tokens",
    "execute_tool",
    "extract_prev_turn",
    "format_profile_block",
    "format_recall_block",
    "full_trace",
    "get_logger",
    "merge_sources",
    "queue",
    "re",
    "rerank_items",
    "reset_event_emitter",
    "reset_kb_question",
    "reset_kb_scope",
    "reset_progress",
    "reset_rewrite_context",
    "resolve_profile",
    "run",
    "run_loop",
    "run_stream",
    "session_scope",
    "set_event_emitter",
    "set_kb_question",
    "set_kb_scope",
    "set_progress",
    "set_rewrite_context",
    "settings",
    "short_term_memory",
    "shutdown_turn_finalizer",
    "skill_catalog",
    "store",
    "stream_chat",
    "time",
    "tool_result_index",
    "tool_result_pool",
    "tool_results_scope",
    "tool_schemas",
    "user_scope",
    "uuid",
]
