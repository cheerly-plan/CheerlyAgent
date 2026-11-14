"""agents 主循环的实现层（内部）。对外请用 ai_agent.agents.loop。"""

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

__all__ = [
    "_tool_progress_pool",
    "_llm_error_message",
    "_profile_prompt",
    "_recall_block",
    "_kb_capability_block",
    "_period_inventory",
    "_system_prompt",
    "_today_line",
    "_KB_CAP_MAX_DOC_TYPES",
    "_DYNAMIC_CLIP_MARK",
    "_latest_user_text",
    "_question_already_retrieved",
    "_followup_subjects",
    "_kg_prefetch_present",
    "_attach_kg_prefetch",
    "_must_call_kb_tool",
    "_KB_RETRIEVAL_TOOLS",
    "_KG_PREFETCH_MARK",
    "_REPORT_VERB",
    "_REPORT_NOUN",
    "_run_tool_with_progress",
    "_remember_tool_result",
    "_attach_sources",
    "_PROGRESS_TOOLS",
    "shutdown_turn_finalizer",
    "_fitted_context",
    "run_stream",
    "run_loop",
    "run",
    "_turn_finalizer",
]
