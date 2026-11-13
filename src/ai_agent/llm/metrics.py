"""轻量链路的进程内计数器：让"静默降级"变得看得见。

记忆抽取、会话摘要、标题这几条链路的设计原则是**失败不阻断对话**——出错就
退回摘录、退回压缩主题。代价是它们可能长期失效而没人发现：这轮之前
`agent_episode` / `agent_user_profile` 一直是 0 行，日志里只有一行 warning。

这里记两类事：

* **降级**（`memory.extract.zero`、`summary.crude`、`llm.native_fallback`）：
  本该由小模型产出的东西退回了兜底方案；
* **失败**（`ollama.native.error`）：原生调用直接报错。

计数器挂进程内、无锁争用的代价可以忽略；`/health` 会把它带出去（只报数字，
不暴露内部路径），管理员巡检时一眼能看出"记忆是不是又在空转"。
"""

from __future__ import annotations

from threading import Lock
from typing import Final

_LOCK = Lock()
_COUNTERS: dict[str, int] = {}

# 只导出这几个固定名字，避免各处随手拼字符串导致 /health 里长出一堆一次性 key
OK: Final = "ok"
MEMORY_ZERO: Final = "memory.extract.zero"
MEMORY_OK: Final = "memory.extract.ok"
SUMMARY_CRUDE: Final = "summary.crude"
SUMMARY_OK: Final = "summary.ok"
TITLE_FALLBACK: Final = "title.fallback"
TITLE_OK: Final = "title.ok"
NATIVE_ERROR: Final = "llm.native.error"
NATIVE_EMPTY: Final = "llm.native.empty"
NATIVE_FALLBACK: Final = "llm.native_fallback"
CONTEXT_SOFT: Final = "context.shrink.soft"
CONTEXT_HARD: Final = "context.shrink.hard"
TOOL_RESULT_POOL_HIT: Final = "tool_result.pool.hit"
TOOL_RESULT_POOL_MISS: Final = "tool_result.pool.miss"


def incr(name: str, count: int = 1) -> None:
    if count <= 0:
        return
    with _LOCK:
        _COUNTERS[name] = _COUNTERS.get(name, 0) + count


def snapshot() -> dict[str, int]:
    """当前计数快照（只返回非零项，避免 /health 里全是一堆 0）。"""
    with _LOCK:
        return {name: value for name, value in sorted(_COUNTERS.items()) if value}


def reset() -> None:
    """测试用；生产不调用（计数器是进程生命周期的累计值）。"""
    with _LOCK:
        _COUNTERS.clear()