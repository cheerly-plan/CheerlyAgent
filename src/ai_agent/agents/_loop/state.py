"""共享的可变模块级单例（工具进度线程池）。

由原 agents/loop.py 拆分而来，逻辑未改动。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


# 测试 monkeypatch 打在 ``ai_agent.agents.loop`` 上；内部调用一律经
# _public() 取当前属性，否则补丁会被本地绑定绕过（静默失效）。
def _public():
    import ai_agent.agents.loop as public

    return public


_tool_progress_pool = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="tool-progress"
)
