"""工具执行超时兜底：工具挂死不能拖死整个 run。

背景：get_weather 访问 open-meteo 时 DNS 卡死，单次调用阻塞了 8 分钟，
期间 execute_tool 不返回，run_loop 停在原地，前端一直转圈。
"""

import threading
import time

import pytest

from pydantic import BaseModel

from ai_agent.tools import registry
from ai_agent.tools.base import Tool


class _Args(BaseModel):
    pass


def _hang_tool(seconds: float) -> Tool:
    def _func(**_kwargs):
        time.sleep(seconds)
        return "迟到的结果"

    return Tool(name="_hang", description="测试用挂死工具", func=_func, args_model=_Args)


@pytest.fixture
def register_tool(monkeypatch):
    def _install(tool: Tool) -> None:
        monkeypatch.setitem(registry.TOOLS_MAP, tool.name, tool)

    return _install


def test_timeout_returns_error_text_and_does_not_block(register_tool, monkeypatch):
    """超过预算立刻返回超时文案，而不是等工具跑完。"""
    register_tool(_hang_tool(5))
    monkeypatch.setattr(registry.settings, "tool_execute_timeout_seconds", 0.3)

    started = time.monotonic()
    out = registry.execute_tool("_hang", {})
    elapsed = time.monotonic() - started

    assert out.startswith("ERROR: 工具执行超时")
    assert "不要用相同参数重试" in out
    assert elapsed < 3, f"应当按预算返回，实际等了 {elapsed:.1f}s"


def test_fast_tool_untouched(register_tool, monkeypatch):
    """预算内正常返回，超时机制不能改变原有输出。"""
    register_tool(_hang_tool(0))
    monkeypatch.setattr(registry.settings, "tool_execute_timeout_seconds", 10)

    assert registry.execute_tool("_hang", {}) == "迟到的结果"


def test_zero_budget_disables_timeout(register_tool, monkeypatch):
    register_tool(_hang_tool(0.2))
    monkeypatch.setattr(registry.settings, "tool_execute_timeout_seconds", 0)

    assert registry.execute_tool("_hang", {}) == "迟到的结果"


def test_sentinel_distinguishes_none_result(monkeypatch):
    """工具合法返回 None 时不能被当成超时。"""
    monkeypatch.setattr(registry.settings, "tool_execute_timeout_seconds", 5)

    assert registry._call_with_timeout(lambda: None, 5) is None
    assert registry._call_with_timeout(lambda: "ok", 5) == "ok"
    assert registry._call_with_timeout(lambda: time.sleep(2), 0.1) is registry._TIMED_OUT


def test_exception_still_propagates(monkeypatch):
    monkeypatch.setattr(registry.settings, "tool_execute_timeout_seconds", 5)

    with pytest.raises(ValueError):
        registry._call_with_timeout(lambda: (_ for _ in ()).throw(ValueError("boom")), 5)


def test_hung_thread_is_daemon(register_tool, monkeypatch):
    """挂死的线程留在后台，不能阻止进程退出。"""
    register_tool(_hang_tool(1))
    monkeypatch.setattr(registry.settings, "tool_execute_timeout_seconds", 0.1)
    before = {t.ident for t in threading.enumerate()}

    registry.execute_tool("_hang", {})

    hung = [
        t for t in threading.enumerate()
        if t.ident not in before and t.name == "tool-exec"
    ]
    assert hung, "挂起的调用线程应当仍在后台"
    assert all(t.daemon for t in hung)

def test_explicit_zero_timeout_bypasses_budget(register_tool, monkeypatch):
    """进度型长任务传 timeout=0：不受默认短预算影响，完整跑完。"""
    register_tool(_hang_tool(0.4))
    monkeypatch.setattr(registry.settings, "tool_execute_timeout_seconds", 0.1)

    assert registry.execute_tool("_hang", {}, timeout=0) == "迟到的结果"


def test_explicit_timeout_overrides_settings(register_tool, monkeypatch):
    register_tool(_hang_tool(5))
    monkeypatch.setattr(registry.settings, "tool_execute_timeout_seconds", 999)

    started = time.monotonic()
    out = registry.execute_tool("_hang", {}, timeout=0.3)
    assert out.startswith("ERROR: 工具执行超时")
    assert time.monotonic() - started < 3
