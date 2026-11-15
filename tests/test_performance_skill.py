"""业绩分析 skill：命中时脚本化预取，不命中/失败时降级给模型。"""

from unittest.mock import patch

from ai_agent.agents._loop.performance import (
    Prefetch,
    _period_of,
    already_injected,
    prefetch_once,
    wants_performance,
)


def _prefetch(text="预取内容", tools=("report_extract_indicators",)):
    return Prefetch(text=text, tools=tools)


# ——— 命中判定 ———

def test_wants_performance_on_metric_words():
    for text in ("中科飞测的业绩情况", "绿的谐波营收多少", "这家公司利润如何", "财务情况"):
        assert wants_performance(text), text


def test_wants_performance_ignores_report_and_export():
    """要求生成研报/导出时不该走业绩 skill——那是另一条流水线。"""
    for text in ("帮我生成一份研报", "把业绩表导出成 PDF"):
        assert not wants_performance(text), text


def test_wants_performance_false_without_metric_words():
    assert not wants_performance("")
    assert not wants_performance("中科飞测是做什么的")


# ——— 报告期解析 ———

def test_period_of_parses_common_forms():
    assert _period_of("中科飞测2026Q3业绩") == "2026Q3"
    assert _period_of("2026 年 Q3 营收") == "2026Q3"
    assert _period_of("2026年上半年业绩") == "2026H1"
    assert _period_of("中科飞测的业绩情况") == ""


# ——— 注入与幂等 ———

def test_prefetch_once_injects_and_is_idempotent():
    messages = [{"role": "user", "content": "中科飞测的业绩情况"}]
    with patch(
        "ai_agent.agents._loop.performance.prefetch", return_value=_prefetch()
    ):
        first = prefetch_once(messages, enabled=True)
        assert first is not None
        assert already_injected(messages)
        # 续跑：已有预取结果，不再重复
        assert prefetch_once(messages, enabled=True) is None


def test_prefetch_once_skipped_when_disabled_or_already_retrieved():
    messages = [{"role": "user", "content": "中科飞测的业绩情况"}]
    with patch("ai_agent.agents._loop.performance.prefetch", return_value=_prefetch()):
        assert prefetch_once(messages, enabled=False) is None
        assert prefetch_once(messages, enabled=True, already_retrieved=True) is None
    assert not already_injected(messages)


# ——— 降级 ———

def test_prefetch_returns_none_without_subjects():
    from ai_agent.agents._loop import performance

    with patch(
        "ai_agent.kb.retrieval.mentioned_subjects", return_value=set()
    ):
        assert performance.prefetch("中科飞测的业绩") is None


def test_prefetch_returns_none_when_extraction_empty():
    from ai_agent.agents._loop import performance
    from ai_agent.report.extract import ExtractResult

    with (
        patch("ai_agent.kb.retrieval.mentioned_subjects", return_value={"中科飞测"}),
        patch.object(performance, "_latest_period", return_value="2026Q3"),
        patch(
            "ai_agent.tools.report.report_extract._extract_by_basis",
            return_value=ExtractResult(),
        ),
    ):
        assert performance.prefetch("中科飞测的业绩") is None


def test_loop_drops_extract_tool_when_prefetched():
    """预取成功后不再提供 report_extract_indicators。

    prompt 说「不要重复抽」拦不住——实测三轮模型每轮都调了一次。
    """
    from ai_agent.agents.loop import run_loop

    schemas = [
        {"function": {"name": "report_extract_indicators"}},
        {"function": {"name": "kb_search"}},
        {"function": {"name": "kg_search"}},
        {"function": {"name": "kb_read"}},
    ]
    seen: list[list[str]] = []

    def fake_stream(messages, tools, **kwargs):
        seen.append([t["function"]["name"] for t in tools])
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.tool_schemas", return_value=schemas),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
        patch(
            "ai_agent.agents.loop._prefetch_performance_once",
            return_value=_prefetch(),
        ),
        patch("ai_agent.kb.retrieval.mentioned_subjects", return_value={"中科飞测"}),
    ):
        list(run_loop([{"role": "user", "content": "中科飞测的业绩情况"}], use_kb=True))

    assert "report_extract_indicators" not in seen[0]
    assert "kb_search" in seen[0]


def test_loop_keeps_extract_tool_when_not_prefetched():
    """没命中 skill（预取返回 None）时工具照旧可用，不误伤普通问法。"""
    from ai_agent.agents.loop import run_loop

    schemas = [
        {"function": {"name": "report_extract_indicators"}},
        {"function": {"name": "kb_search"}},
    ]
    seen: list[list[str]] = []

    def fake_stream(messages, tools, **kwargs):
        seen.append([t["function"]["name"] for t in tools])
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.tool_schemas", return_value=schemas),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
        patch("ai_agent.agents.loop._prefetch_performance_once", return_value=None),
        patch("ai_agent.kb.retrieval.mentioned_subjects", return_value=set()),
    ):
        list(run_loop([{"role": "user", "content": "你好"}], use_kb=True))

    assert "report_extract_indicators" in seen[0]
