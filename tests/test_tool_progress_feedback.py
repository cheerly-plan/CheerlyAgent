"""工具调用的可见性与防打转。

真实事故：用户问「一周热点新闻」，模型连调 13 次工具，其中同一份行情数据
换了 6 种关键词反复搜（那天是周六、没有行情），前端全程只有「正在处理…」。
"""

from unittest.mock import patch

from ai_agent.agents._loop.tools_exec import (
    _tool_call_index_factory,
    _tool_call_label,
    _tool_progress_hint,
)


# ——— 方案 B：tool_start 带可读标签 ———

def test_label_extracts_query_for_search_tools():
    assert _tool_call_label("web_search", {"query": "本周热点新闻"}) == "本周热点新闻"
    assert _tool_call_label("kb_search", {"query": "绿的谐波 营收"}) == "绿的谐波 营收"


def test_label_joins_list_arguments():
    assert (
        _tool_call_label("report_extract_indicators", {"subjects": ["绿的谐波", "中大力德"]})
        == "绿的谐波、中大力德"
    )


def test_label_truncates_long_text():
    label = _tool_call_label("web_search", {"query": "一" * 200})
    assert len(label) == 48
    assert label.endswith("…")


def test_label_empty_when_no_key_argument():
    assert _tool_call_label("kb_list_files", {}) == ""
    assert _tool_call_label("web_search", {}) == ""
    assert _tool_call_label("unknown_tool", {"query": "x"}) == ""


def test_label_skips_blank_and_bool_values():
    assert _tool_call_label("web_search", {"query": "   "}) == ""
    assert _tool_call_label("report_compare", {"points": True}) == ""


def test_call_index_counts_per_tool():
    index = _tool_call_index_factory()
    assert index("web_search") == 1
    assert index("web_search") == 2
    # 分工具独立计数
    assert index("kb_search") == 1
    assert index("web_search") == 3


# ——— 方案 C：重复结果达到阈值后封掉工具 ———

def test_tool_start_event_carries_args_and_label():
    """tool_start 带上参数与可读标签，前端能显示「在查什么」。"""
    from ai_agent.agents.loop import run_loop

    schemas = [{"function": {"name": "web_search"}}]
    calls = {"n": 0}

    def fake_stream(messages, tools, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "web_search",
                            "arguments": {"query": "本周热点新闻"},
                        }
                    ],
                },
            )
            return
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.tool_schemas", return_value=schemas),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
        patch("ai_agent.agents.loop.execute_tool", return_value="搜索结果"),
        patch("ai_agent.kb.retrieval.mentioned_subjects", return_value=set()),
    ):
        events = list(run_loop([{"role": "user", "content": "一周热点新闻"}]))

    starts = [e for e in events if e.get("type") == "tool_start"]
    assert starts
    assert starts[0]["name"] == "web_search"
    assert starts[0]["label"] == "本周热点新闻"
    assert starts[0]["call_index"] == 1
    assert starts[0]["args"]["query"] == "本周热点新闻"


def test_repeated_identical_output_blocks_tool():
    """同一工具连续返回相同内容 RETRY_LIMIT 次后，本轮不再执行它。"""
    from ai_agent.agents.loop import run_loop
    from ai_agent.prompts.chat import RETRY_LIMIT

    schemas = [{"function": {"name": "web_search"}}]
    executed: list[str] = []
    calls = {"n": 0}

    def fake_execute(name, arguments, **kwargs):
        executed.append(name)
        return "完全相同的搜索结果"

    def fake_stream(messages, tools, **kwargs):
        calls["n"] += 1
        # 一直要求调 web_search，每次换关键词（避开同参数短路），模拟打转
        yield (
            "result",
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"c{calls['n']}",
                        "name": "web_search",
                        "arguments": {"query": f"关键词{calls['n']}"},
                    }
                ],
            },
        )

    with (
        patch("ai_agent.agents.loop.tool_schemas", return_value=schemas),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
        patch("ai_agent.agents.loop.execute_tool", side_effect=fake_execute),
        patch("ai_agent.kb.retrieval.mentioned_subjects", return_value=set()),
    ):
        list(run_loop([{"role": "user", "content": "一周热点新闻"}]))

    # 第 RETRY_LIMIT 次拿到相同输出后封掉工具，之后的调用不再真正执行
    assert len(executed) == RETRY_LIMIT, executed


# ——— 方案 D：单工具调用次数上限 ———

def _run_with_budget(limit: int, *, same_output: bool = False) -> list[str]:
    """一直要求调 web_search，返回实际执行过的工具序列。"""
    from ai_agent.agents.budget import Budget
    from ai_agent.agents.loop import run_loop

    schemas = [{"function": {"name": "web_search"}}]
    executed: list[str] = []
    calls = {"n": 0}

    def fake_execute(name, arguments, **kwargs):
        executed.append(name)
        # 结果各不相同，避开方案 C 的「相同结果封禁」，只测次数预算
        return f"结果{calls['n']}" if not same_output else "完全相同"

    def fake_stream(messages, tools, **kwargs):
        calls["n"] += 1
        yield (
            "result",
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"c{calls['n']}",
                        "name": "web_search",
                        "arguments": {"query": f"关键词{calls['n']}"},
                    }
                ],
            },
        )

    with (
        patch("ai_agent.agents.loop.tool_schemas", return_value=schemas),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
        patch("ai_agent.agents.loop.execute_tool", side_effect=fake_execute),
        patch("ai_agent.kb.retrieval.mentioned_subjects", return_value=set()),
    ):
        list(
            run_loop(
                [{"role": "user", "content": "一周热点新闻"}],
                budget=Budget(max_steps=20, max_tool_calls=limit),
            )
        )
    return executed


def test_tool_call_budget_stops_single_tool():
    """结果各不相同（不触发方案 C）时，次数预算仍然兜住打转。"""
    executed = _run_with_budget(4)
    assert len(executed) == 4


def test_tool_call_budget_zero_means_unlimited():
    executed = _run_with_budget(0)
    # 上限 0 = 不限，只受 max_steps=20 约束
    assert len(executed) == 20


def test_retry_limit_fires_before_budget_when_output_identical():
    """相同结果时方案 C 先触发（3 次），不必等到次数预算。"""
    executed = _run_with_budget(10, same_output=True)
    assert len(executed) == 3


def test_budget_defaults_from_settings():
    from ai_agent.agents.budget import Budget
    from ai_agent.core.config import settings

    budget = Budget.from_settings()
    assert budget.max_steps == settings.max_turns
    assert budget.max_tool_calls == settings.max_tool_calls_per_run


# ——— 修法 1：补发 tool_progress（前端实际渲染的就是它）———


def test_hint_formats_verb_label_and_index():
    from ai_agent.agents._loop.tools_exec import _tool_progress_hint

    assert _tool_progress_hint("web_search", "本周热点新闻", 1) == "正在搜索「本周热点新闻」…"
    assert (
        _tool_progress_hint("web_search", "本周热点新闻", 3)
        == "正在搜索「本周热点新闻」（第 3 次）…"
    )
    # label 为空时只说做什么，不硬凑参数
    assert _tool_progress_hint("kb_list_files", "", 1) == "正在查看文件清单…"
    # 未收录的工具回退成通用动词，仍带 label
    assert _tool_progress_hint("weird_tool", "x", 1) == "正在处理「x」…"


def test_tool_start_is_followed_by_tool_progress():
    """前端渲染的是 tool_progress.message；tool_start 只给静态文案。"""
    from ai_agent.agents.loop import run_loop

    schemas = [{"function": {"name": "web_search"}}]
    calls = {"n": 0}

    def fake_stream(messages, tools, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            yield (
                "result",
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "web_search",
                            "arguments": {"query": "本周热点新闻"},
                        }
                    ],
                },
            )
            return
        yield ("text", "ok")
        yield ("result", {"content": "ok", "tool_calls": []})

    with (
        patch("ai_agent.agents.loop.tool_schemas", return_value=schemas),
        patch("ai_agent.agents.loop.stream_chat", side_effect=fake_stream),
        patch("ai_agent.agents.loop.execute_tool", return_value="搜索结果"),
        patch("ai_agent.kb.retrieval.mentioned_subjects", return_value=set()),
    ):
        events = list(run_loop([{"role": "user", "content": "一周热点新闻"}]))

    kinds = [e.get("type") for e in events]
    assert "tool_start" in kinds
    start = kinds.index("tool_start")
    # tool_progress 紧跟在 tool_start 之后，且带可读文案
    assert kinds[start + 1] == "tool_progress"
    progress = events[start + 1]
    assert progress["message"] == "正在搜索「本周热点新闻」…"
    assert progress["name"] == "web_search"


def test_tool_progress_absent_when_hint_empty():
    """hint 恒非空（最差也是「正在处理…」），保证前端一定有文案可显示。"""
    from ai_agent.agents._loop.tools_exec import _tool_progress_hint

    assert _tool_progress_hint("web_search", "", 1) == "正在搜索…"


# ——— kb_search 走线程通道并推重排进度 ———

def test_kb_search_is_a_progress_tool():
    """kb_search 冷态 9s、热态 2.7s，必须走线程通道才能推中途进度。"""
    from ai_agent.agents._loop.tools_exec import _PROGRESS_TOOLS

    assert "kb_search" in _PROGRESS_TOOLS


def test_rerank_emits_progress():
    """重排前推一条；未绑定回调时是空操作，不能抛异常。"""
    from unittest.mock import patch

    from ai_agent.kb.retrieval._impl import search as search_mod
    from ai_agent.tools.progress import reset_progress, set_progress

    rows = [{"file_id": "f", "seq": i} for i in range(12)]
    with patch.object(search_mod._public(), "_rerank", return_value=rows[:3]):
        # 未绑定：静默通过
        assert search_mod._rerank_with_progress("q", rows, 3) == rows[:3]

        captured: list[str] = []
        token = set_progress(captured.append)
        try:
            search_mod._rerank_with_progress("q", rows, 3)
        finally:
            reset_progress(token)
    assert captured == ["正在重排 12 条候选结果…"]


def test_kb_read_label_uses_location():
    """kb_read 的参数名是 location，不是 target/query。"""
    label = _tool_call_label("kb_read", {"location": "绿的谐波2026Q3 · 第2页"})
    assert label == "绿的谐波2026Q3 · 第2页"


def test_label_never_renders_python_literal():
    """dict 元素不能把 Python 字面量漏进进度文案。

    实测前端状态栏出现了
    「正在对比指标「{'subject': '中科飞测', 'name': '营业收入'…」…」
    ——report_compare 的 points 是 dict 列表，原实现直接 str() 了元素。
    """
    points = [
        {"subject": "中科飞测", "name": "营业收入", "period": "2026Q3",
         "value": "673,041,663.13"},
        {"subject": "中科飞测", "name": "营业收入", "period": "2025Q3",
         "value": "500,000,000"},
    ]
    label = _tool_call_label("report_compare", {"points": points})
    assert "{" not in label and "'" not in label
    assert label == "营业收入"

    hint = _tool_progress_hint("report_compare", label, 1)
    assert "{" not in hint
    assert hint == "正在对比指标「营业收入」…"


def test_label_dedupes_repeated_items():
    """同一指标的多个期次去重，不拼成「营业收入、营业收入」。"""
    points = [
        {"name": "营业收入", "period": "2026Q3"},
        {"name": "营业收入", "period": "2025Q3"},
        {"name": "净利润"},
    ]
    assert _tool_call_label("report_compare", {"points": points}) == "营业收入、净利润"


def test_label_without_readable_fields_falls_back():
    """dict 里一个可读字段都没有时，别把整段 dict 打出来。"""
    assert _tool_call_label("report_compare", {"points": [{"x": 1}]}) == "…"
    assert _tool_call_label("report_compare", {"points": []}) == ""
