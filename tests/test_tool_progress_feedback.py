"""工具调用的可见性与防打转。

真实事故：用户问「一周热点新闻」，模型连调 13 次工具，其中同一份行情数据
换了 6 种关键词反复搜（那天是周六、没有行情），前端全程只有「正在处理…」。
"""

from unittest.mock import patch

from ai_agent.agents._loop.tools_exec import (
    _tool_call_index_factory,
    _tool_call_label,
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
