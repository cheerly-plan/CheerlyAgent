"""强制工具选择：问业绩时走确定性抽取，而不是 kb_search。"""

from unittest.mock import patch

from ai_agent.agents._loop.followup import _forced_kb_tools, _METRIC_TOOL
from ai_agent.tools.registry import tool_schemas


def _tools(*names: str) -> list[dict]:
    keep = set(names)
    return [s for s in tool_schemas() if s["function"]["name"] in keep]


ALL = _tools(
    "kb_search", "kg_search", "kb_read", _METRIC_TOOL, "report_compare", "web_search"
)


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def _fake_subjects(matched: bool):
    return lambda text, file_ids=None: {"绿的谐波"} if matched else set()


def test_metric_question_forces_extract_tool():
    with patch("ai_agent.kb.retrieval.mentioned_subjects", _fake_subjects(True)), \
         patch("ai_agent.kb.retrieval.mentioned_entities", return_value=set()):
        forced = _forced_kb_tools(_msgs("绿的谐波的业绩怎么样"), ALL)
    assert forced == frozenset({_METRIC_TOOL})


def test_non_metric_question_keeps_retrieval_tools():
    with patch("ai_agent.kb.retrieval.mentioned_subjects", _fake_subjects(True)), \
         patch("ai_agent.kb.retrieval.mentioned_entities", return_value=set()):
        forced = _forced_kb_tools(_msgs("绿的谐波的公司简介"), ALL)
    assert forced == frozenset({"kb_search", "kg_search", "kb_read"})


def test_subject_not_in_kb_does_not_force():
    with patch("ai_agent.kb.retrieval.mentioned_subjects", _fake_subjects(False)), \
         patch("ai_agent.kb.retrieval.mentioned_entities", return_value=set()):
        assert _forced_kb_tools(_msgs("某某公司的业绩怎么样"), ALL) is None


def test_metric_falls_back_to_retrieval_when_tool_missing():
    """报表演示工具不可用时仍强制检索，不至于完全不查。"""
    tools = _tools("kb_search", "kg_search", "kb_read")
    with patch("ai_agent.kb.retrieval.mentioned_subjects", _fake_subjects(True)), \
         patch("ai_agent.kb.retrieval.mentioned_entities", return_value=set()):
        forced = _forced_kb_tools(_msgs("绿的谐波的业绩怎么样"), tools)
    assert forced == frozenset({"kb_search", "kg_search", "kb_read"})


def test_already_retrieved_does_not_force():
    messages = [
        {"role": "user", "content": "绿的谐波业绩怎么样"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": _METRIC_TOOL, "arguments": {}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": _METRIC_TOOL, "content": "..."},
    ]
    assert _forced_kb_tools(messages, ALL) is None
