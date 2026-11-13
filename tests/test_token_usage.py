"""token 采集与展示：stream_chat 抓 usage、落库、管理面板聚合。

主对话走流式，网关的 usage 只在最后一帧返回，且那一帧 choices 为空。
此前代码在 `if not chunk.choices: continue` 里把它整帧跳过了，全链路
拿不到 token 数。
"""

from types import SimpleNamespace

import pytest

from ai_agent.api.inspect_turns import sum_tokens, turns_from_messages
from ai_agent.llm import client as llm_client


def _chunk(content=None, *, usage=None, tool_calls=None):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=None,
        reasoning=None,
        tool_calls=tool_calls or [],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, message=None)],
        usage=usage,
    )


def _usage_frame(prompt: int, completion: int):
    """网关的 usage 帧：choices 为空，只有 usage。"""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


@pytest.fixture
def fake_stream_client(monkeypatch):
    """把 _stream_client 换成返回固定 chunks 的假客户端，并记录请求参数。"""

    def _install(chunks):
        seen: dict = {}

        def _create(**kwargs):
            seen.update(kwargs)
            return _FakeStream(chunks)

        fake = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_create)
            )
        )
        monkeypatch.setattr(llm_client, "_stream_client", fake)
        return seen

    return _install


def test_usage_is_captured_from_trailing_frame(fake_stream_client):
    seen = fake_stream_client(
        [_chunk("你好"), _usage_frame(1200, 30)]
    )

    items = list(llm_client.stream_chat([], []))
    result = items[-1][1]

    assert result["content"] == "你好"
    assert result["prompt_tokens"] == 1200
    assert result["completion_tokens"] == 30


def test_stream_requests_include_usage():
    """没这个字段网关不会回 usage，后续一切统计都是空。"""
    captured = {}

    def _create(**kwargs):
        captured.update(kwargs)
        return _FakeStream([])

    import ai_agent.llm.client as mod

    original = mod._stream_client
    mod._stream_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    try:
        list(mod.stream_chat([], []))
    finally:
        mod._stream_client = original

    # stream_options 走 extra_body：OpenAI SDK 的顶层参数不接受它
    assert captured["extra_body"]["stream_options"] == {"include_usage": True}
    # 关思考链的原有字段不能被覆盖掉
    assert captured["extra_body"]["enable_thinking"] is False


def test_gateway_without_usage_yields_none(fake_stream_client):
    """不支持 usage 的网关：字段为 None，不能伪造 0。"""
    fake_stream_client([_chunk("正文")])

    result = list(llm_client.stream_chat([], []))[-1][1]

    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None


def test_sum_tokens_ignores_missing():
    assert sum_tokens([1, None, 2]) == 3
    assert sum_tokens([None, None]) is None
    assert sum_tokens([]) is None


def test_turns_carry_token_totals():
    messages = [
        {"role": "user", "content": "汇川技术利润怎么样"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "name": "kb_search", "arguments": {}}
        ], "prompt_tokens": 5000, "completion_tokens": 40},
        {"role": "tool", "content": "片段", "tool_call_id": "c1", "name": "kb_search"},
        {"role": "assistant", "content": "回答", "prompt_tokens": 6000,
         "completion_tokens": 300},
    ]

    turns = turns_from_messages(messages)

    assert len(turns) == 1
    assert turns[0].prompt_tokens == 11000
    assert turns[0].completion_tokens == 340
    llm_steps = [s for s in turns[0].steps if s.kind == "llm"]
    assert llm_steps[-1].prompt_tokens == 6000
    assert llm_steps[-1].completion_tokens == 300


def test_legacy_messages_show_no_tokens():
    """加列前的历史消息没有 token：接口给 None，前端显示"—"。"""
    messages = [
        {"role": "user", "content": "老对话"},
        {"role": "assistant", "content": "老回答"},
    ]

    turns = turns_from_messages(messages)

    assert turns[0].prompt_tokens is None
    assert turns[0].completion_tokens is None
    assert turns[0].steps[0].prompt_tokens is None

def test_turn_input_tokens_counts_whole_turn():
    """「当次输入」= 本轮新增内容，工具结果占大头也要算进去。"""
    tool_text = "检索片段" * 200
    messages = [
        {"role": "user", "content": "现金流情况怎么样"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "name": "kb_search", "arguments": {"q": "现金流"}}
        ], "prompt_tokens": 28552, "completion_tokens": 46},
        {"role": "tool", "content": tool_text, "tool_call_id": "c1", "name": "kb_search"},
        {"role": "assistant", "content": "经营活动现金流净额增长。",
         "prompt_tokens": 30151, "completion_tokens": 879},
    ]

    turns = turns_from_messages(messages)

    turn = turns[0]
    # 远小于网关 prompt 之和（历史被重复计费），但远大于提问本身
    assert turn.input_tokens < 28552 + 30151
    assert turn.input_tokens > len(tool_text) * 0.5
    # 输出仍是网关口径
    assert turn.completion_tokens == 925


def test_turn_input_tokens_includes_question_alone():
    messages = [{"role": "user", "content": "你好"}]

    turns = turns_from_messages(messages)

    assert turns[0].input_tokens > 0
    assert turns[0].completion_tokens is None


def test_input_tokens_resets_between_turns():
    messages = [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "第二问"},
        {"role": "assistant", "content": "第二答"},
    ]

    turns = turns_from_messages(messages)

    assert len(turns) == 2
    # 第二轮只算自己那几条消息：不能把第一轮的量带过来。
    # 直接拿单条消息的基准比较，避免写出恒真的断言。
    from ai_agent.api.inspect_turns import input_tokens as _msg_tokens

    expected_first = _msg_tokens(messages[0]) + _msg_tokens(messages[1])
    expected_second = _msg_tokens(messages[2]) + _msg_tokens(messages[3])
    assert turns[0].input_tokens == expected_first
    assert turns[1].input_tokens == expected_second
    assert turns[1].input_tokens < expected_first + expected_second
