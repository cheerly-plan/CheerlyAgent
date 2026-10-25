from unittest.mock import patch

from ai_agent.core.exceptions import LLMError
from ai_agent.memory.summarize import summarize_session

_OVERFLOW = [
    {"role": "user", "content": "想去杭州"},
    {"role": "assistant", "content": "可以走周末两日"},
    {"role": "user", "content": "顺带查下深圳天气"},
    {"role": "assistant", "content": "深圳今天晴。"},
]
# 预过滤（turn_* key）：两轮都保留
_ALL_KEEP = {"turn_0": 0.9, "turn_1": 0.9}
_CHECK = "ai_agent.memory.summarize.jev_noul_batch"
_COMPLETE = "ai_agent.memory.summarize.complete_chat"


def test_summarize_session_check_passes():
    """双 Noul 均低于阈值：摘要落库，且校验输入含完整 ground truth。"""
    prompts: list[str] = []

    def fake_complete(messages, **kwargs):
        prompts.append(messages[1]["content"])
        return "用户在规划杭州旅行"

    with (
        patch(_COMPLETE, side_effect=fake_complete),
        patch(
            _CHECK,
            side_effect=[
                _ALL_KEEP,
                {"fabrication": 0.05, "omission": 0.1},
            ],
        ) as check,
    ):
        text = summarize_session("用户叫小明", _OVERFLOW)
    assert text == "用户在规划杭州旅行"
    # 第一次调用 = 逐轮预过滤，第二次 = 质量校验
    assert set(check.call_args_list[0].args[0]) == {"turn_0", "turn_1"}
    questions, state = check.call_args_list[1].args
    assert set(questions) == {"fabrication", "omission"}
    # ground truth 必须是旧摘要 + 溢出对话的并集
    assert "用户叫小明" in state
    assert "想去杭州" in state
    assert "待检摘要" in state
    # 精摘输入仍是「旧摘要 + 新增对话」
    assert "已有摘要" in prompts[0]
    assert "用户叫小明" in prompts[0]


def test_summarize_session_skips_check_when_unavailable():
    """校验不可用（未配 key / Jev 失败）：fail-open 直接接受。"""
    with (
        patch(_COMPLETE, return_value="摘要文本") as complete,
        patch(_CHECK, return_value=None),
    ):
        text = summarize_session("", _OVERFLOW)
    assert text == "摘要文本"
    complete.assert_called_once()


def test_summarize_session_filters_low_value_turns():
    """天气/寒暄轮次被预过滤剔除，不进入精摘输入。"""
    overflow = [
        {"role": "user", "content": "你好呀"},
        {"role": "assistant", "content": "你好！有什么可以帮你？"},
        {"role": "user", "content": "我在做 Node.js BFF 聚合层"},
        {"role": "assistant", "content": "好的，先定路由。"},
    ]
    prompts: list[str] = []

    def fake_complete(messages, **kwargs):
        prompts.append(messages[1]["content"])
        return "用户在做 BFF 聚合层"

    with (
        patch(_COMPLETE, side_effect=fake_complete),
        patch(
            _CHECK,
            side_effect=[
                {"turn_0": 0.05, "turn_1": 0.9},
                {"fabrication": 0.05, "omission": 0.1},
            ],
        ),
    ):
        text = summarize_session("", overflow)
    assert text == "用户在做 BFF 聚合层"
    assert "BFF" in prompts[0]
    assert "你好呀" not in prompts[0]


def test_summarize_session_all_turns_filtered_keeps_previous():
    """全部轮次被过滤：旧摘要原样保留，不调精摘。"""
    with (
        patch(_COMPLETE) as complete,
        patch(
            _CHECK,
            return_value={"turn_0": 0.05, "turn_1": 0.05},
        ),
    ):
        text = summarize_session("旧摘要", _OVERFLOW)
    assert text == "旧摘要"
    complete.assert_not_called()


def test_summarize_session_single_turn_skips_pre_filter():
    """只有一轮时过滤无意义（整段保留或整段丢），跳过预过滤。"""
    with (
        patch(_COMPLETE, return_value="摘要文本"),
        patch(
            _CHECK,
            return_value={"fabrication": 0.05, "omission": 0.1},
        ) as check,
    ):
        text = summarize_session("", [{"role": "user", "content": "想去杭州"}])
    assert text == "摘要文本"
    assert set(check.call_args_list[0].args[0]) == {"fabrication", "omission"}


def test_summarize_session_retries_on_fabrication_then_passes():
    """首版疑似编造：带落选原因重试，重试版合格则采用。"""
    prompts: list[str] = []

    def fake_complete(messages, **kwargs):
        prompts.append(messages[1]["content"])
        return ["编造的摘要", "修正后的摘要"][len(prompts) - 1]

    with (
        patch(_COMPLETE, side_effect=fake_complete),
        patch(
            _CHECK,
            side_effect=[
                _ALL_KEEP,
                {"fabrication": 0.9, "omission": 0.1},
                {"fabrication": 0.05, "omission": 0.1},
            ],
        ),
    ):
        text = summarize_session("", _OVERFLOW)
    assert text == "修正后的摘要"
    assert "编造" in prompts[1]
    assert "严格基于对话原文" in prompts[1]


def test_summarize_session_falls_back_when_retry_still_fabricated():
    """连续两版都疑似编造：宁可保真截断也不能带错信息进上下文。"""
    with (
        patch(_COMPLETE, side_effect=["编造一", "编造二"]),
        patch(
            _CHECK,
            side_effect=[
                _ALL_KEEP,
                {"fabrication": 0.95, "omission": 0.1},
                {"fabrication": 0.95, "omission": 0.1},
            ],
        ),
    ):
        text = summarize_session("旧摘要", _OVERFLOW)
    assert "旧摘要" in text
    assert "想去杭州" in text  # crude 摘录包含原文


def test_summarize_session_accepts_retry_on_omission_only():
    """重试后仅遗漏：遗漏可容忍，接受重试版而不是降级 crude。"""
    with (
        patch(_COMPLETE, side_effect=["第一版", "第二版"]),
        patch(
            _CHECK,
            side_effect=[
                _ALL_KEEP,
                {"fabrication": 0.1, "omission": 0.9},
                {"fabrication": 0.1, "omission": 0.9},
            ],
        ),
    ):
        text = summarize_session("", _OVERFLOW)
    assert text == "第二版"


def test_summarize_session_falls_back_on_llm_error():
    with (
        patch(_COMPLETE, side_effect=LLMError("timeout")),
        patch(_CHECK, return_value=_ALL_KEEP),
    ):
        text = summarize_session(
            "旧摘要",
            [{"role": "user", "content": "我叫小明"}, {"role": "assistant", "content": "你好"}],
        )
    assert "旧摘要" in text
    assert "我叫小明" in text
