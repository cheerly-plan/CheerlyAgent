"""重新生成要沿用「上一轮生成时」的研报开关。

前端把「研报」做成一次性标签：发送后立刻重置，所以重新生成时传上来的
永远是 false。开关值必须在生成那一轮随回答落进 meta，重新生成时读回。
"""

from unittest.mock import patch

from ai_agent.agents._loop.loop import _tag_switches_on_turn


def test_tags_only_final_assistant_message():
    turn = [
        {"role": "user", "content": "生成研报"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "工具结果"},
        {"role": "assistant", "content": "研报正文"},
    ]
    _tag_switches_on_turn(turn, use_report=True)
    assert turn[0].get("meta") is None
    assert turn[1].get("meta") is None
    assert turn[3]["meta"] == {"use_report": True}


def test_tag_preserves_existing_meta():
    turn = [{"role": "assistant", "content": "x", "meta": {"report_job_id": "j1"}}]
    _tag_switches_on_turn(turn, use_report=False)
    assert turn[0]["meta"] == {"report_job_id": "j1", "use_report": False}


def test_reader_returns_flag_of_last_turn():
    from ai_agent.api.routers import chat as chat_router

    turns = [
        {"role": "user", "content": "生成研报"},
        {"role": "assistant", "content": "旧回答", "meta": {"use_report": True}},
        {"role": "user", "content": "重新生成"},
        {"role": "assistant", "content": "新回答", "meta": {"use_report": True}},
    ]
    with patch.object(chat_router.store, "load_all_turns", return_value=turns):
        assert chat_router._report_flag_of_last_turn("u", "s") is True


def test_reader_ignores_previous_turn_flag():
    """只看最后一条用户消息之后的回答，别把上一轮的开关带进这一轮。"""
    from ai_agent.api.routers import chat as chat_router

    turns = [
        {"role": "user", "content": "生成研报"},
        {"role": "assistant", "content": "旧回答", "meta": {"use_report": True}},
        {"role": "user", "content": "你好"},
    ]
    with patch.object(chat_router.store, "load_all_turns", return_value=turns):
        assert chat_router._report_flag_of_last_turn("u", "s") is False


def test_reader_falls_back_when_meta_missing():
    """老数据没有 meta：退回前端传值，不改变既有行为。"""
    from ai_agent.api.routers import chat as chat_router

    turns = [
        {"role": "user", "content": "生成研报"},
        {"role": "assistant", "content": "老回答"},
    ]
    with patch.object(chat_router.store, "load_all_turns", return_value=turns):
        assert chat_router._report_flag_of_last_turn("u", "s", fallback=True) is True
        assert chat_router._report_flag_of_last_turn("u", "s", fallback=False) is False


def test_reader_survives_load_failure():
    from ai_agent.api.routers import chat as chat_router

    with patch.object(
        chat_router.store, "load_all_turns", side_effect=RuntimeError("db down")
    ):
        assert chat_router._report_flag_of_last_turn("u", "s", fallback=True) is True
