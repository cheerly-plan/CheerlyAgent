"""意图识别模块的测试。

覆盖：标签解析、关键词兜底、模型主路径（含同义词）、模型失败/失效退回兜底、
路由提示文案、以及与 loop._system_prompt 的集成。
"""

import pytest

from ai_agent.agents import loop as loop_mod
from ai_agent.agents.intent import (
    Intent,
    _keyword_fallback,
    _parse_label,
    classify_intent,
    intent_hint,
)
from ai_agent.core.config import settings


# ——— _parse_label ———


def test_parse_label_exact_value():
    assert _parse_label("kb_fact") is Intent.KB_FACT
    assert _parse_label("kg_relation") is Intent.KG_RELATION
    assert _parse_label("web") is Intent.WEB
    assert _parse_label("weather") is Intent.WEATHER
    assert _parse_label("chitchat") is Intent.CHITCHAT
    assert _parse_label("general") is Intent.GENERAL


def test_parse_label_tolerates_noise():
    assert _parse_label("意图：weather") is Intent.WEATHER
    assert _parse_label("WEB\n") is Intent.WEB
    assert _parse_label("Kb_Fact") is Intent.KB_FACT  # 大小写不敏感


def test_parse_label_chinese_synonym():
    assert _parse_label("查下今天的天气") is Intent.WEATHER
    assert _parse_label("帮我看看上下游产业链") is Intent.KG_RELATION
    assert _parse_label("最新新闻有哪些") is Intent.WEB
    assert _parse_label("随便聊聊") is Intent.CHITCHAT


def test_parse_label_garbage_returns_none():
    assert _parse_label("") is None
    assert _parse_label(None) is None
    assert _parse_label("这是一段无关的回答文本，没有任何标签") is None


# ——— _keyword_fallback ———


def test_keyword_fallback_weather():
    assert _keyword_fallback("今天深圳天气怎么样") is Intent.WEATHER
    assert _keyword_fallback("要不要带伞") is Intent.WEATHER


def test_keyword_fallback_chitchat():
    assert _keyword_fallback("你好，在吗") is Intent.CHITCHAT
    assert _keyword_fallback("谢谢你") is Intent.CHITCHAT


def test_keyword_fallback_general():
    assert _keyword_fallback("帮我查下公司去年营收") is Intent.GENERAL
    assert _keyword_fallback("这个接口怎么实现") is Intent.GENERAL


# ——— classify_intent ———


def test_classify_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "intent_enabled", False)
    assert classify_intent("今天天气怎么样") is None


def test_classify_model_path(monkeypatch):
    monkeypatch.setattr(settings, "intent_enabled", True)

    def fake(messages, **kwargs):
        return "weather"

    monkeypatch.setattr("ai_agent.agents.intent.complete_chat", fake)
    assert classify_intent("深圳今天气温多少") is Intent.WEATHER

    def fake2(messages, **kwargs):
        return "kg_relation"

    monkeypatch.setattr("ai_agent.agents.intent.complete_chat", fake2)
    assert classify_intent("它和哪些公司有关") is Intent.KG_RELATION


def test_classify_model_raises_falls_back_to_keyword(monkeypatch):
    monkeypatch.setattr(settings, "intent_enabled", True)

    def boom(messages, **kwargs):
        raise RuntimeError("gateway down")

    monkeypatch.setattr("ai_agent.agents.intent.complete_chat", boom)
    # 含天气关键词 → 兜底命中
    assert classify_intent("明天会下雨吗") is Intent.WEATHER
    # 无关键词 → 兜底 GENERAL，绝不抛异常
    assert classify_intent("解释一下 TCP 握手") is Intent.GENERAL


def test_classify_model_junk_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "intent_enabled", True)

    def fake(messages, **kwargs):
        return "模型跑题了，给了无关文本"

    monkeypatch.setattr("ai_agent.agents.intent.complete_chat", fake)
    # 解析不出标签 → 关键词兜底：这句含天气词则命中
    assert classify_intent("今天天气真好") is Intent.WEATHER
    assert classify_intent("讲讲量子计算") is Intent.GENERAL


# ——— intent_hint ———


def test_intent_hint_offtopic_nonempty():
    hint = intent_hint(Intent.WEATHER)
    assert "天气" in hint and "kb_search" in hint
    hint2 = intent_hint(Intent.CHITCHAT)
    assert "寒暄" in hint2 and "kb_search" in hint2


def test_intent_hint_others_empty():
    for intent in (Intent.KB_FACT, Intent.KG_RELATION, Intent.WEB, Intent.GENERAL):
        assert intent_hint(intent) == ""
    assert intent_hint(None) == ""


# ——— loop 集成 ———


def test_system_prompt_includes_hint_for_offtopic():
    prompt = loop_mod._system_prompt("u", use_kb=True, intent=Intent.WEATHER)
    assert "【意图提示】" in prompt
    assert "天气" in prompt


def test_system_prompt_no_hint_for_kb_intent():
    prompt = loop_mod._system_prompt("u", use_kb=True, intent=Intent.KB_FACT)
    assert "【意图提示】" not in prompt


def test_system_prompt_no_hint_when_kb_disabled():
    # 没有知识库时，即便判定为天气也不应追加"别查知识库"的冗余提示
    prompt = loop_mod._system_prompt("u", use_kb=False, intent=Intent.WEATHER)
    assert "【意图提示】" not in prompt


def test_system_prompt_no_hint_when_intent_none():
    # 未启用/无法判定：与重构前逐字一致（只差画像块，由 _profile_prompt 决定）
    prompt = loop_mod._system_prompt("u", use_kb=True, intent=None)
    assert "【意图提示】" not in prompt
