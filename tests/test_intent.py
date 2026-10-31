"""意图识别模块的测试。

覆盖：关键词判定、开关关闭退回 None、路由提示文案、与 loop._system_prompt 的集成。
（此前的小模型分类主路径已移除——天气/闲聊用确定性关键词识别，不发起任何 LLM 调用。）
"""

from ai_agent.agents import loop as loop_mod
from ai_agent.agents.intent import (
    Intent,
    _keyword_fallback,
    classify_intent,
    intent_hint,
)
from ai_agent.core.config import settings


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


def test_classify_weather_keyword(monkeypatch):
    monkeypatch.setattr(settings, "intent_enabled", True)
    assert classify_intent("深圳今天气温多少") is Intent.WEATHER
    assert classify_intent("明天会下雨吗") is Intent.WEATHER


def test_classify_chitchat_keyword(monkeypatch):
    monkeypatch.setattr(settings, "intent_enabled", True)
    assert classify_intent("你好，在吗") is Intent.CHITCHAT
    assert classify_intent("谢谢你") is Intent.CHITCHAT


def test_classify_general_no_llm_call(monkeypatch):
    # 纯关键词判定，绝不发起 LLM 调用；无关键词命中即退回 GENERAL。
    # 此处不打 complete_chat 的桩——若实现误发起调用会直接报错暴露回归。
    monkeypatch.setattr(settings, "intent_enabled", True)
    assert classify_intent("解释一下 TCP 握手") is Intent.GENERAL
    assert classify_intent("讲讲量子计算") is Intent.GENERAL


# ——— intent_hint ———


def test_intent_hint_offtopic_nonempty():
    hint = intent_hint(Intent.WEATHER)
    assert "天气" in hint and "kb_search" in hint
    hint2 = intent_hint(Intent.CHITCHAT)
    assert "寒暄" in hint2 and "kb_search" in hint2


def test_intent_hint_others_empty():
    # 仅天气/闲聊给强提示，其余（含 None）空串，不干扰主模型自由裁量
    assert intent_hint(Intent.GENERAL) == ""
    assert intent_hint(None) == ""


# ——— loop 集成 ———


def test_system_prompt_includes_hint_for_offtopic():
    prompt = loop_mod._system_prompt("u", use_kb=True, intent=Intent.WEATHER)
    assert "【意图提示】" in prompt
    assert "天气" in prompt


def test_system_prompt_no_hint_for_general():
    # 非天气/闲聊意图不追加"别查知识库"提示，保留主模型自由裁量
    prompt = loop_mod._system_prompt("u", use_kb=True, intent=Intent.GENERAL)
    assert "【意图提示】" not in prompt


def test_system_prompt_no_hint_when_kb_disabled():
    # 没有知识库时，即便判定为天气也不应追加"别查知识库"的冗余提示
    prompt = loop_mod._system_prompt("u", use_kb=False, intent=Intent.WEATHER)
    assert "【意图提示】" not in prompt


def test_system_prompt_no_hint_when_intent_none():
    # 未启用/无法判定：与重构前逐字一致（只差画像块，由 _profile_prompt 决定）
    prompt = loop_mod._system_prompt("u", use_kb=True, intent=None)
    assert "【意图提示】" not in prompt


# ——— 知识库可用文件清单注入（去掉隐式预检索后，模型对照清单显式圈定 files）———


def test_kb_files_block_injects_ready_files(monkeypatch):
    """可用文件清单注入系统提示词，且只列 ready 状态的文件。"""
    monkeypatch.setattr(
        loop_mod.store,
        "list_kb_files",
        lambda **k: [
            {
                "file_id": "a",
                "filename": "中国能建三季报.pdf",
                "status": "ready",
                "description": "能源基建",
            },
            {
                "file_id": "b",
                "filename": "安克创新三季报.pdf",
                "status": "processing",
                "description": "",
            },
        ],
    )
    block = loop_mod._kb_files_block()
    assert "【知识库可用文件】" in block
    assert "中国能建三季报.pdf" in block
    assert "能源基建" in block
    # 还在处理的文件不进清单
    assert "安克创新三季报.pdf" not in block


def test_kb_files_block_filters_to_session_scope(monkeypatch):
    """会话限定文件范围时，清单只列那些文件。"""
    monkeypatch.setattr(
        loop_mod.store,
        "list_kb_files",
        lambda **k: [
            {"file_id": "a", "filename": "A.pdf", "status": "ready", "description": ""},
            {"file_id": "b", "filename": "B.pdf", "status": "ready", "description": ""},
        ],
    )
    block = loop_mod._kb_files_block(("b",))
    assert "B.pdf" in block
    assert "A.pdf" not in block


def test_kb_files_block_empty_when_no_ready(monkeypatch):
    monkeypatch.setattr(loop_mod.store, "list_kb_files", lambda **k: [])
    assert loop_mod._kb_files_block() == ""


def test_system_prompt_includes_kb_files_block(monkeypatch):
    """use_kb 时系统提示词注入可用文件清单；关闭知识库时不注入。"""
    monkeypatch.setattr(
        loop_mod.store,
        "list_kb_files",
        lambda **k: [
            {"file_id": "a", "filename": "财报.pdf", "status": "ready", "description": ""}
        ],
    )
    with_kb = loop_mod._system_prompt("u", use_kb=True)
    assert "财报.pdf" in with_kb
    without_kb = loop_mod._system_prompt("u", use_kb=False)
    assert "财报.pdf" not in without_kb
