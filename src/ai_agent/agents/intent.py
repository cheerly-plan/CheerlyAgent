"""用户意图识别（关键词兜底版）。

把「这一句话是否明显与知识库无关」用**纯关键词**判定，给主模型一个显式但软性的路由提示。

为什么不用小模型（此前方案已移除）：
- 分类只在「天气 / 闲聊」两类上有运行时行为（其余标签不产生任何效果），而这两类
  关键词就能确定性识别，没必要每轮多花一次 LLM 调用 + 几十毫秒延迟；
- 小模型抽风误判，反而可能在 false-positive 时让主模型跳过本该查的库。

知识库开关（use_kb）与意图完全解耦：本模块**永远不关闭检索工具**，只追加软提示，
最终工具调用仍由主模型在 function calling 里决定。早期散落在 rewrite/retrieval 里的
关键词/正则启发式（_KB_OFFTOPIC / _KB_CHITCHAT，已在 bb47db6 删除）以关键词形式收编回来。
"""

from __future__ import annotations

from enum import Enum

from ai_agent.core.config import settings


class Intent(str, Enum):
    """用户一句话的主要意图。值即分类标签。"""

    WEATHER = "weather"      # 天气、气温、穿衣、带伞、出行
    CHITCHAT = "chitchat"    # 打招呼、感谢、闲聊、无明确任务
    GENERAL = "general"      # 其它通用问答，由主模型自行判断

    def __str__(self) -> str:  # 让日志/提示里直接出现标签而不是 "Intent.WEATHER"
        return self.value


_WEATHER_KW = ("天气", "气温", "带伞", "下雨", "穿衣", "出行", "温度", "气象", "降温", "升温")
_CHITCHAT_KW = ("你好", "您好", "在吗", "在不在", "谢谢", "感谢", "拜拜", "早安", "晚安", "嗨", "哈喽")


def _keyword_fallback(query: str) -> Intent:
    """纯关键词判定：只识别天气 / 闲聊两类，其余一律 GENERAL。

    确定性、零延迟、永不抛异常——复刻早年正则表的安全网职责。
    """
    q = (query or "").strip()
    if any(k in q for k in _WEATHER_KW):
        return Intent.WEATHER
    if any(k in q for k in _CHITCHAT_KW):
        return Intent.CHITCHAT
    return Intent.GENERAL


def classify_intent(query: str) -> Intent | None:
    """用关键词判定意图。

    返回 None 表示「未启用 / 无法判定」——调用方应把它当作「不注入任何路由提示」，
    退回主模型自行决定的旧行为。纯字符串匹配，绝不发起任何 LLM 调用。
    """
    if not settings.intent_enabled:
        return None
    return _keyword_fallback(query)


def intent_hint(intent: Intent | None) -> str:
    """把意图翻成塞进 system prompt 的显式路由提示。

    只对「明确与知识库无关」的两类给强提示；其余（含 None）返回空串，
    不改动主模型既有的自由裁量——避免过指令化把该查库的句子也误伤。
    """
    if intent is Intent.WEATHER:
        return (
            "【意图提示】用户意图为天气/出行查询，与知识库无关。"
            "请勿调用 kb_search/kg_search，改用天气工具或直接回答。"
        )
    if intent is Intent.CHITCHAT:
        return (
            "【意图提示】用户为寒暄/闲聊，与知识库无关。"
            "请勿调用 kb_search/kg_search，直接自然回应。"
        )
    return ""
