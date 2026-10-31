"""用户意图识别。

把「这一句话主要想干什么」分成 6 类，给主模型一个显式的路由提示。

为什么要有这一层：原先知识库检索触发与否完全交给主模型按提示词自由裁量，
而早年用关键词/正则硬判天气、闲聊的做法（rewrite.py 里那张 _KB_OFFTOPIC /
_KB_CHITCHAT 表）已经在提交 bb47db6 移除了。现在补回一个**显式、可测、可降级**
的意图信号：

- 主路径：调一次小模型做单标签分类（快、temperature=0、max_tokens 极小）；
- 降级：模型调用失败 / 超时 / 解析不出标签时，退回关键词兜底（相当于把当年
  那张正则表的职责保留为安全网，而不是主路径）；
- 任何情况下都不抛异常、不挡住回答——分类只是「提示」，最终工具调用仍由主
  模型在 function calling 里决定。

分类标签直接对应项目已实现的能力：kb_search（KB_FACT）、kg_search
（KG_RELATION）、web_search（WEB）、天气工具（WEATHER）、闲聊（CHITCHAT）、
其余（GENERAL）。
"""

from __future__ import annotations

from enum import Enum

from ai_agent.core.config import settings
from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import complete_chat

logger = get_logger(__name__)


class Intent(str, Enum):
    """用户一句话的主要意图。值即分类 prompt 要求的输出标签。"""

    KB_FACT = "kb_fact"          # 问知识库文档里的具体事实（公司/产品/财报/数字/条款）
    KG_RELATION = "kg_relation"  # 关系型：和谁有关、上下游、投资、产业链位置
    WEB = "web"                  # 实时信息、新闻、最新版本、价格（需联网）
    WEATHER = "weather"          # 天气、气温、穿衣、带伞、出行
    CHITCHAT = "chitchat"        # 打招呼、感谢、闲聊、无明确任务
    GENERAL = "general"          # 其它通用问答，由主模型自行判断

    def __str__(self) -> str:  # 让日志/提示里直接出现标签而不是 "Intent.KB_FACT"
        return self.value


# ——— 分类 system prompt ———
#
# 要求只回一个标签、不要解释：分类调用 max_tokens 极小，多输出一个字都浪费。
INTENT_SYSTEM = """你是用户意图分类器。判断用户这一句话的主要意图，只输出下面 6 个标签之一，不要任何解释或标点：
kb_fact：问知识库文档里的具体事实（公司、产品、财报、数字、日期、条款）
kg_relation：问关系（某公司/技术/指标和谁有关、上下游、投资、产业链位置）
web：问实时信息、新闻、最新版本、价格（需要联网）
weather：问天气、气温、穿衣、要不要带伞、出行安排
chitchat：打招呼、感谢、闲聊、无明确任务
general：其它通用问答

判断要点：
1. 关系型问题优先标 kg_relation（即使也能查知识库）
2. 实时/新闻类优先标 web
3. 天气/出行类标 weather
4. 寒暄无任务标 chitchat
5. 拿不准标 general"""


# ——— 关键词兜底（降级安全网，原正则表职责的继承） ———
_WEATHER_KW = ("天气", "气温", "带伞", "下雨", "穿衣", "出行", "温度", "气象", "降温", "升温")
_CHITCHAT_KW = ("你好", "您好", "在吗", "在不在", "谢谢", "感谢", "拜拜", "早安", "晚安", "嗨", "哈喽")

# 中文同义词：模型偶尔会回中文标签或夹带中文，解析时兜底识别
_SYNONYMS: dict[Intent, tuple[str, ...]] = {
    Intent.WEATHER: ("天气", "气温", "带伞", "出行", "穿衣", "气象"),
    Intent.CHITCHAT: ("闲聊", "聊聊", "寒暄", "打招呼", "在吗", "谢谢", "你好"),
    Intent.WEB: ("新闻", "实时", "联网", "最新", "价格"),
    Intent.KG_RELATION: ("关系", "上下游", "产业链", "有关", "投资"),
    Intent.KB_FACT: ("财报", "知识库", "文档", "招股"),
}


def _keyword_fallback(query: str) -> Intent:
    """模型不可用时的兜底：复刻早年正则表的行为，只识别天气/闲聊两类。"""
    q = (query or "").strip()
    if any(k in q for k in _WEATHER_KW):
        return Intent.WEATHER
    if any(k in q for k in _CHITCHAT_KW):
        return Intent.CHITCHAT
    return Intent.GENERAL


def _parse_label(raw: str | None) -> Intent | None:
    """从模型输出里抠出意图标签，容忍大小写、前后空白、夹带中文。"""
    if not raw:
        return None
    s = raw.strip().lower()
    # 先按枚举值（英文标签）直接命中
    for intent in Intent:
        if intent.value in s:
            return intent
    # 再按中文同义词兜底
    for intent, words in _SYNONYMS.items():
        if any(w in s for w in words):
            return intent
    return None


def classify_intent(query: str) -> Intent | None:
    """分类用户意图。

    返回 None 表示「未启用 / 无法判定」——调用方应把它当作「不注入任何路由提示」，
    退回主模型自行决定的旧行为。任何异常都不会外泄：模型失败一律退回关键词兜底。
    """
    if not settings.intent_enabled:
        return None
    model = settings.resolve_llm(settings.intent_model)
    try:
        raw = complete_chat(
            [
                {"role": "system", "content": INTENT_SYSTEM},
                {"role": "user", "content": (query or "").strip()},
            ],
            model=model,
            timeout=settings.intent_timeout_seconds,
            max_retries=0,
            max_tokens=8,
            temperature=settings.intent_temperature,
        )
    except LLMError as e:
        logger.info("意图分类模型调用失败，退回关键词兜底：%s", e)
        return _keyword_fallback(query)
    except Exception:
        logger.exception("意图分类异常，退回关键词兜底")
        return _keyword_fallback(query)
    parsed = _parse_label(raw)
    if parsed is None:
        return _keyword_fallback(query)
    return parsed


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
