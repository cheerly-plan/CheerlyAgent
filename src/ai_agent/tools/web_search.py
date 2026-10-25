"""联网搜索工具。

设计约束：
- 零第三方依赖：三个搜索源都是一次 REST 调用，标准库 urllib 足够；
- 级联策略：按配置的 key 组成候选链（tavily > bocha > serper），
  逐个尝试——失败或空结果自动落到下一个源，第一个有结果的胜出；
  都未配置时返回引导文案（不报错——工具层异常会被 execute_tool 吞成
  通用 ERROR，模型拿不到原因）；
- 时间敏感路由：模型传 freshness（day/week/month/year）表示需要新近内容时，
  候选链重排为 serper > tavily > bocha——博查索引收录滞后、Tavily 中文
  覆盖弱，实时性要求高时 Google（serper）的新鲜度最可靠；
- 输出是给模型读的紧凑文本：序号 + 标题 + URL + 摘要，总长由
  execute_tool 统一截断到 3000 字符。
"""

import json
import urllib.request

from pydantic import BaseModel, Field

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.tools.base import Tool

logger = get_logger(__name__)

_NO_KEY_HINT = (
    "ERROR: 联网搜索未配置。请在服务端 .env 里配置任一搜索 API key 后重启：\n"
    "TAVILY_API_KEY（免费 1000 次/月，https://tavily.com）\n"
    "BOCHA_API_KEY（国内直连，https://bocha-ai.com）\n"
    "SERPER_API_KEY（Google 结果，免费 2500 次，https://serper.dev）"
)

# 模型侧 freshness 取值 → 各 provider 的时间参数
_TAVILY_DAYS = {"day": 1, "week": 7, "month": 30}
_SERPER_TBS = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}
_BOCHA_FRESHNESS = {
    "day": "oneDay",
    "week": "oneWeek",
    "month": "oneMonth",
    "year": "oneYear",
}


class WebSearchArgs(BaseModel):
    query: str = Field(description="搜索关键词，尽量具体，可用引号精确匹配短语")
    max_results: int = Field(
        default=5, ge=1, le=10, description="返回结果条数，默认 5"
    )
    freshness: str = Field(
        default="",
        description=(
            "时间范围过滤：day（24小时内）/ week / month / year。"
            "查新闻、刚发生的事件、最新版本时必传 day 或 week；"
            "一般性查询留空。"
        ),
    )


def _http_json(url: str, body: dict, headers: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(
        req, timeout=settings.web_search_timeout_seconds
    ) as resp:
        return json.loads(resp.read().decode())


def _tavily_search(query: str, max_results: int, freshness: str) -> list[dict]:
    body: dict = {
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    days = _TAVILY_DAYS.get(freshness)
    if days:
        # Tavily 的新近内容走 news 主题 + 天数窗口
        body.update({"topic": "news", "days": days})
    data = _http_json(
        "https://api.tavily.com/search",
        body,
        {"Authorization": f"Bearer {settings.tavily_api_key}"},
    )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in data.get("results", [])
    ]


def _bocha_search(query: str, max_results: int, freshness: str) -> list[dict]:
    body: dict = {"query": query, "count": max_results, "summary": True}
    if freshness in _BOCHA_FRESHNESS:
        body["freshness"] = _BOCHA_FRESHNESS[freshness]
    data = _http_json(
        "https://api.bochaai.com/v1/web-search",
        body,
        {"Authorization": f"Bearer {settings.bocha_api_key}"},
    )
    pages = (data.get("data") or {}).get("webPages") or {}
    return [
        {
            "title": r.get("name", ""),
            "url": r.get("url", ""),
            "snippet": r.get("summary") or r.get("snippet", ""),
        }
        for r in pages.get("value", [])
    ]


def _serper_search(query: str, max_results: int, freshness: str) -> list[dict]:
    body: dict = {"q": query, "num": max_results}
    if freshness in _SERPER_TBS:
        body["tbs"] = _SERPER_TBS[freshness]
    data = _http_json(
        "https://google.serper.dev/search",
        body,
        {"X-API-KEY": settings.serper_api_key},
    )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "snippet": r.get("snippet", ""),
        }
        for r in data.get("organic", [])
    ]


# 常规优先级：tavily 质量最稳 > bocha 国内直连 > serper
# 注意只存"是否配置"的判断，函数经 globals() 调用时解析——
# 若在此处固化函数引用，测试就无法用 monkeypatch 替换实现
_PROVIDERS = {
    "tavily": lambda: bool(settings.tavily_api_key),
    "bocha": lambda: bool(settings.bocha_api_key),
    "serper": lambda: bool(settings.serper_api_key),
}
# 时间敏感查询优先级：新鲜度 serper(Google) > tavily(news 窗口) > bocha
_FRESH_PROVIDERS = ("serper", "tavily", "bocha")


def _provider_chain(freshness: str) -> list[str]:
    order = _FRESH_PROVIDERS if freshness in _TAVILY_DAYS else (
        "tavily",
        "bocha",
        "serper",
    )
    return [name for name in order if _PROVIDERS[name]()]


def _provider_fn(name: str):
    return globals()[f"_{name}_search"]


def web_search(query: str, max_results: int = 5, freshness: str = "") -> str:
    """执行联网搜索，返回格式化文本。级联：失败/空结果自动尝试下一个
    已配置的源；全部失败时返回带原因的 ERROR 文案（模型可据此告知
    用户或换关键词重试）。"""
    chain = _provider_chain(freshness)
    if not chain:
        return _NO_KEY_HINT

    last_error: Exception | None = None
    for name in chain:
        fn = _provider_fn(name)
        try:
            results = fn(query, max_results, freshness)
        except Exception as e:
            logger.warning("web search 失败 provider=%s: %s", name, e)
            last_error = e
            continue
        if results:
            return _format(query, name, results)

    if last_error is not None:
        return (
            "ERROR: 搜索服务暂时不可用"
            f"（{type(last_error).__name__}），请稍后重试或更换关键词。"
        )
    return f"没有找到与「{query}」相关的结果，建议更换或简化关键词后重试。"


def _format(query: str, name: str, results: list[dict]) -> str:
    lines = [f"搜索「{query}」的结果（来源：{name}）："]
    for i, r in enumerate(results, 1):
        title = (r["title"] or "无标题").strip()
        snippet = " ".join(r["snippet"].split())
        lines.append(f"{i}. {title}\n   {r['url']}\n   {snippet}")
    return "\n".join(lines)


web_search_tool = Tool(
    name="web_search",
    display_name="智能搜索",
    description=(
        "联网搜索最新信息。当问题涉及实时数据、新闻、价格、版本更新、"
        "或你的知识可能过时时调用。query 用具体的搜索词，不要整句照抄"
        "用户提问；查刚发生的事情时传 freshness=\"day\"。"
    ),
    func=web_search,
    args_model=WebSearchArgs,
)
