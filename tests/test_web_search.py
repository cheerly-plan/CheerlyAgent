import pytest

from ai_agent.tools.registry import execute_tool, tool_schemas
from ai_agent.tools import web_search as ws


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    """默认无任何 key，避免真实网络调用。"""
    for key in ("tavily_api_key", "bocha_api_key", "serper_api_key"):
        monkeypatch.setattr(ws.settings, key, "")


def test_web_search_not_configured_returns_hint():
    out = execute_tool("web_search", {"query": "今天的新闻"})
    assert out.startswith("ERROR: 联网搜索未配置")
    assert "TAVILY_API_KEY" in out


def test_web_search_formats_results(monkeypatch):
    monkeypatch.setattr(ws.settings, "tavily_api_key", "k-test")
    monkeypatch.setattr(
        ws,
        "_tavily_search",
        lambda q, n, f: [
            {
                "title": "FastAPI 发布新版本",
                "url": "https://example.com/a",
                "snippet": "  FastAPI  1.0  正式发布。\n支持新特性。 ",
            }
        ],
    )
    out = execute_tool("web_search", {"query": "FastAPI 新版本"})
    assert "来源：tavily" in out
    assert "1. FastAPI 发布新版本" in out
    assert "https://example.com/a" in out
    assert "FastAPI 1.0 正式发布。 支持新特性。" in out  # 摘要压成单行


def test_web_search_cascades_on_failure(monkeypatch):
    """第一个源失败自动落到下一个已配置的源。"""
    monkeypatch.setattr(ws.settings, "tavily_api_key", "k")
    monkeypatch.setattr(ws.settings, "serper_api_key", "k")

    def boom(q, n, f):
        raise TimeoutError("timeout")

    monkeypatch.setattr(ws, "_tavily_search", boom)
    monkeypatch.setattr(
        ws,
        "_serper_search",
        lambda q, n, f: [{"title": "t", "url": "u", "snippet": "s"}],
    )
    out = execute_tool("web_search", {"query": "x"})
    assert "来源：serper" in out


def test_web_search_cascades_on_empty_results(monkeypatch):
    """空结果不算成功，继续尝试下一个源。"""
    monkeypatch.setattr(ws.settings, "tavily_api_key", "k")
    monkeypatch.setattr(ws.settings, "bocha_api_key", "k")
    monkeypatch.setattr(ws, "_tavily_search", lambda q, n, f: [])
    monkeypatch.setattr(
        ws,
        "_bocha_search",
        lambda q, n, f: [{"title": "t", "url": "u", "snippet": "s"}],
    )
    out = execute_tool("web_search", {"query": "x"})
    assert "来源：bocha" in out


def test_freshness_routes_to_serper_first(monkeypatch):
    """时间敏感查询重排候选链：serper > tavily > bocha。"""
    monkeypatch.setattr(ws.settings, "tavily_api_key", "k")
    monkeypatch.setattr(ws.settings, "serper_api_key", "k")
    calls: list[str] = []

    def record(name, results):
        def fn(q, n, f):
            calls.append(name)
            return results

        return fn

    monkeypatch.setattr(ws, "_tavily_search", record("tavily", []))
    monkeypatch.setattr(
        ws,
        "_serper_search",
        record(
            "serper",
            [{"title": "刚刚发生", "url": "u", "snippet": "s"}],
        ),
    )
    out = execute_tool("web_search", {"query": "热点", "freshness": "day"})
    assert calls == ["serper"]  # 没轮到 tavily
    assert "来源：serper" in out

    # freshness 透传给 provider：tavily 走 news 窗口
    captured = {}
    monkeypatch.setattr(ws.settings, "serper_api_key", "")
    monkeypatch.setattr(
        ws, "_tavily_search", lambda q, n, f: captured.update(f=f) or []
    )
    execute_tool("web_search", {"query": "热点", "freshness": "week"})
    assert captured["f"] == "week"


def test_all_sources_fail_returns_error(monkeypatch):
    monkeypatch.setattr(ws.settings, "bocha_api_key", "k")

    def boom(q, n, f):
        raise TimeoutError("timeout")

    monkeypatch.setattr(ws, "_bocha_search", boom)
    out = execute_tool("web_search", {"query": "x"})
    assert out.startswith("ERROR: 搜索服务暂时不可用")
    assert "TimeoutError" in out


def test_all_sources_empty_returns_no_result_hint(monkeypatch):
    monkeypatch.setattr(ws.settings, "bocha_api_key", "k")
    monkeypatch.setattr(ws, "_bocha_search", lambda q, n, f: [])
    out = execute_tool("web_search", {"query": "生僻词"})
    assert out.startswith("没有找到")


def test_web_search_registered_with_schema():
    names = [s["function"]["name"] for s in tool_schemas()]
    assert "web_search" in names
    schema = next(
        s["function"]["parameters"]
        for s in tool_schemas()
        if s["function"]["name"] == "web_search"
    )
    assert "freshness" in schema["properties"]
