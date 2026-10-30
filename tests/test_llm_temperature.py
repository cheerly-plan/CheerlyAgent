"""温度参数的下传链路。

背景：`complete_chat` 早期不传 temperature，走服务端默认（OpenAI 系 1.0）。
结构化任务（图谱抽取 / 实体对齐 / 重排 / 改写）在 1.0 下的症状是
「同一批输入两次跑出不同结果」——实体数差 ±20、JSON 偶发裸词坏输出、
评测失败集合每次换一批。修法不是靠重试补救，而是把这几个任务钉在 0 温。

这些用例盯两件事：
1. `complete_chat` 只在显式传值时下发 `temperature`（主对话不受影响）；
2. 四个结构化调用点确实把 settings 里的值传下去了——参数漏传是最容易
   复发的一类改动，且线上完全看不出来（只是"结果又抖了"）。
"""

import json
from types import SimpleNamespace

from ai_agent.agents import loop
from ai_agent.core.config import settings
from ai_agent.kb import align, graph, retrieval
from ai_agent.kb.chunking import Chunk
from ai_agent.llm import client


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


def _patch_client(monkeypatch, sink: list[dict], content: str = "{}") -> None:
    """把模块级 OpenAI 客户端换成记录入参的假对象。"""

    class _Completions:
        def create(self, **kwargs):
            sink.append(kwargs)
            return _Resp(content)

    class _Client:
        chat = SimpleNamespace(completions=_Completions())

        def with_options(self, **_kwargs):
            return self

    monkeypatch.setattr(client, "_client", _Client())


def test_complete_chat_sends_temperature_when_given(monkeypatch):
    sink: list[dict] = []
    _patch_client(monkeypatch, sink)
    client.complete_chat([{"role": "user", "content": "hi"}], temperature=0.0)
    assert sink[0]["temperature"] == 0.0


def test_graph_client_uses_dedicated_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "kb_graph_base_url", "https://graph.example/v1")
    monkeypatch.setattr(settings, "kb_graph_api_key", "graph-key")
    monkeypatch.setattr(settings, "kb_graph_model", "online-extract")
    other = client.graph_client()
    assert other is not client._client
    assert "graph.example" in str(other.base_url)
    assert client.graph_model() == "online-extract"


def test_graph_client_falls_back_to_main(monkeypatch):
    monkeypatch.setattr(settings, "kb_graph_base_url", "")
    monkeypatch.setattr(settings, "kb_graph_api_key", "")
    assert client.graph_client() is client._client


def test_complete_chat_omits_temperature_by_default(monkeypatch):
    """不传就不下发：主对话/摘要这类需要一点发散的任务仍走服务端默认。"""
    sink: list[dict] = []
    _patch_client(monkeypatch, sink)
    client.complete_chat([{"role": "user", "content": "hi"}])
    assert "temperature" not in sink[0]


def test_graph_extraction_forwards_temperature(monkeypatch):
    seen: dict = {}

    def fake(messages, **kwargs):
        seen.update(kwargs)
        return json.dumps(
            {
                "entities": [
                    {"name": "甲公司", "type": "org", "description": "一家公司"}
                ],
                "relations": [],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(graph, "complete_chat", fake)
    monkeypatch.setattr(settings, "kb_graph_temperature", 0.3)
    graph._extract_batch([Chunk("甲公司本期营业收入同比增长百分之十。")], 0, "system")
    assert seen["temperature"] == 0.3
    assert "client" in seen


def test_align_verdict_forwards_temperature(monkeypatch):
    seen: dict = {}

    def fake(messages, **kwargs):
        seen.update(kwargs)
        return json.dumps({"same": [False]})

    monkeypatch.setattr(align, "complete_chat", fake)
    monkeypatch.setattr(settings, "kb_graph_temperature", 0.25)
    pair = {
        "a": {"name": "甲公司", "type": "org", "description": ""},
        "b": {"name": "甲股份公司", "type": "org", "description": ""},
    }
    align.judge_pairs([pair])
    assert seen["temperature"] == 0.25


def test_rerank_forwards_temperature(monkeypatch):
    seen: dict = {}

    def fake(messages, **kwargs):
        seen.update(kwargs)
        return "[1]"

    # 重排用的是模块内延迟导入的 complete_chat，从目标模块取
    monkeypatch.setattr(client, "complete_chat", fake)
    monkeypatch.setattr(settings, "kb_rerank_temperature", 0.1)
    retrieval.complete_chat_once("把下面几条排序")
    assert seen["temperature"] == 0.1


def test_rewrite_forwards_temperature(monkeypatch):
    seen: dict = {}

    def fake(messages, **kwargs):
        seen.update(kwargs)
        return "中国能建 2026 年三季度营业收入"

    monkeypatch.setattr(loop, "complete_chat", fake)
    monkeypatch.setattr(settings, "kb_rewrite_temperature", 0.2)
    loop._rewrite_query("它的营业收入呢", ["中国能建三季报怎么样"])
    assert seen["temperature"] == 0.2
