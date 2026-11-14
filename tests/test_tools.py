from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ai_agent.tools.registry import execute_tool, tool_schemas
from ai_agent.tools.kb.kb_search import KbSearchArgs
from ai_agent.tools.kb.kg_search import KgSearchArgs


def test_execute_unknown_tool():
    result = execute_tool("not_exist", {})
    assert result.startswith("ERROR:")


def test_write_memory_profile_kind():
    with (
        patch(
            "ai_agent.tools.memory.memory.apply_profile_text",
            return_value="已写入用户画像: 喜欢 pytest",
        ) as apply_profile,
        patch("ai_agent.tools.memory.memory.store.write_episode") as write_episode,
    ):
        result = execute_tool("write_memory", {"content": "喜欢 pytest", "kind": "profile"})
    assert "用户画像" in result
    apply_profile.assert_called_once_with("喜欢 pytest")
    write_episode.assert_not_called()


def test_write_memory_defaults_to_profile():
    with (
        patch(
            "ai_agent.tools.memory.memory.apply_profile_text",
            return_value="已写入用户画像: 喜欢 pytest",
        ) as apply_profile,
        patch("ai_agent.tools.memory.memory.store.write_episode") as write_episode,
    ):
        result = execute_tool("write_memory", {"content": "喜欢 pytest"})
    assert "用户画像" in result
    apply_profile.assert_called_once_with("喜欢 pytest")
    write_episode.assert_not_called()


def test_write_memory_episode_kind():
    item = SimpleNamespace(text="上周修过登录")
    with (
        patch("ai_agent.tools.memory.memory.store.write_profile") as write_profile,
        patch("ai_agent.tools.memory.memory.store.write_episode", return_value=item) as write_episode,
    ):
        result = execute_tool("write_memory", {"content": "上周修过登录", "kind": "episode"})
    assert "情景记忆" in result
    write_episode.assert_called_once_with("上周修过登录")
    write_profile.assert_not_called()


def test_search_memory_uses_episode_store():
    item = SimpleNamespace(text="上周修过登录")
    with patch("ai_agent.tools.memory.memory.store.search_episode", return_value=[item]) as search:
        result = execute_tool("search_memory", {"query": "登录"})
    assert "上周修过登录" in result
    search.assert_called_once_with("登录")


# ——— kb_search / kg_search 的 files 选填：不填时由文档路由选范围 ———


def test_kb_search_files_optional():
    """files 选填：缺省不再被 schema 拦下。

    范围不再依赖模型对着文件清单挑——文档路由（kb/router.py）按摘要向量
    自己选定文件，文件清单也因此不必占用上下文。
    """
    args = KbSearchArgs(query="中国能建三季报怎么样")
    assert args.files is None
    args = KbSearchArgs(query="中国能建三季报怎么样", files=["中国能建三季报.pdf"])
    assert args.files == ["中国能建三季报.pdf"]


def test_kg_search_files_optional():
    args = KgSearchArgs(query="中国能建 关联方")
    assert args.files is None
    args = KgSearchArgs(query="中国能建 关联方", files=["中国能建三季报.pdf"])
    assert args.files == ["中国能建三季报.pdf"]


def test_execute_kb_search_accepts_files_json_string(monkeypatch):
    """模型把 files 收成一段 JSON 字符串时，还原成列表再检索。"""
    from ai_agent.tools.registry import TOOLS_MAP

    seen: dict = {}

    def fake(query, files=None, top_k=3, kinds=None, sections=None):
        seen["query"] = query
        seen["files"] = files
        return "ok"

    monkeypatch.setattr(TOOLS_MAP["kb_search"], "func", fake)
    result = execute_tool(
        "kb_search",
        {
            "query": "吴义东",
            "files": '["中国黄金集团黄金珠宝股份有限公司2026年第三季度报告"]',
        },
    )
    assert result == "ok"
    assert seen["query"] == "吴义东"
    assert seen["files"] == ["中国黄金集团黄金珠宝股份有限公司2026年第三季度报告"]


def test_execute_kb_search_accepts_single_filename_string(monkeypatch):
    from ai_agent.tools.registry import TOOLS_MAP

    seen: dict = {}

    def fake(query, files=None, top_k=3, kinds=None, sections=None):
        seen["files"] = files
        return "ok"

    monkeypatch.setattr(TOOLS_MAP["kb_search"], "func", fake)
    result = execute_tool(
        "kb_search",
        {"query": "吴义东", "files": "中国黄金集团黄金珠宝股份有限公司2026年第三季度报告"},
    )
    assert result == "ok"
    assert seen["files"] == ["中国黄金集团黄金珠宝股份有限公司2026年第三季度报告"]


def test_execute_kb_search_missing_files():
    # 与 test_execute_missing_city 同理：参数缺失走 schema / 运行兜底，
    # 且不该伪装成「检索成功」
    result = execute_tool("kb_search", {})
    assert result.startswith("ERROR:")


def test_kb_search_without_files_uses_doc_route(monkeypatch):
    """files 没填时，检索范围来自文档路由而不是模型猜文件名。"""
    import uuid

    from ai_agent.tools.kb.kb_search import _kb_search

    fid = uuid.uuid4()
    seen: dict = {}
    monkeypatch.setattr(
        "ai_agent.storage.store.store.has_ready_kb_file", lambda: True
    )
    monkeypatch.setattr(
        "ai_agent.kb.retrieval.router.route_files", lambda query, **kwargs: [fid]
    )
    monkeypatch.setattr(
        "ai_agent.tools.kb.kb_search.search_knowledge_multi",
        lambda query, *, file_ids=None, top_k=3, filters=None: seen.setdefault("file_ids", file_ids)
        and [],
    )
    _kb_search("中国能建营收是多少")
    assert seen["file_ids"] == [fid]
