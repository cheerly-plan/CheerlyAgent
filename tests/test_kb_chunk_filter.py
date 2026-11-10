"""chunk 级元数据过滤（双轨）的单元测试。

- `_chunk_meta_where`：纯函数，生成 SQL 片段 + 参数，无 DB 依赖
- `_kb_search`：验证 kinds/sections 显式参数与 scope 自动推断能正确汇成 filters 透传给 search_knowledge
"""

import uuid

from ai_agent.kb.retrieval.scope import reset_kb_question, set_kb_question
from ai_agent.kb.store import _chunk_meta_where
from ai_agent.kb.ingest.tables import SCOPE_PARENT
from ai_agent.tools.kb import kb_search


def test_chunk_meta_where_kinds_exact():
    params: dict = {}
    sql = _chunk_meta_where(
        kinds=["table"], subjects=None, periods=None, scopes=None, sections=None, params=params
    )
    assert "c.kind = ANY(%(kinds)s)" in sql
    assert params["kinds"] == ["table"]


def test_chunk_meta_where_subjects_fuzzy():
    # subjects 走 ILIKE 模糊：兼容「中国银行」缩写匹配「中国银行股份有限公司」
    params: dict = {}
    sql = _chunk_meta_where(
        kinds=None, subjects=["中国银行"], periods=None, scopes=None, sections=None, params=params
    )
    assert "c.subject ILIKE %(subject0)s" in sql
    assert params["subject0"] == "%中国银行%"


def test_chunk_meta_where_periods_scopes_exact():
    params: dict = {}
    sql = _chunk_meta_where(
        kinds=None, subjects=None, periods=["2026Q3"], scopes=[SCOPE_PARENT], sections=None, params=params
    )
    assert "c.period = ANY(%(periods)s)" in sql
    assert "c.scope = ANY(%(scopes)s)" in sql
    assert params["periods"] == ["2026Q3"]
    assert params["scopes"] == [SCOPE_PARENT]


def test_chunk_meta_where_sections_multi():
    params: dict = {}
    sql = _chunk_meta_where(
        kinds=None, subjects=None, periods=None, scopes=None,
        sections=["管理层讨论", "财务报表"], params=params,
    )
    assert "c.section ILIKE %(section0)s" in sql
    assert "c.section ILIKE %(section1)s" in sql
    assert params["section0"] == "%管理层讨论%"
    assert params["section1"] == "%财务报表%"


def test_chunk_meta_where_empty():
    # 全部 None/空 → 不过滤（与 file_ids 现有「空列表退化为 NULL」一致）
    params: dict = {}
    sql = _chunk_meta_where(
        kinds=None, subjects=None, periods=None, scopes=None, sections=None, params=params
    )
    assert sql == ""


def _patch_kb_search(monkeypatch):
    captured: dict = {}

    def fake_search(queries, *, file_ids=None, top_k=3, filters=None):
        captured["filters"] = filters
        return []

    monkeypatch.setattr(kb_search, "search_knowledge_multi", fake_search)
    monkeypatch.setattr(kb_search.store, "has_ready_kb_file", lambda: True)
    # 返回一个虚拟 file_id，让 _kb_search 越过「未指定文件」早退、走到 search_knowledge
    monkeypatch.setattr(
        kb_search, "resolve_tool_file_ids", lambda files, query=None: [uuid.uuid4()]
    )
    return captured


def test_kb_search_passes_kinds_sections_to_filters(monkeypatch):
    captured = _patch_kb_search(monkeypatch)
    kb_search._kb_search(
        "列出各公司营业收入", files=None, top_k=3, kinds=["table"], sections=["管理层讨论"]
    )
    assert captured["filters"]["kinds"] == ["table"]
    assert captured["filters"]["sections"] == ["管理层讨论"]


def test_kb_search_infers_parent_scope(monkeypatch):
    captured = _patch_kb_search(monkeypatch)
    kb_search._kb_search("母公司报表的未分配利润", files=None, top_k=3)
    # 明确提及母公司口径 → scope 硬过滤为 parent；合并口径不硬过滤（沿用软偏好）
    assert captured["filters"]["scopes"] == [SCOPE_PARENT]


def test_named_table_keeps_kind_filter():
    assert kb_search._kinds_user_asked("看利润表里的营业收入", ["table"]) == ["table"]
    assert kb_search._sections_user_asked("管理层讨论里的增长", ["管理层讨论"]) == ["管理层讨论"]


def test_field_only_query_returns_to_the_user_question(monkeypatch):
    """「中国黄金的营收增长情况」被收成按章节查营业收入时，改回原句。"""
    gold = uuid.uuid4()
    captured: dict = {}

    def mentions(text: str):
        return [gold] if "中国黄金" in text else []

    def resolve(files, query=None):
        captured["resolve_query"] = query
        return [gold] if query and "中国黄金" in query else []

    def search(queries, **kwargs):
        captured["queries"] = queries
        captured["filters"] = kwargs.get("filters")
        return []

    monkeypatch.setattr("ai_agent.kb.retrieval.router.mentioned_file_ids", mentions)
    monkeypatch.setattr(kb_search, "resolve_tool_file_ids", resolve)
    monkeypatch.setattr(kb_search, "search_knowledge_multi", search)
    monkeypatch.setattr(kb_search, "expand_queries", lambda query, count=None: [query])
    monkeypatch.setattr(kb_search.store, "has_ready_kb_file", lambda: True)
    monkeypatch.setattr(
        kb_search.store,
        "get_kb_file",
        lambda fid: {
            "subject": "中国黄金",
            "subject_aliases": [],
            "filename": "中国黄金2025年报告.pdf",
        },
    )
    token = set_kb_question("中国黄金的营收增长情况")
    try:
        kb_search._kb_search(
            "营业收入",
            files=None,
            top_k=3,
            kinds=["table"],
            sections=["管理层讨论"],
        )
    finally:
        reset_kb_question(token)

    assert captured["resolve_query"] == "中国黄金的营收增长情况"
    assert captured["queries"] == ["营收增长情况"]
    assert "kinds" not in (captured["filters"] or {})
    assert "sections" not in (captured["filters"] or {})
