"""文档级摘要索引 + 文档路由。

解决的问题：以前「该查哪几份文件」全靠模型对着注入的文件清单挑 files，
清单一到几十上百份就既撑爆上下文又挑不准（还会编造清单外的文件名）。
现在每份文件有一份摘要向量，检索前先在文档这一层做一次路由。
"""

import uuid
from contextlib import contextmanager

from ai_agent.core.config import settings
from ai_agent.kb import scope, summary as summary_mod
from ai_agent.kb.router import mentioned_file_ids, route_files
from ai_agent.kb.summary import build_file_summary, first_sentences, index_file_summary
from ai_agent.memory import store as memory_store
from ai_agent.memory.store import store


def _file(
    fid: uuid.UUID, filename: str, subject: str = "", aliases=(), status: str = "ready"
) -> dict:
    return {
        "file_id": fid,
        "filename": filename,
        "status": status,
        "subject": subject,
        "subject_aliases": list(aliases),
        "period": "2026Q3",
        "doc_type": "",
        "description": "",
    }


# ---------- 摘要拼装 ----------


def test_first_sentences_cuts_at_sentence_boundary():
    """宁可短一点也不留半句话。"""
    text = first_sentences("第一句讲营业收入。第二句讲毛利率。第三句讲现金流。", 12)
    assert text == "第一句讲营业收入。"


def test_first_sentences_keeps_short_text():
    assert first_sentences("营业收入增长 5%", 100) == "营业收入增长 5%"
    assert first_sentences("", 100) == ""


def test_build_summary_covers_subject_period_and_entities():
    text = build_file_summary(
        filename="顺丰控股股份有限公司2026年第三季度报告.pdf",
        subject="顺丰控股股份有限公司",
        aliases=["顺丰控股", "顺丰"],
        period="2026Q3",
        doc_type="上市公司季报",
        entities=["时效业务"],
        chunks=["顺丰控股第三季度实现营业收入一百亿元，同比增长百分之五。"],
    )
    assert text.startswith("文件名：顺丰控股股份有限公司2026年第三季度报告.pdf")
    assert "主体：顺丰控股股份有限公司（又称：顺丰控股、顺丰）" in text
    assert "报告期：2026Q3" in text
    assert "文档类型：上市公司季报" in text
    assert "涉及主体：时效业务" in text
    assert "内容摘录：" in text


def test_build_summary_skips_empty_fields():
    """空字段不出现在摘要里：「主体：」这种空行对向量有害无益。"""
    assert build_file_summary(filename="员工考勤制度.docx") == (
        "文件名：员工考勤制度.docx"
    )


def test_build_summary_respects_max_chars(monkeypatch):
    monkeypatch.setattr(settings, "kb_summary_max_chars", 120)
    text = build_file_summary(
        filename="x.pdf", subject="某某公司", chunks=["正文很长" * 80]
    )
    assert len(text) <= 120


# ---------- 摘要写库 ----------


def test_index_file_summary_skips_table_chunks_and_marks_ready(monkeypatch):
    """表头（「单位：万元」）不进摘要；写库时状态与向量一起落。"""
    fid = uuid.uuid4()
    written: dict = {}
    monkeypatch.setattr(
        store,
        "get_kb_file",
        lambda _: {
            "filename": "a.pdf",
            "subject": "甲公司股份有限公司",
            "subject_aliases": ["甲公司"],
            "period": "2026Q3",
            "doc_type": "",
            "description": "",
        },
    )
    monkeypatch.setattr(
        store,
        "list_kb_chunk_inputs",
        lambda _: [
            {"kind": "text", "text": "甲公司第三季度营业收入增长。"},
            {"kind": "table", "text": "单位：万元"},
        ],
    )
    monkeypatch.setattr(store, "list_kb_file_entity_names", lambda _f, limit: ["乙公司"])
    monkeypatch.setattr(
        store,
        "set_kb_file_summary",
        lambda file_id, text, emb, status="": written.update(
            text=text, emb=emb, status=status
        ),
    )
    monkeypatch.setattr(
        "ai_agent.memory.embeddings.embed_texts", lambda texts: [[0.1, 0.2]]
    )
    out = index_file_summary(fid)
    assert out
    assert written["status"] == "ready"
    assert written["emb"] == [0.1, 0.2]
    # 表格片段没被收录，实体名进摘要
    assert "单位：万元" not in written["text"]
    assert "涉及主体：乙公司" in written["text"]


# ---------- 点名命中 ----------


def test_mentioned_file_ids_covers_both_sides_of_comparison(monkeypatch):
    """「A 和 B 的营业收入」：两家都得进得来，不能只回一家。"""
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    rows = [
        _file(
            a,
            "中国能源建设股份有限公司2026年第三季度报告.pdf",
            "中国能源建设股份有限公司",
            ["中国能源建设", "中国能建"],
        ),
        _file(
            b,
            "顺丰控股股份有限公司2026年第三季度报告.pdf",
            "顺丰控股股份有限公司",
            ["顺丰控股", "顺丰"],
        ),
        _file(c, "中国银行股份有限公司2026年半年度报告.pdf", "中国银行股份有限公司"),
    ]
    monkeypatch.setattr(memory_store.store, "list_kb_files", lambda **k: rows)
    hits = mentioned_file_ids("中国能源建设和顺丰控股的营业收入分别是多少")
    assert set(hits) == {a, b}


def test_mentioned_file_ids_ignores_unrelated_query(monkeypatch):
    a = uuid.uuid4()
    monkeypatch.setattr(
        memory_store.store,
        "list_kb_files",
        lambda **k: [
            _file(a, "顺丰控股股份有限公司2026年第三季度报告.pdf", "顺丰控股股份有限公司")
        ],
    )
    # 「季度报告」这种噪音不该把文件拉进来
    assert mentioned_file_ids("今年季度报告里毛利率怎么算") == []


# ---------- 路由 ----------


def test_route_files_pads_named_files_with_summary_hits(monkeypatch):
    """点名的在前（确定性命中），摘要向量补齐没点名的那部分。"""
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    monkeypatch.setattr(
        memory_store.store,
        "list_kb_files",
        lambda **k: [
            _file(
                a,
                "中国能源建设股份有限公司2026年第三季度报告.pdf",
                "中国能源建设股份有限公司",
                ["中国能建"],
            ),
            _file(c, "新能源行业研究报告.pdf", "新能源行业"),
        ],
    )
    monkeypatch.setattr("ai_agent.kb.router.embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(
        store,
        "search_kb_files",
        lambda emb, **kw: [{"file_id": c, "distance": 0.2}, {"file_id": b, "distance": 0.3}],
    )
    assert route_files("中国能建的营业收入") == [a, c, b]


def test_route_files_stays_inside_session_scope(monkeypatch):
    """会话已限定范围时，路由不该把范围外的文件放进来。"""
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    monkeypatch.setattr(
        memory_store.store,
        "list_kb_files",
        lambda **k: [_file(a, "甲公司2026年第三季度报告.pdf", "甲公司股份有限公司")],
    )
    seen: dict = {}
    # 真实库里 search_kb_files 已按 file_ids 过滤，这里让假实现照做，
    # 断言才落在「路由把范围透传下去」这件事上
    def fake_search(emb, *, file_ids=None, top_k=6, max_distance=None):
        seen["file_ids"] = file_ids
        seen["top_k"] = top_k
        return [
            {"file_id": fid, "distance": 0.2}
            for fid in (b,)
            if file_ids is None or fid in file_ids
        ]

    monkeypatch.setattr("ai_agent.kb.router.embed_query", lambda q: [0.1])
    monkeypatch.setattr(store, "search_kb_files", fake_search)
    assert route_files("甲公司营收", file_ids=[b, a]) == [a, b]
    assert seen["file_ids"] == [b, a]


def test_route_files_disabled_returns_empty(monkeypatch):
    """关掉开关即退回旧行为（会话范围 / 全库）。"""
    monkeypatch.setattr(settings, "kb_doc_route_enabled", False)
    assert route_files("甲公司营收") == []


def test_route_files_empty_query_returns_empty():
    assert route_files("") == []


# ---------- 接入 scope 解析 ----------


def test_resolve_tool_file_ids_uses_route_when_no_files(monkeypatch):
    fid = uuid.uuid4()
    monkeypatch.setattr(scope, "_route", lambda q: [fid])
    assert scope.resolve_tool_file_ids(None, query="甲公司营收") == [fid]


def test_resolve_tool_file_ids_falls_back_to_none_without_route(monkeypatch):
    """路由也没结果、库里又没有可收窄的文件时，退回会话范围 / 全库。"""
    monkeypatch.setattr(scope, "_route", lambda q: [])
    monkeypatch.setattr(memory_store.store, "list_kb_files", lambda **k: [])
    assert scope.resolve_tool_file_ids(None, query="随便问点什么") is None


def test_kb_search_without_files_hits_summary_index(monkeypatch):
    """端到端：不填 files 时，kb_search 的范围来自摘要索引而不是全库。"""
    from ai_agent.tools.kb_search import _kb_search

    fid = uuid.uuid4()
    seen: dict = {}
    monkeypatch.setattr("ai_agent.memory.store.store.has_ready_kb_file", lambda: True)
    monkeypatch.setattr(
        memory_store.store,
        "list_kb_files",
        lambda **k: [_file(fid, "甲公司2026年第三季度报告.pdf", "甲公司股份有限公司")],
    )
    monkeypatch.setattr("ai_agent.kb.router.embed_query", lambda q: [0.1, 0.2])

    def fake_files(emb, *, file_ids=None, top_k=6, max_distance=None):
        seen["max_distance"] = max_distance
        seen["top_k"] = top_k
        return [{"file_id": fid, "distance": 0.2}]

    monkeypatch.setattr(memory_store.store, "search_kb_files", fake_files)
    monkeypatch.setattr(
        "ai_agent.tools.kb_search.search_knowledge_multi",
        lambda query, *, file_ids=None, top_k=3, filters=None: seen.setdefault("file_ids", file_ids)
        and [],
    )
    _kb_search("甲公司第三季度的净利润是多少")
    assert seen["file_ids"] == [fid]
    assert seen["max_distance"] == settings.kb_doc_route_max_distance


def test_resolve_tool_file_ids_prefers_explicit_files(monkeypatch):
    """模型点名了就以它为准，路由不插手。"""
    fid = uuid.uuid4()
    monkeypatch.setattr(scope, "_route", lambda q: [uuid.uuid4()])
    monkeypatch.setattr(
        "ai_agent.kb.retrieval.resolve_files_by_name", lambda names: [fid]
    )
    assert scope.resolve_tool_file_ids(["中国能建"], query="营收") == [fid]


def test_resolve_tool_file_ids_single_ready_file_not_treated_as_not_found(monkeypatch):
    """回归：库里只有一份就绪文件、模型点名命中时，不应被误判为「文件不存在」。

    原实现 `len(matched) == len(ready)` 在 ready==1 时把合法命中短路成 None，
    上层再统一转成 []，导致 kb_search 返回「未找到」、检索被停。
    """
    fid = uuid.uuid4()
    rows = [_file(fid, "中国银行2026年第三季度报告.pdf", subject="中国银行", aliases=["中行"])]
    monkeypatch.setattr(memory_store.store, "list_kb_files", lambda **k: rows)
    assert scope.resolve_tool_file_ids(["中国银行"]) == [fid]


# ---------- 检索 SQL：文档层而不是片段层 ----------


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, sink):
        self._sink = sink

    def execute(self, sql, params=None):
        self._sink.append(sql)
        return _FakeResult([])

    def commit(self):
        return None


class _FakePool:
    """假连接池：只记录 SQL，不真连库（单测环境没有 Postgres）。"""

    def __init__(self, sink):
        self._sink = sink

    @contextmanager
    def connection(self):
        yield _FakeConn(self._sink)


def _run_sql(monkeypatch, call) -> str:
    sink: list[str] = []
    monkeypatch.setattr(store, "_get_pool", lambda: _FakePool(sink))
    call()
    return "\n".join(sink)


def test_search_kb_files_ranks_by_summary_vector(monkeypatch):
    """文档路由查的是 summary_embedding，不是切片 embedding。"""
    sql = _run_sql(monkeypatch, lambda: store.search_kb_files([0.1, 0.2], top_k=3))
    assert "summary_embedding <=> %(emb)s" in sql
    assert "FROM agent_kb_file" in sql
    assert "status = 'ready'" in sql
    assert "summary_embedding IS NOT NULL" in sql
    assert "ORDER BY distance" in sql
    assert "LIMIT %(top_k)s" in sql


def test_set_kb_file_summary_writes_text_and_vector_together(monkeypatch):
    """摘要与向量必须同一次 UPDATE：分两次写会出现「新文本 + 旧向量」。"""
    sql = _run_sql(
        monkeypatch,
        lambda: store.set_kb_file_summary(
            uuid.uuid4(), "摘要文本", [0.1, 0.2], status="ready"
        ),
    )
    assert "UPDATE agent_kb_file" in sql
    assert "summary = %s" in sql
    assert "summary_embedding = COALESCE(%s, summary_embedding)" in sql
    assert "summary_status = %s" in sql
