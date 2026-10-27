"""混合检索（关键词+向量 RRF）与 GraphRAG 证据融合的单元测试。"""

import uuid

from ai_agent.core.config import settings
from ai_agent.kb import retrieval
from ai_agent.kb.citations import collect_citations
from ai_agent.tools import kg_search

FILE_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")
OTHER_ID = uuid.UUID("66666666-2222-3333-4444-555555555555")


def _row(file_id, seq, text, *, distance=0.3):
    return {
        "id": uuid.uuid4(),
        "file_id": file_id,
        "filename": "考勤制度.docx",
        "seq": seq,
        "page": None,
        "kind": "text",
        "text": text,
        "distance": distance,
    }


# ---------- 关键词提取 ----------


def test_extract_keywords_keeps_codes_and_drops_stopwords():
    kws = retrieval._extract_keywords("迟到怎么处罚 OT-01 编号是什么")
    assert "OT-01" in kws  # 英文/数字 token 必须保留（向量最容易丢）
    assert "怎么" not in kws and "什么" not in kws  # 疑问词没有区分度
    assert all(len(kw) >= 2 for kw in kws)
    assert len(kws) <= settings.kb_hybrid_keywords


def test_extract_keywords_survives_jieba_failure(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name.startswith("jieba"):
            raise RuntimeError("jieba 不可用")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    assert retrieval._extract_keywords("迟到怎么处罚") == []


# ---------- RRF 融合 ----------


def test_rrf_merge_ranks_double_hits_first():
    a = _row(FILE_ID, 0, "甲")
    b = _row(FILE_ID, 1, "乙")
    c = _row(OTHER_ID, 2, "丙")
    merged = retrieval._rrf_merge([a, b], [b, c])
    # 两路都命中的 b 叠加分最高，排第一；其余按各自名次
    assert [row["seq"] for row in merged] == [1, 0, 2]


def test_rrf_merge_keyword_only_rows_carry_no_distance():
    only_kw = _row(FILE_ID, 3, "关键词命中")
    only_kw.pop("distance")
    merged = retrieval._rrf_merge([], [only_kw])
    assert float(merged[0].get("distance") or 0.0) == 0.0


def test_fuse_disabled_returns_vector_rows_untouched(monkeypatch):
    monkeypatch.setattr(settings, "kb_hybrid_enabled", False)
    vector_rows = [_row(FILE_ID, 0, "甲")]
    assert retrieval._fuse_with_keywords("迟到", vector_rows, None) is vector_rows


def test_fuse_falls_back_on_keyword_failure(monkeypatch):
    monkeypatch.setattr(settings, "kb_hybrid_enabled", True)

    def boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(retrieval.store, "search_kb_chunks_keyword", boom)
    vector_rows = [_row(FILE_ID, 0, "甲")]
    assert retrieval._fuse_with_keywords("迟到", vector_rows, None) == vector_rows


def test_search_knowledge_merges_both_recalls(monkeypatch):
    """向量路与关键词路结果合并后一起走去重/截断。"""
    vec = _row(FILE_ID, 0, "向量召回片段")
    kw = _row(OTHER_ID, 5, "关键词召回片段")
    monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1])
    monkeypatch.setattr(
        retrieval.store, "search_kb_chunks", lambda *a, **k: [vec]
    )
    monkeypatch.setattr(
        retrieval.store, "search_kb_chunks_keyword", lambda *a, **k: [kw]
    )
    monkeypatch.setattr(retrieval.store, "kb_chunk_neighbors", lambda *a, **k: {})
    monkeypatch.setattr(settings, "kb_hybrid_enabled", True)
    monkeypatch.setattr(settings, "kb_rerank_enabled", False)
    monkeypatch.setattr(settings, "kb_neighbor_expand", 0)

    hits = retrieval.search_knowledge("迟到处罚", top_k=3)
    texts = [hit.text for hit in hits]
    assert "向量召回片段" in texts and "关键词召回片段" in texts

    # 同一片段两路都命中时只保留一条
    monkeypatch.setattr(
        retrieval.store,
        "search_kb_chunks_keyword",
        lambda *a, **k: [_row(FILE_ID, 0, "向量召回片段")],
    )
    hits = retrieval.search_knowledge("迟到处罚", top_k=3)
    assert len(hits) == 1


# ---------- GraphRAG 证据融合 ----------


def _entity(name):
    return {"id": uuid.uuid4(), "name": name, "type": "公司", "score": 0.9}


def _evidence(entity_id, seq, text, filename="策略参考.pdf"):
    return {
        "file_id": uuid.uuid4() if entity_id is None else entity_id,
        "chunk_seq": seq,
        "page": None,
        "kind": "text",
        "text": text,
        "filename": filename,
    }


def _stub_graph(monkeypatch, seeds, nodes, edges, chunks):
    monkeypatch.setattr(kg_search.store, "graph_stats", lambda **k: {"entities": 10})
    monkeypatch.setattr(kg_search, "embed_query", lambda q: [0.1])
    monkeypatch.setattr(kg_search.store, "search_entities", lambda *a, **k: seeds)
    monkeypatch.setattr(
        kg_search.store, "expand_subgraph", lambda *a, **k: {"nodes": nodes, "edges": edges}
    )
    monkeypatch.setattr(kg_search.store, "get_entity_chunks", lambda ids, per_entity=3: chunks)


def test_kg_search_fuses_seed_neighbor_and_vector_evidence(monkeypatch):
    e1, e2 = _entity("长鑫存储"), _entity("DRAM")
    ev1 = _evidence(e1["id"], 0, "长鑫存储是存储原厂")
    ev2 = _evidence(e2["id"], 7, "DRAM 合约价上涨")
    chunks = {str(e1["id"]): [ev1], str(e2["id"]): [ev2]}

    # 向量兜底召回与种子证据重复 → 应被去重掉
    vec_dup = {
        "file_id": ev1["file_id"],
        "seq": 0,
        "page": None,
        "kind": "text",
        "text": "长鑫存储是存储原厂",
        "filename": "策略参考.pdf",
        "distance": 0.2,
    }
    monkeypatch.setattr(
        kg_search.store, "search_kb_chunks", lambda *a, **k: [vec_dup]
    )
    _stub_graph(
        monkeypatch,
        seeds=[e1],
        nodes=[{"id": e1["id"], "name": "长鑫存储"}, {"id": e2["id"], "name": "DRAM"}],
        edges=[
            {
                "source_id": e1["id"],
                "target_id": e2["id"],
                "relation": "生产",
                "description": "",
            }
        ],
        chunks=chunks,
    )
    monkeypatch.setattr(settings, "kb_search_max_distance", 0.72)

    with collect_citations() as bucket:
        out = kg_search._kg_search("长鑫存储的产业链定位")

    assert "命中实体：长鑫存储" in out
    assert "—生产→" in out
    assert "长鑫存储是存储原厂" in out
    assert "DRAM 合约价上涨" in out  # 邻居实体证据被带出
    assert out.count("长鑫存储是存储原厂") == 1  # 向量重复证据已去重
    # 引用旁路：图谱证据同样可点开
    keys = {c.key for c in bucket}
    assert f"{ev1['file_id']}#0" in keys and f"{ev2['file_id']}#7" in keys


def test_kg_search_evidence_cap_and_order(monkeypatch):
    """总量封顶：种子在前、邻居在后，超出 _MAX_EVIDENCE 的丢弃。"""
    seeds = [_entity(f"实体{i}") for i in range(4)]
    chunks = {
        str(s["id"]): [
            _evidence(s["id"], seq=10 + i, text=f"种子证据{i}"),
            _evidence(s["id"], seq=20 + i, text=f"种子证据{i}-b"),
        ]
        for i, s in enumerate(seeds)
    }
    monkeypatch.setattr(kg_search.store, "search_kb_chunks", lambda *a, **k: [])
    _stub_graph(
        monkeypatch,
        seeds=seeds,
        nodes=[{"id": s["id"], "name": s["name"]} for s in seeds],
        edges=[],
        chunks=chunks,
    )

    out = kg_search._kg_search("问题")
    assert "种子证据0" in out and "种子证据3" in out
    # 4 个种子 × 2 条 = 8 条恰好封顶
    assert kg_search._MAX_EVIDENCE == 8
    assert out.count("原文依据") == 1


def test_kg_search_evidence_where_matches_kb_conventions():
    assert kg_search._evidence_where({"kind": "table", "page": 2, "chunk_seq": 0}) == "表2"
    assert (
        kg_search._evidence_where({"kind": "figure", "page": 3, "chunk_seq": 0})
        == "第3页图表"
    )
    assert kg_search._evidence_where({"kind": "text", "page": 5, "chunk_seq": 0}) == "第5页"
    assert kg_search._evidence_where({"kind": "text", "page": None, "chunk_seq": 4}) == "第5段"


def test_kg_search_reports_empty_graph(monkeypatch):
    monkeypatch.setattr(kg_search.store, "graph_stats", lambda **k: {"entities": 0})
    assert "知识图谱还没有数据" in kg_search._kg_search("任何问题")
