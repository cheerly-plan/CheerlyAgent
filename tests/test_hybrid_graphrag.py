"""混合检索（关键词+向量 RRF）与 GraphRAG 证据融合的单元测试。"""

import uuid

import pytest

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


def test_rrf_merge_keyword_path_is_discounted():
    """关键词路第 1 名不能压过向量路第 2 名。

    关键词路只数「命中了几个词」，没有 IDF、没有长度归一，同分还按
    (file_id, seq) 排——封面页、目录这种又靠前又短、恰好写着公司名的
    片段会排到第一。同权时它能和向量第一名打平，把带数据的切片挤下去；
    实测「安克创新和顺丰控股哪家营收增速更快」就因此 miss。
    """
    first = _row(FILE_ID, 10, "向量第一名")
    second = _row(FILE_ID, 11, "向量第二名")
    kw_only = _row(OTHER_ID, 20, "只是字面命中了公司名")
    merged = retrieval._rrf_merge([first, second], [kw_only])
    assert [row["seq"] for row in merged] == [10, 11, 20]


def test_rrf_merge_keyword_only_rows_carry_no_distance():
    only_kw = _row(FILE_ID, 3, "关键词命中")
    only_kw.pop("distance")
    merged = retrieval._rrf_merge([], [only_kw])
    assert merged[0].get("distance") is None
    assert retrieval._to_hit(merged[0]).distance is None


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
    kw.pop("distance")
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
    by_text = {hit.text: hit for hit in hits}
    assert by_text["向量召回片段"].distance == 0.3
    assert by_text["关键词召回片段"].distance is None

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


def test_format_neighborhood_lists_seeds_and_edges():
    e1, e2 = _entity("长鑫存储"), _entity("DRAM")
    e1["type"] = "org"
    e2["type"] = "product"
    lines = kg_search.format_neighborhood(
        [e1],
        {
            "nodes": [e1, e2],
            "edges": [
                {
                    "source_id": e1["id"],
                    "target_id": e2["id"],
                    "relation": "生产",
                    "description": "存储原厂",
                }
            ],
        },
    )
    assert lines[0] == "命中实体：长鑫存储(org)"
    assert "长鑫存储 —生产→ DRAM：存储原厂" in lines[2]


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
    assert kg_search._evidence_where(
        {"kind": "table", "page": 2, "chunk_seq": 0, "filename": "招股书.pdf"}
    ) == "第2页"
    assert kg_search._evidence_where(
        {"kind": "table", "page": 2, "chunk_seq": 0, "filename": "财务.xlsx"}
    ) == "表2"
    assert (
        kg_search._evidence_where({"kind": "figure", "page": 3, "chunk_seq": 0})
        == "第3页图表"
    )
    assert kg_search._evidence_where({"kind": "text", "page": 5, "chunk_seq": 0}) == "第5页"
    assert kg_search._evidence_where({"kind": "text", "page": None, "chunk_seq": 4}) == "第5段"
    assert (
        kg_search._evidence_where(
            {
                "kind": "text",
                "page": 17,
                "page_end": 19,
                "section": "四、季度财务报表",
                "chunk_seq": 0,
            }
        )
        == "第17–19页 · 四、季度财务报表"
    )


def test_kg_search_citations_keep_section_out_of_chip_where():
    """给模型的原文依据可以带章节；前端芯片的 where 只能是页码。"""
    [citation] = kg_search._citations_of(
        [
            {
                "file_id": FILE_ID,
                "chunk_seq": 3,
                "page": 17,
                "page_end": 19,
                "kind": "text",
                "filename": "中国能源建设股份有限公司2026年第三季度报告.pdf",
                "section": "四、季度财务报表 · （二） 财务报表",
                "text": "营业收入",
            }
        ]
    )
    assert citation.where == "第17–19页"
    assert citation.section == "四、季度财务报表 · （二） 财务报表"
    assert citation.page_end == 19
    payload = citation.payload()
    assert payload["where"] == "第17–19页"
    assert payload["section"] == citation.section
    assert payload["page_start"] == 17
    assert payload["page_end"] == 19


def test_kg_search_reports_empty_graph(monkeypatch):
    monkeypatch.setattr(kg_search.store, "graph_stats", lambda **k: {"entities": 0})
    assert "知识图谱还没有数据" in kg_search._kg_search("任何问题")


# ---------- 标签预过滤 ----------


def _kb_file(fid, tags, status="ready", filename=""):
    return {"file_id": fid, "tags": tags, "status": status, "filename": filename}


def test_tag_scope_narrows_to_tagged_match():
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    files = [_kb_file(a, ["考勤制度"]), _kb_file(b, ["FAQ"]), _kb_file(c, ["薪酬"])]
    assert retrieval._tag_scope_ids(["考勤"], files) == [a]


def test_tag_scope_falls_back_to_filename_when_tags_empty():
    """文件一个标签都没打时，靠文件名主体收窄——否则这层逻辑永远不生效。"""
    a, b = uuid.uuid4(), uuid.uuid4()
    files = [
        _kb_file(a, [], filename="顺丰控股股份有限公司2026 年第三季度报告"),
        _kb_file(b, [], filename="安克创新科技股份有限公司2026 年第三季度报告"),
    ]
    assert retrieval._tag_scope_ids(["顺丰控股"], files) == [a]


def test_tag_scope_matches_abbreviated_subject():
    """简称与全称差字，子串对不上，靠二元组重合判定同一主体。"""
    a, b = uuid.uuid4(), uuid.uuid4()
    files = [
        _kb_file(a, [], filename="中国能源建设股份有限公司2026 年第三季度报告"),
        _kb_file(b, [], filename="安克创新科技股份有限公司2026 年第三季度报告"),
    ]
    assert retrieval._tag_scope_ids(["中国能建"], files) == [a]


def test_tag_scope_ignores_unrelated_subject():
    a, b = uuid.uuid4(), uuid.uuid4()
    files = [
        _kb_file(a, [], filename="顺丰控股股份有限公司2026 年第三季度报告"),
        _kb_file(b, [], filename="安克创新科技股份有限公司2026 年第三季度报告"),
    ]
    # 库外的主体不该把范围收到任何一份文件上
    assert retrieval._tag_scope_ids(["宁德时代"], files) is None


def test_file_subject_strips_year_and_report_type():
    assert (
        retrieval._file_subject("顺丰控股股份有限公司2026 年第三季度报告.pdf")
        == "顺丰控股股份有限公司"
    )


def test_resolve_files_by_name_matches_abbreviation_and_full_name(monkeypatch):
    nengjian = uuid.uuid4()
    anker = uuid.uuid4()
    monkeypatch.setattr(
        retrieval.store,
        "list_kb_files",
        lambda **k: [
            {
                "file_id": nengjian,
                "filename": "中国能源建设股份有限公司2026年第三季度报告.pdf",
                "status": "ready",
            },
            {
                "file_id": anker,
                "filename": "安克创新科技股份有限公司2026年第三季度报告.pdf",
                "status": "ready",
            },
            {
                "file_id": uuid.uuid4(),
                "filename": "顺丰控股股份有限公司2026年第三季度报告.pdf",
                "status": "ready",
            },
            {
                "file_id": uuid.uuid4(),
                "filename": "还在处理.pdf",
                "status": "processing",
            },
        ],
    )
    assert retrieval.resolve_files_by_name(["中国能建"]) == [nengjian]
    assert retrieval.resolve_files_by_name(
        ["中国能源建设股份有限公司2026年第三季度报告.pdf"]
    ) == [nengjian]
    assert retrieval.resolve_files_by_name(
        ["中国能源建设股份有限公司2026年第三季度报告.pdf｜标签：季报"]
    ) == [nengjian]
    assert retrieval.resolve_files_by_name(["安克创新"]) == [anker]
    # 两个提示都能命中 → 两份都纳入
    assert retrieval.resolve_files_by_name(["中国能建", "安克创新"]) == [
        nengjian,
        anker,
    ]
    # 命中全部就绪文件 = 没圈范围
    assert retrieval.resolve_files_by_name(["季度报告"]) is None
    assert retrieval.resolve_files_by_name([]) is None


def test_query_mentions_allows_abbreviation_but_not_other_company():
    full = "中国能源建设股份有限公司"
    # 简称与全称只共享「中国能」，但足以判定是同一主体
    assert retrieval._query_mentions("中国能建前三季度营收", full)
    assert retrieval._query_mentions("中国能源建设营收", full)
    # 库外的公司一个连续字都对不上，不能收窄
    assert not retrieval._query_mentions("宁德时代营收", full)
    # 少于 3 字的重合不足以判定（"公司" 这类通用后缀不算）
    assert not retrieval._query_mentions("公司营收", full)


def test_keep_distinctive_keywords_drops_ubiquitous_terms(monkeypatch):
    """全库过半切片都含的词没有区分度：命中它不代表相关。"""
    monkeypatch.setattr(
        retrieval.store,
        "keyword_doc_freq",
        lambda kws, file_ids=None: (100, {"营业收入": 80, "601868": 1}),
    )
    assert retrieval._keep_distinctive_keywords(
        ["营业收入", "601868"], None
    ) == ["601868"]


def test_keep_distinctive_keywords_keeps_rare_terms_and_boundary(monkeypatch):
    # 恰好一半：仍保留（只剔除「过半」的）
    monkeypatch.setattr(
        retrieval.store,
        "keyword_doc_freq",
        lambda kws, file_ids=None: (100, {"营业": 50, "收入": 12}),
    )
    assert retrieval._keep_distinctive_keywords(["营业", "收入"], None) == [
        "营业",
        "收入",
    ]


def test_keep_distinctive_keywords_survives_stats_failure(monkeypatch):
    """统计失败时保留全部：不能让一次查询失败改变召回行为。"""

    def boom(kws, file_ids=None):
        raise RuntimeError("db down")

    monkeypatch.setattr(retrieval.store, "keyword_doc_freq", boom)
    assert retrieval._keep_distinctive_keywords(["营业", "收入"], None) == [
        "营业",
        "收入",
    ]


def test_fuse_falls_back_to_vector_when_all_keywords_are_ubiquitous(monkeypatch):
    """关键词全被筛掉时退回纯向量，而不是拿空词表去查库。"""
    monkeypatch.setattr(
        retrieval, "_extract_keywords", lambda q: ["营业收入"]
    )
    monkeypatch.setattr(
        retrieval.store,
        "keyword_doc_freq",
        lambda kws, file_ids=None: (100, {"营业收入": 90}),
    )
    monkeypatch.setattr(
        retrieval.store,
        "search_kb_chunks_keyword",
        lambda *a, **k: pytest.fail("全被筛掉时不该走关键词路"),
    )
    rows = [{"file_id": "f", "seq": 0, "text": "t"}]
    assert retrieval._fuse_with_keywords("营收多少", rows, None) is rows


def test_tag_scope_none_when_no_discrimination():
    a, b = uuid.uuid4(), uuid.uuid4()
    files = [_kb_file(a, ["考勤"]), _kb_file(b, ["考勤制度"])]
    assert retrieval._tag_scope_ids(["考勤"], files) is None  # 全员命中
    assert retrieval._tag_scope_ids(["报销"], files) is None  # 全员未命中


def test_tag_scope_matches_bidirectional_substring_case_insensitive():
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    files = [_kb_file(a, ["OT-01 考勤"]), _kb_file(b, ["薪酬"]), _kb_file(c, ["FAQ"])]
    assert retrieval._tag_scope_ids(["ot-01"], files) == [a]  # 大小写不敏感
    # 方向一：关键词是标签的子串，"考勤" 命中复合标签 "OT-01 考勤"
    assert retrieval._tag_scope_ids(["考勤"], files) == [a]
    # 方向二：标签是关键词的子串，"考勤" 命中 "考勤处罚规定"
    files2 = [_kb_file(a, ["考勤"]), _kb_file(b, ["薪酬"])]
    assert retrieval._tag_scope_ids(["考勤处罚规定"], files2) == [a]


def _hit(fid, seq):
    from ai_agent.kb.retrieval import Hit

    return Hit(
        file_id=str(fid), filename="f", seq=seq, page=None,
        kind="text", text="t", distance=0.4,
    )


def test_search_knowledge_narrows_then_pads_from_full(monkeypatch):
    """收窄后召回不足 top_k：用全库补位，收窄结果仍排前面。"""
    from ai_agent.kb import retrieval

    scoped, other = uuid.uuid4(), uuid.uuid4()
    monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1])
    monkeypatch.setattr(retrieval, "tag_file_scope", lambda q: [scoped])

    calls = []

    def fake_core(query, embedding, *, file_ids, top_k):
        calls.append(file_ids)
        if file_ids:  # 收窄那一搜只有 1 条：弱召回
            return [_hit(scoped, 0)]
        return [_hit(scoped, 0), _hit(other, 1), _hit(other, 2)]

    monkeypatch.setattr(retrieval, "_core_search", fake_core)
    hits = retrieval.search_knowledge("考勤怎么算", top_k=3)
    assert calls == [[scoped], None]  # 先收窄，兜底走全库
    assert len(hits) == 3
    assert (hits[0].file_id, hits[0].seq) == (str(scoped), 0)  # 收窄结果排前


def test_search_knowledge_skips_scope_when_none(monkeypatch):
    from ai_agent.kb import retrieval

    monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1])
    monkeypatch.setattr(retrieval, "tag_file_scope", lambda q: None)
    calls = []

    def fake_core(query, embedding, *, file_ids, top_k):
        calls.append(file_ids)
        return []

    monkeypatch.setattr(retrieval, "_core_search", fake_core)
    retrieval.search_knowledge("随便问", top_k=3)
    assert calls == [None]  # 没有可用范围：只搜一次全库


def test_search_knowledge_skips_scope_when_file_ids_given(monkeypatch):
    """模型/前端已显式圈定文件时不该再猜范围。"""
    from ai_agent.kb import retrieval

    pinned = [uuid.uuid4()]
    monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1])
    monkeypatch.setattr(
        retrieval, "tag_file_scope", lambda q: pytest.fail("显式范围下不该推导")
    )
    calls = []

    def fake_core(query, embedding, *, file_ids, top_k):
        calls.append(file_ids)
        return []

    monkeypatch.setattr(retrieval, "_core_search", fake_core)
    retrieval.search_knowledge("考勤怎么算", file_ids=pinned, top_k=3)
    assert calls == [pinned]


def test_search_knowledge_survives_scope_failure(monkeypatch):
    """范围推导抛异常时退回全库，不能把检索拖挂。"""
    from ai_agent.kb import retrieval

    monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1])

    def boom(q):
        raise RuntimeError("db down")

    monkeypatch.setattr(retrieval, "tag_file_scope", boom)
    calls = []

    def fake_core(query, embedding, *, file_ids, top_k):
        calls.append(file_ids)
        return []

    monkeypatch.setattr(retrieval, "_core_search", fake_core)
    retrieval.search_knowledge("考勤怎么算", top_k=3)
    assert calls == [None]


# ---------- 图谱抽取并行化 ----------


def _chunk(seq):
    from ai_agent.kb.chunking import Chunk

    return Chunk(text=f"片段{seq}", page=None, kind="text")


def _extract_with_marker(chunks, start_seq, _system=None, **_kw):
    """让结果携带 start_seq，验证批序与分批正确性。"""
    from ai_agent.kb.graph_schema import _Entity, _Extraction

    return _Extraction(
        entities=[_Entity(name=f"e{start_seq}", chunk=0)],
    )


def test_extract_all_keeps_batch_order(monkeypatch):
    """并发执行但结果按批序排列：合并阶段的跨批语义不因并行而变。"""
    from ai_agent.kb import graph, graph_extract

    monkeypatch.setattr(graph_extract, "_extract_batch", _extract_with_marker)
    monkeypatch.setattr(graph.settings, "kb_graph_concurrency", 4)

    chunks = [_chunk(i) for i in range(20)]  # 8/批 → 3 批
    out = graph._extract_all(chunks, 8, "sys")

    assert [start for start, _b, _r in out] == [0, 8, 16]
    assert [r.entities[0].name for _s, _b, r in out] == ["e0", "e8", "e16"]
    assert [len(b) for _s, b, _r in out] == [8, 8, 4]  # 尾批不足 8


def test_extract_all_skips_failed_batch(monkeypatch):
    """单批异常只跳过该批，其余批结果保留。"""
    from ai_agent.kb import graph, graph_extract

    calls = []

    def flaky(chunks, start_seq, _system=None, **_kw):
        calls.append(start_seq)
        if start_seq == 8:
            raise ConnectionError("网关抖动")
        return _extract_with_marker(chunks, start_seq)

    monkeypatch.setattr(graph_extract, "_extract_batch", flaky)
    monkeypatch.setattr(graph.settings, "kb_graph_concurrency", 2)

    chunks = [_chunk(i) for i in range(24)]
    out = graph._extract_all(chunks, 8, "sys")

    assert sorted(calls) == [0, 8, 16]
    assert [
        r.entities[0].name if r.entities else "(跳过)" for _s, _b, r in out
    ] == ["e0", "(跳过)", "e16"]


def test_extract_all_raises_when_all_batches_fail(monkeypatch):
    """全部批失败 = 配置类错误：抛给上层标 failed，不能伪装成空图。"""
    import pytest as _pytest

    from ai_agent.kb import graph, graph_extract

    def boom(chunks, start_seq, _system=None, **_kw):
        raise ConnectionError("模型不存在")

    monkeypatch.setattr(graph_extract, "_extract_batch", boom)
    monkeypatch.setattr(graph.settings, "kb_graph_concurrency", 2)

    chunks = [_chunk(i) for i in range(16)]
    with _pytest.raises(RuntimeError):
        graph._extract_all(chunks, 8, "sys")


def test_extract_all_serial_when_concurrency_one(monkeypatch):
    from ai_agent.kb import graph, graph_extract

    monkeypatch.setattr(graph_extract, "_extract_batch", _extract_with_marker)
    monkeypatch.setattr(graph.settings, "kb_graph_concurrency", 1)

    out = graph._extract_all([_chunk(i) for i in range(8)], 8, "sys")
    assert len(out) == 1 and out[0][2].entities[0].name == "e0"


# ---------- 按章节组批（图谱抽取） ----------


def _section_chunk(seq, section):
    from ai_agent.kb.chunking import Chunk

    return Chunk(text=f"片段{seq}", page=None, kind="text", section=section)


def test_batch_bounds_never_splits_a_small_section():
    """一节只有 3 片时不能和上一节的尾巴各切一半：整节成批。"""
    from ai_agent.kb.graph_extract import _batch_bounds

    chunks = [_section_chunk(i, "第一节") for i in range(3)] + [
        _section_chunk(i, "第二节") for i in range(3, 6)
    ]
    bounds = _batch_bounds(chunks, 8)
    # 两个小节合进一批（都是小章节），但每节在批内是连续的
    assert [len(b) for _s, b in bounds] == [6]
    assert [c.section for c in bounds[0][1]] == ["第一节"] * 3 + ["第二节"] * 3


def test_batch_bounds_splits_oversize_section_without_crossing_it():
    from ai_agent.kb.graph_extract import _batch_bounds

    chunks = [_section_chunk(i, "第一节") for i in range(10)] + [
        _section_chunk(i, "第二节") for i in range(10, 12)
    ]
    bounds = _batch_bounds(chunks, 8)
    # 第一节占 10 片 → 满批 8 片 + 余 2 片；余量与第二节合批，不留空转批
    assert [start for start, _b in bounds] == [0, 8]
    assert [len(b) for _s, b in bounds] == [8, 4]
    # 第一节的片没有被拆到两批里（2 片连续），第二节接在后面
    assert [c.section for c in bounds[1][1]] == ["第一节"] * 2 + ["第二节"] * 2


def test_batch_bounds_falls_back_to_sequential_without_sections():
    from ai_agent.kb.graph_extract import _batch_bounds

    chunks = [_chunk(i) for i in range(20)]
    assert [start for start, _b in _batch_bounds(chunks, 8)] == [0, 8, 16]


def test_prompt_carries_section_and_page_range():
    from ai_agent.kb.chunking import Chunk
    from ai_agent.kb.graph_extract import _build_prompt

    chunks = [
        Chunk(
            text="正文一",
            page=134,
            kind="text",
            page_end=135,
            section="第十节 管理层讨论与分析",
        )
    ]
    prompt = _build_prompt(chunks, 40)
    assert "全部来自【第十节 管理层讨论与分析】" in prompt
    assert "第134–135页" in prompt
    assert "第十节 管理层讨论与分析" in prompt.split("\n\n")[1]


def test_prompt_omits_section_head_for_mixed_batch():
    from ai_agent.kb.chunking import Chunk
    from ai_agent.kb.graph_extract import _build_prompt

    chunks = [
        Chunk(text="甲", page=1, kind="text", section="第一节 释义"),
        Chunk(text="乙", page=9, kind="text", section="第二节 概览"),
    ]
    prompt = _build_prompt(chunks, 0)
    assert "全部来自" not in prompt
    # 每片自己的 section 标签仍在，模型分得清片段归属
    assert "第一节 释义" in prompt and "第二节 概览" in prompt


def test_prompt_carries_doc_name_and_span():
    """文件名必须进 prompt：表格片段里没有公司全称，模型只能靠它认主体。

    实录：安克 74 片里只有 3 片出现公司全称，某批全为现金流量表片段时，
    模型把 prompt 示例里的公司名抄成了主语，长出 14 条错误关系。
    """
    from ai_agent.kb.chunking import Chunk
    from ai_agent.kb.graph_extract import _build_prompt

    chunks = [
        Chunk(text="甲", page=18, kind="table"),
        Chunk(text="乙", page=19, kind="table"),
    ]
    prompt = _build_prompt(chunks, 65, "安克创新科技股份有限公司2026 年第三季度报告")
    head = prompt.split("\n\n")[0]
    assert head.startswith("以下是《安克创新科技股份有限公司2026 年第三季度报告》")
    assert "第 65–66 段" in head
    assert "第 18–19 页" in head


def test_prompt_drops_file_extension_and_keeps_backward_compat():
    from ai_agent.kb.chunking import Chunk
    from ai_agent.kb.graph_extract import _build_prompt

    prompt = _build_prompt([Chunk(text="甲", page=3)], 0, "2026年三季报.pdf")
    assert "《2026年三季报》" in prompt
    assert ".pdf" not in prompt
    # 不传 doc_name 时退回原文案，老调用方不受影响
    plain = _build_prompt([Chunk(text="甲", page=3)], 0)
    assert plain.startswith("以下是文档的第 0 段")
