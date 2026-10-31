"""知识图谱 P2：名称归一化、抽取合并、子图合并与工具注册。

这些用例都不碰数据库：replace_file_graph / complete_chat / embed_texts
全部替换成桩，只验证纯逻辑（合并、过滤、去重、格式）。
"""

import uuid

import pytest

from ai_agent.core.config import settings
from ai_agent.kb import graph, graph_extract
from ai_agent.kb.chunking import Chunk
from ai_agent.kb.graph import build_file_graph
from ai_agent.kb.graph_names import normalize_name
from ai_agent.kb.graph_schema import _Entity, _Extraction, _parse_extraction
from ai_agent.kb.graph_store import KgStore
from ai_agent.memory import parse
from ai_agent.tools.registry import TOOLS_MAP


def test_normalize_name_strips_punct_and_space():
    # 全角括号归一成半角，这样「（合肥）」与「(合肥)」能匹配上
    assert normalize_name("  长鑫存储（合肥） ") == "长鑫存储(合肥)"
    assert normalize_name("长鑫存储(合肥)") == "长鑫存储(合肥)"
    assert normalize_name(" SK 海力士 ") == "sk海力士"
    assert normalize_name("“HBM”") == "hbm"
    assert normalize_name("《存储芯片》") == "存储芯片"


def test_normalize_name_empty():
    assert normalize_name("   ") == ""
    assert normalize_name("——") == ""


def test_build_file_graph_merges_and_links(monkeypatch):
    payload = """
    {"entities":[
      {"name":"长鑫存储","type":"org","description":"国产 DRAM 厂商","chunk":0},
      {"name":" DRAM ","type":"product","description":"内存颗粒","chunk":0},
      {"name":"DRAM","type":"product","description":"动态随机存储器，容量大","chunk":1},
      {"name":"报告","type":"concept","description":"泛化词应被丢弃","chunk":0}
    ],
    "relations":[
      {"source":"长鑫存储","target":"DRAM","relation":"生产","description":"量产 DRAM","chunk":0},
      {"source":"长鑫存储","target":"DRAM","relation":"生产","chunk":1},
      {"source":"长鑫存储","target":"不存在的实体","relation":"投资","chunk":0}
    ]}
    """
    monkeypatch.setattr(graph_extract, "complete_chat", lambda *a, **k: payload)
    monkeypatch.setattr(
        graph, "embed_texts_parallel", lambda texts, **k: [[0.01, 0.02] for _ in texts]
    )
    captured: dict = {}
    monkeypatch.setattr(
        graph.store,
        "replace_file_graph",
        lambda fid, e, r, m: captured.update(entities=e, relations=r, mentions=m),
    )

    # 切片得够长：太短的碎片在进批之前就被当成低信息切片滤掉了
    body = (
        "长鑫存储是国内 DRAM 厂商，报告期内 DRAM 产能持续爬坡，"
        "产品覆盖 DDR4 与 DDR5 颗粒，客户以模组厂为主。"
    )
    counts = build_file_graph(uuid.uuid4(), [Chunk(body), Chunk(body)])

    # DRAM 两条合并成一个；「报告」被停用词表过滤；指向「不存在的实体」的
    # 那条关系连同悬空端点一起丢掉——端点不是实体就不造点（见 finalize）
    assert {e["name"] for e in captured["entities"]} == {"长鑫存储", "DRAM"}
    assert not any("仅作为关系端点" in e["description"] for e in captured["entities"])
    # 重复关系去重（同一 source/target/relation 只留一条）；「投资」已随悬空边丢弃
    assert counts["relations"] == 1
    assert {r["relation"] for r in captured["relations"]} == {"生产"}
    # 合并后描述取更长的那个
    dram = next(e for e in captured["entities"] if e["name"] == "DRAM")
    assert "动态随机存储器" in dram["description"]


def test_usable_name_drops_years_and_quantities():
    # 截图里的「2004年」、装机容量取值、金额都是限定语，不是点
    drops = (
        "2004年",
        "2017年",
        "2017年12月31日",
        "62,200万千瓦",
        "1,017,077.00 元",
        "12.5%",
        "62200",
    )
    for name in drops:
        assert not graph._usable_name(name, graph.normalize_name(name)), name
    # 指标名、带年份的事件名、产品型号要留
    keeps = ("装机容量", "营业收入", "2017年重大资产重组", "HBM3E", "5G", "国电电力")
    for name in keeps:
        assert graph._usable_name(name, graph.normalize_name(name)), name


def test_build_file_graph_drops_year_nodes_and_their_edges(monkeypatch):
    payload = """
    {"entities":[
      {"name":"发行人","type":"org","description":"国电电力","chunk":0},
      {"name":"大渡河公司","type":"org","description":"水电子公司","chunk":0},
      {"name":"2004年","type":"metric","description":"年度","chunk":0},
      {"name":"装机容量","type":"metric","description":"控股装机","chunk":0}
    ],
    "relations":[
      {"source":"发行人","target":"2004年","relation":"控股装机容量","chunk":0},
      {"source":"大渡河公司","target":"2004年","relation":"生产","chunk":0},
      {"source":"发行人","target":"装机容量","relation":"拥有","description":"2004年末","chunk":0}
    ]}
    """
    monkeypatch.setattr(graph_extract, "complete_chat", lambda *a, **k: payload)
    monkeypatch.setattr(
        graph, "embed_texts_parallel", lambda texts, **k: [[0.01, 0.02] for _ in texts]
    )
    captured: dict = {}
    monkeypatch.setattr(
        graph.store,
        "replace_file_graph",
        lambda fid, e, r, m: captured.update(entities=e, relations=r, mentions=m),
    )
    body = (
        "发行人2004年控股装机容量增长，大渡河公司当年投产，"
        "装机容量达到数万千瓦，电源结构持续优化。"
    )
    counts = graph.build_file_graph(uuid.uuid4(), [Chunk(body)])
    names = {e["name"] for e in captured["entities"]}
    assert names == {"发行人", "大渡河公司", "装机容量"}
    assert counts["entities"] == 3
    # 连到「2004年」的边随节点一起丢掉；连到指标名的边保留
    assert counts["relations"] == 1
    assert captured["relations"][0]["relation"] == "拥有"


def test_build_file_graph_skips_non_json(monkeypatch):
    monkeypatch.setattr(graph_extract, "complete_chat", lambda *a, **k: "抱歉，我无法处理")
    called = {"replace": False}
    monkeypatch.setattr(
        graph.store,
        "replace_file_graph",
        lambda *a, **k: called.update(replace=True),
    )
    counts = build_file_graph(uuid.uuid4(), [Chunk("x")])
    assert counts == {"entities": 0, "relations": 0}
    assert called["replace"] is False


def test_merge_by_norm_folds_cross_file_duplicates():
    a1, a2, b1 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    nodes = [
        {"id": a1, "norm_name": "hbm", "name": "HBM", "type": "product",
         "description": "", "mention_count": 3, "file_id": uuid.uuid4(),
         "filename": "A.pdf"},
        {"id": a2, "norm_name": "hbm", "name": "HBM", "type": "product",
         "description": "", "mention_count": 2, "file_id": uuid.uuid4(),
         "filename": "B.docx"},
        {"id": b1, "norm_name": "dram", "name": "DRAM", "type": "product",
         "description": "", "mention_count": 1, "file_id": uuid.uuid4(),
         "filename": "A.pdf"},
    ]
    edges = [
        {"id": uuid.uuid4(), "source_id": a2, "target_id": b1, "relation": "属于"},
        {"id": uuid.uuid4(), "source_id": a1, "target_id": b1, "relation": "属于"},
        {"id": uuid.uuid4(), "source_id": a1, "target_id": a2, "relation": "别名"},
    ]
    graph_data = KgStore._merge_by_norm(nodes, edges)

    assert len(graph_data["nodes"]) == 2
    hbm = next(n for n in graph_data["nodes"] if n["name"] == "HBM")
    assert hbm["mention_count"] == 5
    assert "A.pdf" in hbm["filename"] and "B.docx" in hbm["filename"]
    # 两条 A→B 的边合并成一条；自环（HBM→HBM）被丢掉
    assert len(graph_data["edges"]) == 1
    assert graph_data["edges"][0]["relation"] == "属于"


def test_kg_search_tool_registered():
    assert "kg_search" in TOOLS_MAP
    tool = TOOLS_MAP["kg_search"]
    schema = tool.schema()
    assert schema["function"]["name"] == "kg_search"


@pytest.mark.parametrize("depth", [1, 2])
def test_graph_depth_bounds_are_sane(depth):
    # 配置项存在且给的是有限跳数：跳数过大会把整张图拽出来
    assert settings.kb_graph_max_nodes >= 20
    assert depth in (1, 2)


# ——— 检索打分：向量分 + 名称/关键词依据 ———
#
# 向量模型对无关词也会给 0.45~0.52，单靠绝对阈值切不干净。这里验证「先判有没有
# 字面/强命中，再决定其余候选的分数线」这套规则。


def _cand(name, vector, *, score=None, literal=False, norm=None, mentions=1, matched="vector"):
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "norm_name": norm or name.lower(),
        "type": "org",
        "description": "",
        "mention_count": mentions,
        "filename": "a.pdf",
        "file_id": str(uuid.uuid4()),
        "score": vector if score is None else score,
        "vector": vector,
        "literal": literal,
        "matched": matched,
    }


def test_query_tokens_split_latin_and_cjk_bigram():
    from ai_agent.kb.kg_match import _query_tokens

    assert _query_tokens("HBM 涨价")[:2] == ["hbm", "涨价"]
    # 中文按二元组切，不保留整串（整串太严），也不会只剩单字（单字太泛）
    assert "存储" in _query_tokens("存储芯片价格")


def test_score_candidate_alias_is_literal_when_canonical_misses():
    """对齐删掉的「发行人」对不上全称，必须靠别名才能成为强命中。"""
    from ai_agent.kb.kg_match import _query_tokens, _score_candidate

    row = {
        "id": "e1",
        "name": "浙江亚厦装饰股份有限公司",
        "description": "",
        "score": 0.41,
    }
    tokens = _query_tokens("发行人")
    missed = _score_candidate("发行人", tokens, row)
    assert missed["literal"] is False
    assert missed["matched"] == "vector"

    scored = _score_candidate("发行人", tokens, row, alias_name="发行人")
    assert scored["literal"] is True
    assert scored["matched"] == "alias"
    assert scored["score"] > missed["score"]


def test_score_candidate_keeps_canonical_name_over_alias():
    from ai_agent.kb.kg_match import _query_tokens, _score_candidate

    row = {"id": "e1", "name": "HBM", "description": "", "score": 0.7}
    scored = _score_candidate("hbm", _query_tokens("HBM"), row, alias_name="HBM3E")
    assert scored["matched"] == "name"
    assert scored["literal"] is True


def test_literal_match_boosts_name_hits():
    from ai_agent.kb.kg_match import _literal_match, _query_tokens

    boost, basis, _ = _literal_match("hbm", _query_tokens("HBM"), "HBM", "")
    assert basis == "name" and boost > 0.2

    boost, basis, _ = _literal_match("长鑫", _query_tokens("长鑫"), "长鑫存储", "")
    assert basis == "name" and 0.1 < boost < 0.2

    boost, basis, _ = _literal_match("hbm", _query_tokens("HBM"), "美光", "HBM 主要供应商")
    assert basis == "keyword" and boost > 0


def test_rank_candidates_drops_noise_when_nothing_matches():
    from ai_agent.kb.kg_match import _rank_candidates

    # 「天气预报」那类查询：全都只是话题沾边，没有任何字面依据
    scored = [_cand("界面新闻", 0.47), _cand("出口管制", 0.46), _cand("Gartner", 0.45)]
    assert _rank_candidates(scored, min_score=0.55, top_k=6) == []


def test_rank_candidates_keeps_related_when_one_strong_hit():
    from ai_agent.kb.kg_match import _rank_candidates

    scored = [
        # 名称命中会把展示分顶到 1.0，但「相对带宽」看的是原始向量分
        _cand("HBM", 0.76, score=1.0, literal=True, matched="name"),
        _cand("SEMI 中国", 0.72, score=0.84, literal=True, matched="keyword"),
        _cand("长鑫存储", 0.52),
    ]
    picked = _rank_candidates(scored, min_score=0.55, top_k=6)
    # 有强命中说明问到了点子上，0.52 的相关实体不该被精度砍掉
    assert [p["name"] for p in picked] == ["HBM", "SEMI 中国", "长鑫存储"]


def test_rank_candidates_trims_far_tail_by_relative_band():
    from ai_agent.kb.kg_match import _rank_candidates

    scored = [
        _cand("三星", 0.9, score=1.0, literal=True, matched="name"),
        _cand("小众实体", 0.45),
    ]
    assert [p["name"] for p in _rank_candidates(scored, min_score=0.55, top_k=6)] == ["三星"]


def test_rank_candidates_merges_same_norm_and_picks_top_score():
    from ai_agent.kb.kg_match import _rank_candidates

    scored = [
        _cand("HBM", 0.72, norm="hbm", mentions=2),
        _cand("HBM", 0.80, norm="hbm", mentions=9),
        _cand("DRAM", 0.60, norm="dram", mentions=1),
    ]
    picked = _rank_candidates(scored, min_score=0.55, top_k=6)
    assert len(picked) == 2
    hbm = next(p for p in picked if p["name"] == "HBM")
    # 分数取组内最高，代表记录（提及数）也来自同一实体组
    assert hbm["score"] == 0.80
    assert hbm["mention_count"] == 9


# ——— 主体收敛：同名 metric 跨文件隔离（串公司防护）———


def _cand_scoped(name, vector, *, file_id, type, norm=None, literal=False, matched="vector", mentions=1):
    c = _cand(name, vector, norm=norm, literal=literal, matched=matched, mentions=mentions)
    c["file_id"] = file_id
    c["type"] = type
    return c


def test_rank_candidates_isolates_cross_file_metric_when_org_anchor():
    """问「安克创新 毛利率」时，顺丰的低分同名 metric 不该漏进种子。

    线上 bug：改写补全了主体「安克创新」，但顺丰文件的「2026 年第三季度」
    等 metric 因向量分过了放宽线（floor）被当成 seed，扩出「单票收入/业务量」。
    这里验证：有 org 主体锚时，跨文件的 metric 候选不享受 WEAK_SCORE_DROP 放宽，
    必须向量分 ≥ min_score 原值才保留。
    """
    from ai_agent.kb.kg_match import _rank_candidates

    ank, sf = "file-ank", "file-sf"
    scored = [
        # 主体锚：安克创新（org，字面强命中）
        _cand_scoped("安克创新科技股份有限公司", 0.85, file_id=ank, type="org", literal=True, matched="name", mentions=10),
        # 安克自己的毛利率（同文件 metric，0.60 过放宽线保留）
        _cand_scoped("毛利率", 0.60, file_id=ank, type="metric", mentions=3),
        # 顺丰的「2026 年第三季度」（跨文件 metric，0.52 < min_score 0.55，应被挡）
        _cand_scoped("2026 年第三季度", 0.52, file_id=sf, type="metric", mentions=1),
        # 顺丰的「单票收入」（跨文件 metric，0.54 < 0.55，应被挡）
        _cand_scoped("单票收入", 0.54, file_id=sf, type="metric", mentions=1),
    ]
    picked = _rank_candidates(scored, min_score=0.55, top_k=6)

    names = [p["name"] for p in picked]
    assert "安克创新科技股份有限公司" in names
    assert "毛利率" in names
    # 跨文件的低分 metric 被隔离，不再混进种子
    assert "2026 年第三季度" not in names
    assert "单票收入" not in names


def test_rank_candidates_keeps_cross_file_metric_when_above_min_score():
    """同名 metric 跨文件但分数都够高：分 file 独立保留，交 LLM 消歧，不合并。"""
    from ai_agent.kb.kg_match import _rank_candidates

    ank, sf = "file-ank", "file-sf"
    scored = [
        _cand_scoped("安克创新科技股份有限公司", 0.85, file_id=ank, type="org", literal=True, matched="name", mentions=10),
        _cand_scoped("毛利率", 0.60, file_id=ank, type="metric", mentions=3),
        # 顺丰的毛利率 0.58 ≥ 0.55，作为独立实体保留（不再与安克毛利率合并成一个）
        _cand_scoped("毛利率", 0.58, file_id=sf, type="metric", mentions=5),
    ]
    picked = _rank_candidates(scored, min_score=0.55, top_k=6)

    gross_margins = [p for p in picked if p["name"] == "毛利率"]
    # 两条「毛利率」都保留，且分属不同文件（不合并、不取错代表）
    assert len(gross_margins) == 2
    assert {p["file_id"] for p in gross_margins} == {ank, sf}


def test_rank_candidates_no_anchor_keeps_original_behavior():
    """无主体锚（只问「毛利率」没提公司）：不触发收敛，保持原合并行为。"""
    from ai_agent.kb.kg_match import _rank_candidates

    ank, sf = "file-ank", "file-sf"
    scored = [
        # 没有任何 org 字面强命中 → 无主体锚
        _cand_scoped("毛利率", 0.60, file_id=ank, type="metric", mentions=3),
        _cand_scoped("毛利率", 0.52, file_id=sf, type="metric", mentions=5),
    ]
    picked = _rank_candidates(scored, min_score=0.55, top_k=6)
    # 无主体锚时：同名仍按 norm_name 合并成一条（保持原行为），
    # 代表记录取提及数最多的（sf 那条 mentions=5）
    gross = [p for p in picked if p["name"] == "毛利率"]
    assert len(gross) == 1
    assert gross[0]["mention_count"] == 5


# ——— 抽取链路的成本与健壮性 ———


def _long_text(seed: str) -> str:
    """够长的切片，不会被低信息过滤掉。"""
    return (
        f"{seed}：报告期内公司产能持续爬坡，产品覆盖 DDR4 与 DDR5 颗粒，"
        "客户以模组厂为主，毛利率随制程升级而改善。"
    )


def test_parse_extraction_accepts_compact_arrays():
    """紧凑格式（e/r 数组）是主线：省掉逐项重复的键，输出 token 少一半。"""
    from ai_agent.kb.graph_schema import _parse_extraction

    parsed = _parse_extraction(
        {
            "e": [["长鑫存储", "org", "国产 DRAM 制造商", 1], ["DRAM", "product"]],
            "r": [["长鑫存储", "DRAM", "生产", "", 1]],
        }
    )
    assert [e.name for e in parsed.entities] == ["长鑫存储", "DRAM"]
    assert parsed.entities[0].chunk == 1
    assert parsed.entities[1].type == "product"  # 省略说明也不影响
    assert parsed.relations[0].target == "DRAM"


def test_parse_extraction_still_accepts_verbose_objects():
    from ai_agent.kb.graph_schema import _parse_extraction

    parsed = _parse_extraction(
        {"entities": [{"name": "HBM", "type": "product"}], "relations": []}
    )
    assert parsed.entities[0].name == "HBM"


def test_parse_extraction_rejects_unknown_shape():
    from ai_agent.kb.graph_schema import _parse_extraction

    assert _parse_extraction({"公司": ["发行人"]}) is None
    assert _parse_extraction("不是 dict") is None


def test_bad_output_splits_batch_and_offsets_chunk(monkeypatch):
    """上下文吃紧时模型会换个 schema：先重问，再拆半；后半的片段序号要偏移回来。

    旧实现直接把这种批当"没有实体"丢掉——静默丢人比慢更糟。
    """
    calls = []

    def fake_complete(messages, **kwargs):
        prompt = messages[-1]["content"]
        calls.append(prompt.count("[0]") + prompt.count("[1]") + prompt.count("[2]") + prompt.count("[3]"))
        if prompt.count("\n[") >= 4:  # 整批（含重问）→ 模型始终跑偏
            return '{"公司": ["发行人"], "关系": []}'
        return '{"e": [["发行人马甲公司", "org", "发行人", 0]], "r": []}'

    monkeypatch.setattr(graph_extract, "complete_chat", fake_complete)
    chunks = [Chunk(_long_text(f"片段{i}")) for i in range(4)]
    result = graph_extract._extract_batch(chunks, 0, "你是抽取器")

    # 整批失败 1 次 + 原样重问 1 次 + 两个半批各 1 次
    assert len(calls) == 4
    assert len(result.entities) == 2
    # 后半批的 chunk=0 要加上偏移，指回原批的第 2 个片段
    assert [e.chunk for e in result.entities] == [0, 2]


def test_low_value_chunks_are_filtered_before_llm(monkeypatch):
    """目录行/页码不值一次 LLM 调用，进批前就滤掉。"""
    monkeypatch.setattr(settings, "kb_graph_min_chunk_chars", 40)
    seen: list[str] = []

    def fake_extract(batch, start_seq, _depth=0, **_kw):
        seen.extend(c.text for c in batch)
        from ai_agent.kb.graph_schema import _Entity

        return _Extraction()

    monkeypatch.setattr(graph_extract, "_extract_batch", fake_extract)
    chunks = [
        Chunk("目录 ........ 12"),
        Chunk("12"),
        Chunk("（此页无正文）"),
        Chunk(_long_text("正文")),
    ]
    graph.build_file_graph(uuid.uuid4(), chunks)

    assert seen == [_long_text("正文")]


def test_graph_flushes_progressively(monkeypatch):
    """每 N 批落一次库：中断也留着部分图，progress 汇报累计计数。"""
    monkeypatch.setattr(settings, "kb_graph_batch_chunks", 1)
    monkeypatch.setattr(settings, "kb_graph_flush_batches", 1)
    monkeypatch.setattr(graph, "embed_texts_parallel", lambda texts, **k: [[0.1] for _ in texts])
    monkeypatch.setattr(
        graph_extract,
        "_extract_batch",
        lambda batch, start_seq, _depth=0, **_kw: __import__(
            "ai_agent.kb.graph", fromlist=["_Extraction"]
        )._Extraction(
            entities=[
                __import__("ai_agent.kb.graph_schema", fromlist=["_Entity"])._Entity(
                    name=f"实体{start_seq}", type="org", chunk=0
                )
            ]
        ),
    )
    writes: list[int] = []
    monkeypatch.setattr(
        graph.store,
        "replace_file_graph",
        lambda fid, e, r, m: writes.append(len(e)),
    )
    progress: list[dict] = []

    graph.build_file_graph(
        uuid.uuid4(),
        [Chunk(_long_text(f"第{i}段")) for i in range(3)],
        progress=lambda c: progress.append(c),
    )

    # 3 批各落一次（含最后一次收尾），实体数一路增长
    assert writes == [1, 2, 3]
    # 开工报 0/3，每完成一批报一次，收尾再报一次 3/3
    assert [p["batches"] for p in progress] == [0, 1, 2, 3, 3]
    assert [p["total_batches"] for p in progress] == [3, 3, 3, 3, 3]
    assert [p["entities"] for p in progress] == [0, 1, 2, 3, 3]


def test_relation_links_across_batches(monkeypatch):
    """关系端点查全局实体表，不查本批。

    招股书里主体的介绍在很前面（"发行人系…"），后面几百页都在说"发行人向
    某某采购"。关系落在后面的批、主体只在前面的批被抽成实体——只在本批内
    找端点的话，这类关系整条丢掉，最后全是孤立点（实测 407 个实体 225 个
    一条关系都没挂上）。
    """
    from ai_agent.kb.graph import _GraphAccumulator
    from ai_agent.kb.graph_schema import _Entity

    monkeypatch.setattr(
        graph, "embed_texts_parallel", lambda texts, **k: [[0.01] for _ in texts]
    )
    captured: dict = {}
    monkeypatch.setattr(
        graph.store,
        "replace_file_graph",
        lambda fid, e, r, m: captured.update(entities=e, relations=r),
    )

    acc = _GraphAccumulator(uuid.uuid4())
    # 第一批只抽到主体
    acc.add(0, [Chunk("第一段正文，发行人主营建筑装饰工程。")], _Extraction(
        entities=[_Entity(name="亚厦股份", type="org", chunk=0)],
    ))
    # 第二批的关系提到主体，但主体本身没被当实体抽出来（只在句子里被顺带提及）
    acc.add(8, [Chunk("第二段正文，亚厦股份向亚厦控股采购幕墙材料。")], _Extraction(
        entities=[_Entity(name="亚厦控股", type="org", chunk=0)],
        relations=[
            graph._Relation(source="亚厦股份", target="亚厦控股", relation="采购", chunk=0)
        ],
    ))
    acc.flush()

    assert len(captured["entities"]) == 2
    rel = captured["relations"][0]
    assert rel["relation"] == "采购"
    # chunk_seq 要落在第二批（第 8 段起），不能因为跨批就被记成 0
    assert rel["chunk_seq"] == 8


def test_doc_type_switches_config_block_in_system_prompt(monkeypatch):
    """换文档类型 = 换配置块，而不是换抽取器。

    同一份切片、同一个抽取函数，doc_type 不同时 system prompt 必须不同，
    且各自的实体范围要出现在里面；通用类型不能带上招股书的白名单。
    """
    monkeypatch.setattr(settings, "kb_graph_min_chunk_chars", 1)
    monkeypatch.setattr(graph.settings, "kb_graph_concurrency", 1)
    systems: list[str] = []

    def fake_complete(messages, **kwargs):
        systems.append(messages[0]["content"])
        # 不返回实体，避免 flush 走到向量化/落库（本用例只验 prompt 装配）
        return '{"e": [], "r": []}'

    monkeypatch.setattr(graph_extract, "complete_chat", fake_complete)
    chunks = [Chunk("国电电力发展股份有限公司向控股子公司采购煤炭，装机容量同比增长。")]

    build_file_graph(uuid.uuid4(), chunks, doc_type="prospectus", level="light")
    build_file_graph(uuid.uuid4(), chunks, doc_type="general")

    assert len(systems) == 2
    assert "募投项目" in systems[0] and "抽取粒度：轻量" in systems[0]
    assert "不限定白名单" in systems[1]
    assert "募投项目" not in systems[1]
    # 底座与输出契约两份完全一致：变的是配置，不是抽取规则
    assert "只输出一行 JSON" in systems[0] and "只输出一行 JSON" in systems[1]


def test_unknown_doc_type_falls_back_to_general(monkeypatch):
    """配置里没有的类型不能把建图搞挂：回落通用，照常抽。"""
    monkeypatch.setattr(settings, "kb_graph_min_chunk_chars", 1)
    monkeypatch.setattr(graph.settings, "kb_graph_concurrency", 1)
    seen: dict[str, str] = {}

    def fake_complete(messages, **kwargs):
        seen["system"] = messages[0]["content"]
        return '{"e": [], "r": []}'

    monkeypatch.setattr(graph_extract, "complete_chat", fake_complete)
    counts = build_file_graph(
        uuid.uuid4(), [Chunk("某份不知道类型的商业文档，提到若干主体。")], doc_type="没这个类型"
    )

    assert counts == {"entities": 0, "relations": 0}
    assert "通用商业文档" in seen["system"]


# ---------- 坏输出修复：裸词补引号 / 截断补括号 ----------


def test_repair_json_quotes_bare_tokens():
    """本地模型偶发把中文名裸着写，一行坏掉会让整批实体丢失。"""
    raw = '{"e":[["甲公司","org","",0]],"r":[["甲公司","由",乙公司,1,0]]}'
    assert parse.load_llm_json(raw) is None  # 严格模式：确实是坏 JSON
    data = parse.load_llm_json(raw, repair=True)
    assert [r[1] for r in data["r"]] == ["由"]
    assert data["r"][0][2] == "乙公司"


def test_repair_json_closes_truncated_output():
    """max_tokens 截断：括号不配平，前几条实体要救回来。"""
    raw = (
        '{"e":[["甲公司","org","",0],["乙公司","org","",1]],'
        '"r":[["甲公司","控股",""'
    )
    assert parse.load_llm_json(raw) is None
    data = parse.load_llm_json(raw, repair=True)
    assert [e[0] for e in data["e"]] == ["甲公司", "乙公司"]


def test_repair_json_keeps_outer_object_when_truncated():
    """截断时不能退化成「取内层数组」——那样 e 留下了、r 整段丢掉。"""
    raw = '{"e":[["甲公司","org","",0]],"r":[["甲公司","持股",1214365'
    data = parse.load_llm_json(raw, repair=True)
    assert isinstance(data, dict) and "r" in data


def test_repair_json_leaves_wellformed_output_alone():
    raw = '{"e":[["甲 公司","org","说明：注意 空格",0]],"r":[]}'
    assert parse.load_llm_json(raw, repair=True) == parse.load_llm_json(raw)
    # 代码块包裹照常剥掉
    fenced = '```json\n{"e":[],"r":[]}\n```'
    assert parse.load_llm_json(fenced, repair=True) == {"e": [], "r": []}


def test_repair_json_returns_none_for_prose():
    assert parse.load_llm_json("抱歉，我无法处理", repair=True) is None


# ---------- 抽取重试：先原样重问，再拆半 ----------


def _ok_payload():
    return '{"e":[["甲公司","org","",0]],"r":[]}'


def test_extract_batch_reasks_before_splitting(monkeypatch):
    """失败是随机的：同批重问一次通常就能好，比拆半便宜且不断关系。"""
    calls: list[int] = []

    def fake_complete(messages, **kwargs):
        calls.append(1)
        return _ok_payload() if len(calls) == 2 else "抱歉，我无法处理"

    monkeypatch.setattr(graph_extract, "complete_chat", fake_complete)
    chunks = [Chunk("甲公司是一家制造企业，主营若干产品线。") for _ in range(4)]
    result = graph_extract._extract_batch(chunks, 0, "system")
    assert [e.name for e in result.entities] == ["甲公司"]
    assert len(calls) == 2  # 原调用 + 重问，没有拆半


def test_extract_batch_splits_when_reask_also_fails(monkeypatch):
    calls: list[int] = []

    def fake_complete(messages, **kwargs):
        calls.append(1)
        return "抱歉，我无法处理"

    monkeypatch.setattr(graph_extract, "complete_chat", fake_complete)
    chunks = [Chunk("甲公司是一家制造企业，主营若干产品线。") for _ in range(4)]
    result = graph_extract._extract_batch(chunks, 0, "system")
    assert result.entities == []
    # 原调用 + 重问 + 拆成两半各一次；拆出来的半批不再重问，避免调用爆炸
    assert len(calls) == 4


def test_extract_batch_does_not_reask_when_split_depth_reached(monkeypatch):
    calls: list[int] = []

    def fake_complete(messages, **kwargs):
        calls.append(1)
        return "抱歉，我无法处理"

    monkeypatch.setattr(graph_extract, "complete_chat", fake_complete)
    # 2 片：拆不动，重问一次后放弃
    chunks = [Chunk("甲公司是一家制造企业。"), Chunk("乙公司是另一家制造企业。")]
    graph_extract._extract_batch(chunks, 0, "system")
    assert len(calls) == 2


# ---------- 关系行缺客体：丢掉，不猜 ----------


def test_compact_relation_missing_target_is_dropped():
    """[主体, 关系, 空/数值, 序号] 是漏了客体的写法，不能补成假关系。"""
    meta = {
        "r": [
            ["深圳明德控股", "持有股份", "", 0],
            ["深圳明德控股", "通过子公司持股", 121436506, 1],
        ]
    }
    parsed = _parse_extraction(meta)
    assert parsed.relations == []


def test_compact_relation_four_items_with_target_is_kept():
    """第 3 项是文字时是 [主体, 客体, 关系, 序号]，照常用。"""
    parsed = _parse_extraction({"r": [["甲公司", "乙公司", "控股", 0]]})
    assert len(parsed.relations) == 1
    rel = parsed.relations[0]
    assert (rel.source, rel.target, rel.relation, rel.description, rel.chunk) == (
        "甲公司",
        "乙公司",
        "控股",
        "",
        0,
    )


def test_compact_relation_five_items_unchanged():
    parsed = _parse_extraction(
        {"r": [["甲公司", "乙公司", "控股", "持股 51%", 2]]}
    )
    rel = parsed.relations[0]
    assert rel.description == "持股 51%" and rel.chunk == 2


def test_system_prompt_requires_five_item_relations():
    from ai_agent.kb.doc_profiles import build_system_prompt

    prompt = build_system_prompt(None)
    assert "r 必须是 5 项" in prompt


# ---------- 关系不能因为「端点晚出现」或「本批没实体」而丢 ----------


def _acc():
    return graph._GraphAccumulator(uuid.uuid4())


def _ext(entities=(), relations=()):
    return graph._Extraction(
        entities=[_Entity(name=n, type="org") for n in entities],
        relations=[
            graph._Relation(source=s, target=t, relation=r) for s, t, r in relations
        ],
    )


def test_batch_with_relations_but_no_entities_is_kept():
    """本批没抽到新实体、只有指向别批实体的关系时，整批不能被跳过。"""
    acc = _acc()
    batch = [Chunk("甲公司向乙公司采购原材料。")]
    acc.add(0, batch, _ext(entities=["甲公司", "乙公司"]))
    acc.add(1, batch, _ext(relations=[("甲公司", "乙公司", "采购自")]))
    assert [(r["source"], r["relation"]) for r in acc.relations.values()] == [
        ("甲公司", "采购自")
    ]


def test_relation_to_entity_in_later_batch_is_recovered():
    """端点在前一批还没出现：先挂起，等后批抽出该实体再补上。

    直接丢会让「关系指向后文才出现的实体」整条消失，表现为大量孤立点。
    """
    acc = _acc()
    batch = [Chunk("甲公司控股丙公司。")]
    acc.add(0, batch, _ext(entities=["甲公司"], relations=[("甲公司", "丙公司", "控股")]))
    assert acc.relations == {}  # 丙公司还没出现，先挂起
    acc.add(5, batch, _ext(entities=["丙公司"]))
    assert [(r["source"], r["target"], r["relation"]) for r in acc.relations.values()] == [
        ("甲公司", "丙公司", "控股")
    ]
    assert acc._pending == []  # 挂起队列已清空


def test_pending_relation_without_endpoint_entity_is_dropped_on_finalize():
    """端点始终没被抽成实体的关系，收尾时连同关系一起丢，不给它补桩。

    挂起队列只在 add() 里重试，最后一批之后没有 add 了。过去这里会给缺失
    端点补一个 type=other 的桩节点把关系保下来：三份季报因此多出 145 个
    没有正文依据、点开一片空白、检索也引用不出原文的点，而它们撑起来的
    边同样无法追问依据。现在一律丢弃——图上每条边的两端都必须是抽出来的
    实体，不能是我们替模型补出来的点。
    """
    acc = _acc()
    acc.add(0, [Chunk("甲公司投资某公司。")], _ext(entities=["甲公司"], relations=[("甲公司", "某公司", "投资")]))
    assert acc.relations == {}  # 收尾前仍然只是挂起，不提前落地
    assert len(acc._pending) == 1

    acc.finalize()
    assert acc._pending == []  # 收尾是终态，不留尾巴
    assert acc.relations == {}
    assert "某公司" not in acc.entities  # 没有被补成桩
    assert set(acc.entities) == {"甲公司"}  # 甲公司是抽出来的实体，留着


def test_no_stub_entity_reaches_the_graph(monkeypatch):
    """端到端：关系端点不是实体就丢边，不能靠造点把边保下来。

    这条盯着 finalize 的出口——上面那条只验证了 accumulator 内部状态，
    而落到库里的实体才是前端、检索、图谱问答看到的全部内容。
    """
    monkeypatch.setattr(
        graph_extract,
        "complete_chat",
        lambda *a, **k: """
        {"entities":[{"name":"长鑫存储","type":"org","description":"国产 DRAM 厂商","chunk":0}],
         "relations":[{"source":"长鑫存储","target":"某未抽出的公司","relation":"投资","chunk":0},
                      {"source":"长鑫存储","target":"DRAM","relation":"生产","chunk":0},
                      {"source":"长鑫存储","target":"DRAM","relation":"生产","chunk":0}]}
        """,
    )
    monkeypatch.setattr(
        graph, "embed_texts_parallel", lambda texts, **k: [[0.01, 0.02] for _ in texts]
    )
    captured: dict = {}
    monkeypatch.setattr(
        graph.store,
        "replace_file_graph",
        lambda fid, e, r, m: captured.update(entities=e, relations=r, mentions=m),
    )
    body = (
        "长鑫存储是国内 DRAM 厂商，报告期内 DRAM 产能持续爬坡，"
        "产品覆盖 DDR4 与 DDR5 颗粒，客户以模组厂为主。"
    )
    counts = build_file_graph(uuid.uuid4(), [Chunk(body)])

    # DRAM 只在关系里出现过，同样是端点不是实体 → 那条「生产」也被丢掉，
    # 图上一个没有依据的点都不留
    assert {e["name"] for e in captured["entities"]} == {"长鑫存储"}
    assert counts == {"entities": 1, "relations": 0}
    assert captured["relations"] == []
    assert "仅作为关系端点" not in " ".join(e["description"] for e in captured["entities"])


def test_finalize_retries_pending_once():
    """收尾先整体重试一次：最后补上的端点不该被当成缺失端点。

    挂起队列只在 add() 里重试，而重试发生在写入实体之前——最后一批抽出的
    端点赶不上这一次重试，得靠收尾再整体试一遍。
    """
    acc = _acc()
    acc.add(0, [Chunk("甲公司控股丙公司。")], _ext(entities=["甲公司"], relations=[("甲公司", "丙公司", "控股")]))
    # 模拟「最后一批补出了端点，但 add() 的重试发生在写入之前」的时序：
    # 手工把实体塞进去，再走收尾
    acc.entities["丙公司"] = {
        "name": "丙公司",
        "norm": "丙公司",
        "type": "org",
        "description": "",
    }
    acc.finalize()
    assert acc._pending == []
    assert [(r["source"], r["target"]) for r in acc.relations.values()] == [("甲公司", "丙公司")]
    assert set(acc.entities) == {"甲公司", "丙公司"}  # 没有第三个点被造出来


def test_relation_with_metric_endpoint_is_dropped_on_finalize():
    """纯数值/时间点端点不是实体：关系丢掉，也不会被当成缺失端点处理。"""
    acc = _acc()
    acc.add(0, [Chunk("甲公司2026年前三季度营收增长。")], _ext(
        entities=["甲公司"],
        relations=[("甲公司", "2026年前三季度", "报告期")],
    ))
    acc.finalize()
    assert acc.relations == {}
    assert acc._pending == []
    assert "2026年前三季度" not in acc.entities


def test_subject_from_filename_takes_full_name():
    assert (
        graph._subject_from_filename("安克创新科技股份有限公司2026 年第三季度报告.pdf")
        == "安克创新科技股份有限公司"
    )
    # 推不出公司全称就不兜底（宁可不改，也不能乱认主体）
    assert graph._subject_from_filename("行业研究方法.pdf") is None


def test_predicate_names_are_not_entities_but_compounds_are():
    """动词不能当实体，但同词根的复合名词必须留着。"""
    for verb in ("搭载", "发布", "任职", "支持", "持股", "对接", "建设", "签约"):
        assert not graph._usable_name(verb, normalize_name(verb))
    for noun in ("持股比例", "研发投入", "发布渠道", "基础设施建设"):
        assert graph._usable_name(noun, normalize_name(noun))


def test_self_reference_endpoint_lands_on_subject():
    """「本公司持股…」不能建泛称节点，也不能整条丢掉：落到主体全称上。"""
    acc = graph._GraphAccumulator(uuid.uuid4(), "安克创新科技股份有限公司")
    acc.add(
        0,
        [Chunk("本公司持有长鑫存储 30% 股权。")],
        _ext(
            entities=["长鑫存储"],
            relations=[("本公司", "长鑫存储", "持股")],
        ),
    )
    assert "本公司" not in acc.entities  # 泛称不建点
    assert [(r["source"], r["target"], r["relation"]) for r in acc.relations.values()] == [
        ("安克创新科技股份有限公司", "长鑫存储", "持股")
    ]
    assert acc.self_ref_hits == 1


def test_self_reference_without_subject_is_dropped():
    """文件名推不出主体时退回老行为：不猜，关系丢掉。"""
    acc = graph._GraphAccumulator(uuid.uuid4(), None)
    acc.add(
        0,
        [Chunk("本公司持有长鑫存储 30% 股权。")],
        _ext(entities=["长鑫存储"], relations=[("本公司", "长鑫存储", "持股")]),
    )
    acc.finalize()
    assert acc.relations == {}
    assert set(acc.entities) == {"长鑫存储"}  # 泛称不会被猜成一个点


def test_placeholder_names_from_prompt_example_are_rejected():
    """示例里的占位词不能变成实体。

    prompt 曾用真实公司名做示例，模型在缺主语的片段里直接照抄，安克文件的
    图里因此出现一个挂着 14 条关系的「顺丰控股」（安克语料 0 片提到它）。
    """
    assert not graph._usable_name("某某控股", normalize_name("某某控股"))
    assert not graph._usable_name("某某公司", normalize_name("某某公司"))
    # 真实公司名不能被这条规则误伤
    assert graph._usable_name("顺丰控股", normalize_name("顺丰控股"))


def test_build_file_graph_passes_doc_name_to_prompt(monkeypatch):
    seen: dict = {}

    def fake_complete(messages, **kwargs):
        seen["user"] = messages[1]["content"]
        return '{"e":[],"r":[]}'

    monkeypatch.setattr(graph_extract, "complete_chat", fake_complete)
    monkeypatch.setattr(
        graph, "embed_texts_parallel", lambda texts, **k: [[0.1, 0.2] for _ in texts]
    )
    monkeypatch.setattr(graph.store, "replace_file_graph", lambda *a, **k: None)

    body = "长鑫存储是国内 DRAM 厂商，报告期内 DRAM 产能持续爬坡，客户以模组厂为主。"
    build_file_graph(
        uuid.uuid4(),
        [Chunk(body)],
        doc_name="安克创新科技股份有限公司2026 年第三季度报告",
    )
    assert "《安克创新科技股份有限公司2026 年第三季度报告》" in seen["user"]


# ---------- 溯源：mention 必须指向真正含该实体的那一页 ----------


def _ext_at(entities=(), relations=(), chunk=0):
    return graph._Extraction(
        entities=[
            _Entity(name=n, type="org", chunk=chunk) for n in entities
        ],
        relations=[
            graph._Relation(source=s, target=t, relation=r, chunk=chunk)
            for s, t, r in relations
        ],
    )


def test_mention_uses_real_seq_not_filtered_index():
    """批内下标 ≠ 库里的 seq 时，mention 要写真实 seq。

    建图前会过滤掉低信息切片（短于阈值的、纯数字的），过滤后的列表下标
    整体左移。拿它当 seq 写 mention，实体详情里的「原文片段」就会指到同批
    别的片上——实测 821 条提及只有 5% 指对，「充电储能」的证据是一张限售股
    表格就是这么来的。
    """
    acc = _acc()
    batch = [
        Chunk("| 股东名称 | 期初限售股数 |", seq=28),
        Chunk("公司充电储能类产品涵盖移动充电设备及配件。", seq=33),
    ]
    acc.add(28, batch, _ext_at(entities=["充电储能"]))
    assert list(acc.mentions) == [("充电储能", 33)]


def test_mention_accepts_absolute_segment_number_from_prompt():
    """prompt 里给的是绝对段号，模型可能直接抄回来，不能当成下标用。"""
    acc = _acc()
    batch = [
        Chunk("公司充电储能类产品涵盖移动充电设备。", seq=33),
        Chunk("报告期内储能业务收入增长。", seq=34),
    ]
    # 模型报 34（绝对段号）= 批内第 1 片
    acc.add(33, batch, _ext_at(entities=["储能业务"], chunk=34))
    assert list(acc.mentions) == [("储能业务", 34)]


def test_mention_falls_back_to_model_position_when_name_absent():
    """名字在批内一个字都对不上（模型改写/别名）时，退回它报的位置。"""
    acc = _acc()
    batch = [Chunk("甲", seq=5), Chunk("乙", seq=6)]
    acc.add(5, batch, _ext_at(entities=["安克创新"], chunk=1))
    assert list(acc.mentions) == [("安克创新", 6)]


def test_relation_chunk_points_at_source_chunk():
    """关系的位置也按端点名定位，否则图谱里「关系出处」同样是错的。"""
    acc = _acc()
    batch = [
        Chunk("公司充电储能类产品涵盖移动充电设备。", seq=33),
        Chunk("安克创新科技股份有限公司持有深圳市玮顺企业管理有限公司 30% 股权。", seq=40),
    ]
    acc.add(
        33,
        batch,
        _ext_at(
            entities=["安克创新科技股份有限公司", "深圳市玮顺企业管理有限公司"],
            relations=[
                ("安克创新科技股份有限公司", "深圳市玮顺企业管理有限公司", "持股")
            ],
        ),
    )
    assert list(acc.relations.values())[0]["chunk_seq"] == 40


def test_prompt_labels_chunks_with_absolute_seq():
    """片段标号必须是绝对段号：批内下标 [0] 和开头的「第 33–34 段」互相矛盾。"""
    chunks = [Chunk("甲段正文够长可以抽", seq=33), Chunk("乙段正文够长可以抽", seq=34)]
    prompt = graph_extract._build_prompt(chunks, 33, "某公司季报")
    assert "[33] （text" in prompt
    assert "[34] （text" in prompt
    assert "[0]" not in prompt


def test_short_selection_chunk_is_not_filtered_out():
    """「……是否经过审计：☑（已勾选）否」这类短句不能被当碎片跳过。

    kb_graph_min_chunk_chars 是给页码行、目录点线用的，勾选行天生就在
    一句话以内（实测 27~36 字），但它是全文最确定的结论。三份季报里
    有 5 片因此没进图谱，问「是否经过审计」图上没有任何依据。
    """
    assert graph._worth_extracting("（一）审计意见类型：☐（未勾选）适用 ☑（已勾选）不适用")
    assert graph._worth_extracting("四、其他重要事项：☑（已勾选）适用 ☐（未勾选）不适用")
    # 没勾选记号的短碎片照样跳过
    assert not graph._worth_extracting("单位：元")
    assert not graph._worth_extracting("2026 10 29")
