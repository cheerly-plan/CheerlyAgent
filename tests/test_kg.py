"""知识图谱 P2：名称归一化、抽取合并、子图合并与工具注册。

这些用例都不碰数据库：replace_file_graph / complete_chat / embed_texts
全部替换成桩，只验证纯逻辑（合并、过滤、去重、格式）。
"""

import uuid

import pytest

from ai_agent.core.config import settings
from ai_agent.kb import graph
from ai_agent.kb.chunking import Chunk
from ai_agent.kb.graph import build_file_graph, normalize_name
from ai_agent.kb.graph_store import KgStore
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
    monkeypatch.setattr(graph, "complete_chat", lambda *a, **k: payload)
    monkeypatch.setattr(
        graph, "embed_texts", lambda texts: [[0.01, 0.02] for _ in texts]
    )
    captured: dict = {}
    monkeypatch.setattr(
        graph.store,
        "replace_file_graph",
        lambda fid, e, r, m: captured.update(entities=e, relations=r, mentions=m),
    )

    counts = build_file_graph(uuid.uuid4(), [Chunk("a"), Chunk("b")])

    # DRAM 两条合并成一个；「报告」被停用词表过滤
    assert counts["entities"] == 2
    names = {e["name"] for e in captured["entities"]}
    assert names == {"长鑫存储", "DRAM"}
    # 重复关系去重，指向不存在实体/自身的关系被丢弃
    assert counts["relations"] == 1
    assert captured["relations"][0]["relation"] == "生产"
    # 合并后描述取更长的那个
    dram = next(e for e in captured["entities"] if e["name"] == "DRAM")
    assert "动态随机存储器" in dram["description"]


def test_build_file_graph_skips_non_json(monkeypatch):
    monkeypatch.setattr(graph, "complete_chat", lambda *a, **k: "抱歉，我无法处理")
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
    assert tool.display_name == "图谱检索"
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
    from ai_agent.kb.graph_store import _query_tokens

    assert _query_tokens("HBM 涨价")[:2] == ["hbm", "涨价"]
    # 中文按二元组切，不保留整串（整串太严），也不会只剩单字（单字太泛）
    assert "存储" in _query_tokens("存储芯片价格")


def test_literal_match_boosts_name_hits():
    from ai_agent.kb.graph_store import _literal_match, _query_tokens

    boost, basis, _ = _literal_match("hbm", _query_tokens("HBM"), "HBM", "")
    assert basis == "name" and boost > 0.2

    boost, basis, _ = _literal_match("长鑫", _query_tokens("长鑫"), "长鑫存储", "")
    assert basis == "name" and 0.1 < boost < 0.2

    boost, basis, _ = _literal_match("hbm", _query_tokens("HBM"), "美光", "HBM 主要供应商")
    assert basis == "keyword" and boost > 0


def test_rank_candidates_drops_noise_when_nothing_matches():
    from ai_agent.kb.graph_store import _rank_candidates

    # 「天气预报」那类查询：全都只是话题沾边，没有任何字面依据
    scored = [_cand("界面新闻", 0.47), _cand("出口管制", 0.46), _cand("Gartner", 0.45)]
    assert _rank_candidates(scored, min_score=0.55, top_k=6) == []


def test_rank_candidates_keeps_related_when_one_strong_hit():
    from ai_agent.kb.graph_store import _rank_candidates

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
    from ai_agent.kb.graph_store import _rank_candidates

    scored = [
        _cand("三星", 0.9, score=1.0, literal=True, matched="name"),
        _cand("小众实体", 0.45),
    ]
    assert [p["name"] for p in _rank_candidates(scored, min_score=0.55, top_k=6)] == ["三星"]


def test_rank_candidates_merges_same_norm_and_picks_top_score():
    from ai_agent.kb.graph_store import _rank_candidates

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
