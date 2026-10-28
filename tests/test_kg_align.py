"""实体对齐（图谱质量 P2）的单测：候选合成、自动判定、并查集合并。

都是纯逻辑用例，store / LLM 全部打桩——这部分算法一旦写错，
症状是"图看起来少了几个点"，在页面上极难发现，必须靠用例盯住。
"""

import uuid

from ai_agent.core.config import settings
from ai_agent.kb import align


def ent(name, type_="org", mentions=1, description=""):
    return {
        "id": uuid.uuid4(),
        "name": name,
        "norm_name": name.lower(),
        "type": type_,
        "description": description,
        "mention_count": mentions,
    }


def test_name_contained_requires_long_enough_short_name():
    # 「国电」这种 2 字词做包含判断几乎必中，不能当证据
    assert align.name_contained("国电电力", "国电电力发展股份有限公司")
    assert not align.name_contained("国电", "国电电力发展股份有限公司")
    # 双向都算
    assert align.name_contained("国电电力发展股份有限公司", "国电电力")


def test_type_compatible_allows_soft_types_only():
    assert align.type_compatible("org", "org")
    assert align.type_compatible("org", "other")
    assert align.type_compatible("concept", "product")
    # 人与公司不能因为名字像就合并
    assert not align.type_compatible("person", "org")


def test_candidate_pairs_containment_without_high_similarity():
    a = ent("国电电力发展股份有限公司", "org", 12)
    b = ent("国电电力", "org", 30)
    pairs = align.candidate_pairs([a, b], [], min_sim=0.86)
    assert len(pairs) == 1
    assert pairs[0]["why"] == "contain"


def test_candidate_pairs_skips_incompatible_type_and_far_pairs():
    a = ent("国电电力", "org")
    b = ent("国电电力集团", "person")
    far = ent("水电装机容量", "metric")
    sim_pairs = [{"src": a["id"], "dst": far["id"], "sim": 0.5}]
    assert align.candidate_pairs([a, b, far], sim_pairs, min_sim=0.86) == []


def test_candidate_pairs_dedupes_vector_and_containment():
    a = ent("国电电力", "org")
    b = ent("国电电力发展股份有限公司", "org")
    sim_pairs = [
        {"src": a["id"], "dst": b["id"], "sim": 0.88},
        {"src": b["id"], "dst": a["id"], "sim": 0.88},
    ]
    pairs = align.candidate_pairs([a, b], sim_pairs, min_sim=0.86)
    assert len(pairs) == 1
    # 字面包含比向量更硬，应覆盖 why 并抬到门槛之上
    assert pairs[0]["why"] == "contain"


def test_auto_accept_rules():
    a, b = ent("国电电力", "org"), ent("国电电力发展股份有限公司", "org")
    contain = {"a": a, "b": b, "score": 0.86, "why": "contain"}
    assert align.auto_accept(contain)

    high = {"a": a, "b": ent("国电电力股份", "org"), "score": 0.97, "why": "vector"}
    assert align.auto_accept(high)

    # 软类型（other/concept）即使很像也要问模型：抽取时类型判错太常见
    soft = {"a": ent("装机容量", "other"), "b": ent("装机容量", "other"), "score": 0.99,
            "why": "vector"}
    assert not align.auto_accept(soft)

    grey = {"a": a, "b": ent("国电电力", "org"), "score": 0.9, "why": "vector"}
    assert not align.auto_accept(grey)


def test_judge_pairs_defaults_to_false_when_model_returns_fewer(monkeypatch):
    pairs = [
        {"a": ent("A公司"), "b": ent("A股份有限公司"), "score": 0.9, "why": "vector"},
        {"a": ent("B电厂"), "b": ent("C电厂"), "score": 0.88, "why": "vector"},
    ]
    monkeypatch.setattr(align, "complete_chat", lambda *a, **k: '{"same":[1]}')
    assert align.judge_pairs(pairs) == [True, False]


def test_judge_pairs_survives_bad_output(monkeypatch):
    pairs = [{"a": ent("A"), "b": ent("B"), "score": 0.9, "why": "vector"}]
    monkeypatch.setattr(align, "complete_chat", lambda *a, **k: "我无法判断")
    assert align.judge_pairs(pairs) == [False]


def test_plan_merges_picks_busiest_as_canonical_and_is_transitive():
    a = ent("国电电力发展股份有限公司", "org", 3)
    b = ent("国电电力", "org", 20)
    c = ent("国电电力股份", "org", 5)
    pairs = [
        {"a": a, "b": b, "score": 0.9, "why": "vector"},
        {"a": b, "b": c, "score": 0.9, "why": "vector"},
    ]
    merges = align.plan_merges([a, b, c], pairs)
    assert len(merges) == 2
    assert {m["to_id"] for m in merges} == {str(b["id"])}
    assert {m["alias_name"] for m in merges} == {a["name"], c["name"]}


def test_align_file_graph_writes_merges_and_reports(monkeypatch):
    a = ent("国电电力发展股份有限公司", "org", 3)
    b = ent("国电电力", "org", 20)
    monkeypatch.setattr(align.store, "list_file_entities", lambda fid: [a, b])
    monkeypatch.setattr(
        align.store, "entity_similar_pairs", lambda fid, **k: [
            {"src": a["id"], "dst": b["id"], "sim": 0.88}
        ]
    )
    captured = {}
    monkeypatch.setattr(
        align.store, "apply_alignment", lambda fid, merges: captured.update(merges=merges)
    )
    result = align.align_file_graph(uuid.uuid4())
    assert result["merged"] == 1
    assert captured["merges"][0]["to_id"] == str(b["id"])
    # 分数要落进 alias 表，方便回看"当初凭什么合的"
    assert captured["merges"][0]["score"] >= settings.kg_align_min_sim


def test_align_file_graph_dry_run_does_not_write(monkeypatch):
    a = ent("国电电力发展股份有限公司", "org", 3)
    b = ent("国电电力", "org", 20)
    monkeypatch.setattr(align.store, "list_file_entities", lambda fid: [a, b])
    monkeypatch.setattr(align.store, "entity_similar_pairs", lambda fid, **k: [])
    called = {"n": 0}
    monkeypatch.setattr(
        align.store, "apply_alignment", lambda fid, merges: called.update(n=called["n"] + 1)
    )
    result = align.align_file_graph(uuid.uuid4(), dry_run=True)
    assert result["merged"] == 1
    assert called["n"] == 0
