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
    assert pairs[0]["why"] == "suffix"


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
    # 字面包含比向量更硬，应覆盖 why；纯通用后缀给到最高分
    assert pairs[0]["why"] == "suffix"
    assert pairs[0]["score"] > 0.9


def test_containment_is_candidate_but_not_auto_accepted():
    # 实测教训：「国电电力」既是发行人简称，也是「国电电力大同发电…」的前缀，
    # 字面包含一律自动合并会把一片法人压成一个点
    a, b = ent("国电电力", "org"), ent("国电电力大同发电有限责任公司", "org")
    contain = {"a": a, "b": b, "score": 0.86, "why": "contain"}
    assert align.candidate_pairs([a, b], [], min_sim=0.86)
    assert not align.auto_accept(contain)


def test_auto_accept_only_exact_same_name():
    # 同名不同型（抽取时类型判错）可以直接合
    same = {"a": ent("国电电力", "org"), "b": ent("国电电力", "other"), "score": 0.9,
            "why": "vector"}
    assert align.auto_accept(same)

    # 高相似但差一个字的两家公司：实测 0.95 以上，绝不能自动合
    near = {"a": ent("中国华电集团公司", "org"), "b": ent("中国华能集团公司", "org"),
            "score": 0.97, "why": "vector"}
    assert not align.auto_accept(near)

    # 干脆不同名的高相似对子也不自动合，一律交模型
    high = {"a": ent("国电电力", "org"), "b": ent("国电电力股份", "org"), "score": 0.99,
            "why": "vector"}
    assert not align.auto_accept(high)


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
    # 包含关系现在要过模型裁决（不再自动合并），这里桩成"判同一实体"
    monkeypatch.setattr(align, "judge_pairs", lambda pairs: [True] * len(pairs))
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
    monkeypatch.setattr(align, "judge_pairs", lambda pairs: [True] * len(pairs))
    called = {"n": 0}
    monkeypatch.setattr(
        align.store, "apply_alignment", lambda fid, merges: called.update(n=called["n"] + 1)
    )
    result = align.align_file_graph(uuid.uuid4(), dry_run=True)
    assert result["merged"] == 1
    assert called["n"] == 0


def test_cluster_cap_cuts_chain_but_keeps_high_score_merges():
    # 前缀链把 6 个法人串起来时，不能整簇丢弃——那样连"发行人简称"这条
    # 明显正确的合并也一起丢了。按分数加边、到容量上限停住：
    # 高分对子先合，靠后的前缀链被截断，簇固定在上限内
    members = [ent(f"国电电力集团{m}有限公司", "org", 10 - m) for m in range(1, 7)]
    pairs = [
        {"a": members[i], "b": members[i + 1], "score": 0.95 - i * 0.01, "why": "vector"}
        for i in range(len(members) - 1)
    ]
    merges = align.plan_merges(members, pairs)
    targets = {m["to_id"] for m in merges}
    # 链在前 4 个处被截断：一个 4 成员的簇（3 条合并）+ 链尾自成一个 2 成员簇
    assert len(merges) == 4
    assert len(targets) == 2
    assert len({m["alias_name"] for m in merges}) == 4  # 每个成员只被合一次


def test_judge_aborts_after_consecutive_failures(monkeypatch):
    # 模型卡死时不能逐批空等：连续三次失败就整轮中止，交给上层标 failed
    from ai_agent.core.exceptions import LLMError

    monkeypatch.setattr(align, "complete_chat", lambda *a, **k: (_ for _ in ()).throw(LLMError("timeout")))
    monkeypatch.setattr(settings, "kg_align_batch", 2)
    pairs = [
        {"a": ent(f"实体{i}"), "b": ent(f"实体{i}号"), "score": 0.9, "why": "vector"}
        for i in range(10)
    ]
    try:
        align.judge_pairs(pairs)
    except align.AlignUnavailable:
        return
    raise AssertionError("连续失败应抛 AlignUnavailable")


def test_pure_suffix_extension_accepts_legal_tail_only():
    # 简称补上通用法人后缀 = 全称，字面就能定
    assert align.pure_suffix_extension("国电电力", "国电电力发展股份有限公司")
    assert align.pure_suffix_extension("国电电力", "国电电力股份")
    # 中间多了地名/厂名，是另一家法人
    assert not align.pure_suffix_extension("国电电力", "国电电力大同发电有限责任公司")
    assert not align.pure_suffix_extension("国电电力", "国电电力大连庄河发电")
    # 反过来（全称是短名的前缀）不算
    assert not align.pure_suffix_extension("国电电力发展股份有限公司", "国电电力")


def test_metric_like_pairs_are_never_merged():
    # 2005年 与 2005 年8 月 向量很近、名字互为前缀，但绝不是同一个东西
    a, b = ent("2005年", "other"), ent("2005 年8 月", "other")
    sim_pairs = [{"src": a["id"], "dst": b["id"], "sim": 0.97}]
    assert align.candidate_pairs([a, b], sim_pairs, min_sim=0.86) == []
    assert align.is_metric_like("2005年") and align.is_metric_like("12.5%")
    assert not align.is_metric_like("国电电力")
