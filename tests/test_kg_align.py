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


def test_number_conflict_blocks_different_versions():
    # 实测错并：「ActiveShield™4.0 温控系统」被并进了 5.0，相似度 0.927
    assert align.number_conflict("ActiveShield™4.0 温控系统", "ActiveShield™5.0 温控系统")
    assert align.number_conflict("2025年第1期A股回购股份方案", "2025年第2期A股回购股份方案")


def test_number_conflict_allows_added_year_or_period():
    # 简称补上期数/年份是正常写法，方向是"包含"而不是"对不上"
    assert not align.number_conflict("A股回购股份", "2025年第1期A股回购股份方案")
    assert not align.number_conflict("国电电力", "国电电力发展股份有限公司")


def test_candidate_pairs_skips_pairs_with_conflicting_numbers():
    a = ent("ActiveShield™4.0 温控系统", "product")
    b = ent("ActiveShield™5.0 温控系统", "product")
    sim_pairs = [{"src": a["id"], "dst": b["id"], "sim": 0.93}]
    assert align.candidate_pairs([a, b], sim_pairs, min_sim=0.86) == []


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


def test_rigid_metric_never_becomes_candidate():
    """财务科目差一个字就是另一个科目：向量再像也不能进候选。

    实测教训：库里 39 条合并有 32 条是 metric，「经营活动现金流入小计」和
    「流出小计」向量相似度 0.99，交给模型逐对裁决被判成同一个，整条科目的
    关系全串到对面去。
    """
    a = ent("经营活动现金流入小计", "metric", 5)
    b = ent("经营活动现金流出小计", "metric", 5)
    sim_pairs = [{"src": a["id"], "dst": b["id"], "sim": 0.99}]
    assert align.candidate_pairs([a, b], sim_pairs, min_sim=0.86) == []


def test_rigid_metric_skips_containment_path_too():
    """包含路径同样要拦：应付职工薪酬 ⊂ 长期应付职工薪酬，但不是同一科目。"""
    a = ent("应付职工薪酬", "metric", 9)
    b = ent("长期应付职工薪酬", "metric", 3)
    assert align.candidate_pairs([a, b], [], min_sim=0.86) == []


def test_pronoun_entities_never_paired():
    """本公司/该公司在同一份文件里可能指不同主体，并成一个会让关系挂错公司。"""
    a = ent("本公司", "org", 5)
    b = ent("该公司", "org", 2)
    sim_pairs = [{"src": a["id"], "dst": b["id"], "sim": 0.97}]
    assert align.candidate_pairs([a, b], sim_pairs, min_sim=0.86) == []


def test_org_abbreviation_still_aligned_after_rigid_rule():
    """新规则不能误伤正常的公司简称对齐。"""
    a = ent("极兔速递", "org", 4)
    b = ent("极兔速递环球有限公司", "org", 6)
    pairs = align.candidate_pairs([a, b], [], min_sim=0.86)
    assert len(pairs) == 1


def test_is_rigid_classification():
    assert align.is_rigid(ent("营业收入", "metric"))
    assert align.is_rigid(ent("本公司", "org"))
    assert align.is_rigid(ent("2005年", "other"))
    assert not align.is_rigid(ent("极兔速递", "org"))
    assert not align.is_rigid(ent("中国能源建设股份有限公司", "org"))


def test_qualifier_prefix_changes_meaning():
    """限定词在前 = 另一个东西：长期↔短期、期初↔期末、流动↔非流动。"""
    assert align.has_qualifier_prefix("应付职工薪酬", "长期应付职工薪酬")
    assert align.has_qualifier_prefix("现金及现金等价物余额", "期初现金及现金等价物余额")
    assert align.has_qualifier_prefix("非流动资产", "其他非流动资产")
    # 短名在前缀、只多出法人后缀，是同一实体，不该被这条规则拦
    assert not align.has_qualifier_prefix("国电电力", "国电电力发展股份有限公司")
    assert not align.has_qualifier_prefix("极兔速递", "极兔速递环球有限公司")


def test_qualifier_pair_never_becomes_candidate():
    """即使类型都是 other（绕过 metric 规则），限定词对也要拦住。"""
    a = ent("应付职工薪酬", "other", 9)
    b = ent("长期应付职工薪酬", "other", 3)
    assert align.candidate_pairs([a, b], [], min_sim=0.86) == []


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
    assert align.is_metric_like("2004年") and align.is_metric_like("62,200万千瓦")
    assert not align.is_metric_like("国电电力")
    assert not align.is_metric_like("2017年重大资产重组")
    assert not align.is_metric_like("装机容量")


def test_subunit_pairs_never_become_candidates():
    # 上市主体与其分公司不是同一实体：合并后母公司的关系全被电厂噪声污染
    parent = ent("国电电力发展股份有限公司", "org")
    plant = ent("国电电力发展股份有限公司朝阳发电厂", "org")
    dept = ent("中国国电集团公司安全生产部", "org")
    pairs = align.candidate_pairs([parent, plant, dept], [], min_sim=0.86)
    assert pairs == []
    assert align.looks_like_subunit(
        "国电电力发展股份有限公司", "国电电力发展股份有限公司朝阳发电厂"
    )
    # 纯法人后缀扩展不算下属单位
    assert not align.looks_like_subunit("国电电力", "国电电力发展股份有限公司")


def test_money_like_names_are_metric_like():
    # 财务附注里金额会被抽成实体，两个不同金额绝不能合并
    assert align.is_metric_like("1,017,077.00 元")
    assert align.is_metric_like("29,774,299.64 元")
    a, b = ent("1,017,077.00 元", "metric"), ent("29,774,299.64 元", "metric")
    sim_pairs = [{"src": a["id"], "dst": b["id"], "sim": 0.95}]
    assert align.candidate_pairs([a, b], sim_pairs, min_sim=0.86) == []


def test_soft_types_need_literal_containment():
    # 概念/指标类只认字面包含：模型在"东北地区 vs 华北地区"这类近邻上会判错
    a, b = ent("东北地区", "concept"), ent("华北地区", "concept")
    sim_pairs = [{"src": a["id"], "dst": b["id"], "sim": 0.93}]
    assert align.candidate_pairs([a, b], sim_pairs, min_sim=0.86) == []
    # 字面包含的仍然进候选（"每股收益" ⊂ "基本每股收益"）
    c, d = ent("每股收益", "concept"), ent("基本每股收益", "concept")
    sim_pairs = [{"src": c["id"], "dst": d["id"], "sim": 0.9}]
    assert len(align.candidate_pairs([c, d], sim_pairs, min_sim=0.86)) == 1


def test_looks_like_different_sku_separates_brand_from_models():
    """同品牌挂一串型号：字面高度相似，但都是独立产品。

    本地 7b 会把「Anker」「Anker Travel Charging Station」「Anker Universal
    Charger」连成一个簇，品牌点吞掉全部产品，图谱里就只剩一个产品。
    """
    assert align.looks_like_different_sku("anker", "ankertravelchargingstation")
    # 同品牌前缀、谁也不包含谁，也是不同型号
    assert align.looks_like_different_sku(
        "ankeruniversalcharger", "ankertravelchargingstation"
    )
    # 中间插限定词（AI 芯片 / AI 音频芯片）
    assert align.looks_like_different_sku("thus™ai芯片", "thus™ai音频芯片")
    # 型号后缀
    assert align.looks_like_different_sku("ipad充电器", "ipad充电器pro")


def test_looks_like_different_sku_allows_translated_alias():
    """一个字都不共用的同义译名要放行：拓展坞 = Hub。"""
    assert not align.looks_like_different_sku("ankerdesktopusbchub", "安克拓展坞")


def test_sku_guard_only_applies_to_product_and_technology():
    """品牌/型号判据只对产品、技术类生效，法人的简称补全不受影响。"""
    a = ent("极兔速递", "org")
    b = ent("极兔速递环球有限公司", "org")
    pairs = align.candidate_pairs([a, b], [])
    assert len(pairs) == 1  # org 仍然进候选，交给模型裁决

    c = ent("Anker", "product")
    d = ent("Anker Travel Charging Station", "product")
    assert align.candidate_pairs([c, d], []) == []
