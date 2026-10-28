"""图谱质量指标的纯函数用例（P3 金标评测的地基）。"""

from ai_agent.kb.graph_metrics import evaluate, norm


def node(i, name):
    return {"id": i, "name": name, "norm_name": norm(name)}


def edge(s, t, rel="相关"):
    return {"id": f"{s}-{t}", "source": s, "target": t, "relation": rel}


# A—B—C 链 + 孤立点 D，用来验证连通/孤立/2 跳可达
NODES = [
    node("a", "国电电力发展股份有限公司"),
    node("b", "国电宣威发电有限责任公司"),
    node("c", "国电大渡河流域水电开发有限公司"),
    node("d", "上海证券交易所"),
]
EDGES = [edge("a", "b"), edge("b", "c")]


def test_structure_metrics():
    gold = {"entities": ["国电电力发展股份有限公司"]}
    report = evaluate(NODES, EDGES, gold)
    assert report["entities"] == 4
    assert report["relations"] == 2
    assert report["isolated_ratio"] == 0.25
    # 最大分量是 a-b-c 三个点
    assert report["largest_component_ratio"] == 0.75


def test_relation_recall_allows_two_hops():
    gold = {"relations": [["国电电力发展股份有限公司", "国电大渡河流域水电开发有限公司"]]}
    report = evaluate(NODES, EDGES, gold)
    # a 到 c 隔着 b，两跳内算连通（不苛求一条直边）
    assert report["relation_recall"] == 1.0
    assert report["relation_linked"] == 1


def test_relation_recall_reports_absent_endpoints():
    gold = {"relations": [["国电电力发展股份有限公司", "不存在的公司"]]}
    report = evaluate(NODES, EDGES, gold)
    assert report["relation_recall"] == 0.0
    assert report["relation_absent"] == [["国电电力发展股份有限公司", "不存在的公司"]]


def test_alias_hit_through_alias_table():
    nodes = [node("a", "国电电力发展股份有限公司"), node("b", "国电宣威发电有限责任公司")]
    aliases = [{"alias_name": "国电电力", "alias_norm": "国电电力", "entity_id": "a"}]
    gold = {
        "entities": ["国电电力"],
        "aliases": [["国电电力", "国电电力发展股份有限公司"]],
    }
    report = evaluate(nodes, [], gold, aliases)
    # 别名能把金标实体召回，也能判定"这两写法已是同一个点"
    assert report["entity_recall"] == 1.0
    assert report["alias_recall"] == 1.0


def test_alias_unmerged_when_two_separate_nodes():
    nodes = [node("a", "国电电力发展股份有限公司"), node("b", "国电电力")]
    gold = {"aliases": [["国电电力", "国电电力发展股份有限公司"]]}
    report = evaluate(nodes, [], gold, [])
    assert report["alias_recall"] == 0.0
    assert report["alias_unmerged"] == [["国电电力", "国电电力发展股份有限公司"]]


def test_alias_missing_when_one_side_absent():
    nodes = [node("a", "国电电力发展股份有限公司")]
    gold = {"aliases": [["国电集团", "中国国电集团公司"]]}
    report = evaluate(nodes, [], gold, [])
    assert report["alias_recall"] == 0.0
    assert report["alias_missing"] == [["国电集团", "中国国电集团公司"]]


def test_empty_graph_does_not_crash():
    report = evaluate([], [], {"entities": ["X"]})
    assert report["entities"] == 0
    assert report["entity_recall"] == 0.0
    assert report["isolated_ratio"] == 0.0
