"""社区检测（簇视图 P4）的纯函数用例。"""

from ai_agent.kb.community import detect_communities, summarize


def node(i, name, mentions=1):
    return {"id": i, "name": name, "mentions": mentions}


def edge(s, t):
    return {"source": s, "target": t}


def test_two_cliques_are_separated():
    # 两个三角形，只有一条桥：应当切成两簇
    nodes = [node(str(i), f"点{i}") for i in range(6)]
    edges = [
        edge("0", "1"), edge("1", "2"), edge("0", "2"),
        edge("3", "4"), edge("4", "5"), edge("3", "5"),
        edge("2", "3"),
    ]
    mapping = detect_communities(nodes, edges)
    assert mapping["0"] == mapping["1"] == mapping["2"]
    assert mapping["3"] == mapping["4"] == mapping["5"]
    assert mapping["0"] != mapping["3"]


def test_isolated_nodes_get_community_zero():
    nodes = [node("a", "甲"), node("b", "乙"), node("c", "丙")]
    mapping = detect_communities(nodes, [edge("a", "b")])
    assert mapping["a"] == mapping["b"] == 1
    assert mapping["c"] == 0


def test_detection_is_deterministic():
    nodes = [node(str(i), f"点{i}") for i in range(4)]
    edges = [edge("0", "1"), edge("1", "2"), edge("2", "3"), edge("3", "0")]
    first = detect_communities(nodes, edges)
    second = detect_communities(list(reversed(nodes)), list(reversed(edges)))
    assert first == second


def test_summarize_labels_cluster_by_busiest_member():
    nodes = [
        node("a", "国电电力发展股份有限公司", 30),
        node("b", "国电宣威发电有限责任公司", 5),
        node("c", "上海证券交易所", 1),
    ]
    mapping = {"a": 1, "b": 1, "c": 0}
    clusters = summarize(nodes, mapping)
    assert clusters[0]["label"] == "国电电力发展股份有限公司"
    assert clusters[0]["size"] == 2
    assert clusters[-1]["isolated"] is True and clusters[-1]["label"] == "未连接"
