"""社区检测：把一张图切成若干"抱团的块"，给簇视图用。

为什么不是"按类型分组"：类型只有 9 个（人/机构/产品…），分完每类几十个点，
等于没分。真正有信息的是**这个文档里的业务板块**——发电资产、股东与子公司、
中介机构、监管与交易所——它们在图上的结构就是"互相连得多、跟外面连得少"，
这正是社区检测要找的东西。

为什么自己写而不用 networkx：只需要 Louvain 的第一阶段（局部移动）这一种，
几十行能写完；为一个函数引一个依赖，还要跟着升级、审计。

为什么不是更简单的标签传播：实测在"两个三角形 + 一条桥"这种对称图上，
标签传播会全场塌成一个社区（邻居票数打平后一路传染），等于没分。
Louvain 的局部移动带模块度增益判据，同样的图能干净地切出两块。

确定性：遍历顺序固定为节点 id 排序、平票取编号小的社区。同一张图每次
结果一致，前端配色不会因为刷新而变。
"""

from __future__ import annotations

# 模块度分辨率：>1 偏向切出更多小簇，<1 偏向合并成大簇。
# 1.0 是标准 Louvain 取值；文档级图谱（几十到几百点）用 1.0 的块大小刚好。
RESOLUTION = 1.0
_EPS = 1e-9


def detect_communities(
    nodes: list[dict],
    edges: list[dict],
    *,
    max_passes: int = 10,
    resolution: float = RESOLUTION,
) -> dict[str, int]:
    """返回 {node_id: community_id}；community_id 从 1 开始按规模降序编号，
    0 专用于孤立点。"""
    ids = [str(n["id"]) for n in nodes]
    adj: dict[str, dict[str, float]] = {i: {} for i in ids}
    for edge in edges:
        s, t = str(edge["source"]), str(edge["target"])
        if s in adj and t in adj and s != t:
            weight = float(edge.get("weight") or 1.0)
            adj[s][t] = adj[s].get(t, 0.0) + weight
            adj[t][s] = adj[t].get(s, 0.0) + weight

    degree = {i: sum(adj[i].values()) for i in ids}
    total = sum(degree.values())
    if total <= 0:
        return {i: 0 for i in ids}

    comm = {i: i for i in ids}
    # 社区总度数（Σ_tot）：模块度增益公式里要用
    tot: dict[str, float] = dict(degree)
    order = sorted(ids)

    for _ in range(max_passes):
        moved = False
        for cur in order:
            k_i = degree[cur]
            if k_i <= 0:
                continue
            current = comm[cur]
            # 从原社区摘出来再算增益，否则会把自己所在的边算成"归属于自己"
            tot[current] -= k_i
            weights: dict[str, float] = {}
            for peer, w in adj[cur].items():
                target = comm[peer]
                weights[target] = weights.get(target, 0.0) + w

            best = current
            best_gain = weights.get(current, 0.0) - resolution * k_i * tot[current] / total
            for candidate, weight in weights.items():
                gain = weight - resolution * k_i * tot.get(candidate, 0.0) / total
                if gain > best_gain + _EPS or (
                    abs(gain - best_gain) <= _EPS and candidate < best
                ):
                    best, best_gain = candidate, gain
            tot[best] = tot.get(best, 0.0) + k_i
            if best != current:
                comm[cur] = best
                moved = True
        if not moved:
            break

    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(comm[i], []).append(i)

    # 按规模降序重编号，0 留给孤立点（只有自己的社区算孤立）
    ranked = sorted(
        (g for g in groups.items() if len(g[1]) > 1),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    mapping: dict[str, int] = {}
    for index, (_label, members) in enumerate(ranked, start=1):
        for member in members:
            mapping[member] = index
    for i in ids:
        mapping.setdefault(i, 0)
    return mapping


def summarize(
    nodes: list[dict],
    mapping: dict[str, int],
    *,
    max_members: int = 6,
) -> list[dict]:
    """把社区整理成前端能直接渲染的簇列表（按规模降序，孤立点最后）。

    每簇给一个"代表名"：取簇内提及最多的实体，比"簇 3"这种标签有用得多。
    """
    by_id = {str(n["id"]): n for n in nodes}
    buckets: dict[int, list[dict]] = {}
    for node_id, community in mapping.items():
        node = by_id.get(node_id)
        if node is None:
            continue
        buckets.setdefault(community, []).append(node)

    out: list[dict] = []
    for community, members in buckets.items():
        # API 层的节点用 mentions 表示提及次数，库层的行用 mention_count
        members.sort(
            key=lambda n: (
                -int(n.get("mention_count") or n.get("mentions") or 0),
                n.get("name") or "",
            )
        )
        names = [m.get("name") or "" for m in members]
        out.append(
            {
                "id": community,
                "label": "未连接" if community == 0 else (names[0] if names else ""),
                "size": len(members),
                "members": names[:max_members],
                "node_ids": [str(m["id"]) for m in members],
                "isolated": community == 0,
            }
        )
    out.sort(key=lambda c: (c["isolated"], -c["size"], c["label"]))
    return out
