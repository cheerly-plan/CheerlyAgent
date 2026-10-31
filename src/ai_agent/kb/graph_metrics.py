"""图谱质量指标：纯函数，输入节点/边/金标，输出可直接对比的一组数字。

为什么要单独的指标层：图谱"看起来还行"是最容易自欺的一环。实体数多了
不等于好，边多了也不等于对。真正要看的是：

- **金标召回**：文档里人一定会记下的机构/指标，图里有没有（缺了就是抽取漏了）；
- **别名合并率**：释义表里写明的简称/全称，图上是不是同一个点（P2 的效果）；
- **孤立率 / 最大连通分量**：图能不能从任意一个点走到主体（散点图没用）；
- **关系密度**：每实体平均几条边（只知道"是谁"不知道"和谁有关"就是词典不是图）。

这些都只依赖 nodes/edges/gold 三个入参，方便在改动前后各跑一次做对照。
"""

from __future__ import annotations

import re
from collections import defaultdict

_PUNCT = re.compile(r"[\s\-_/·・,，。.、:：;；!！?？'\"“”‘’()（）\[\]【】<>《》]+")


def norm(name: str) -> str:
    """与抽取侧同一套归一化口径：去标点空白、转小写。"""
    return _PUNCT.sub("", (name or "").lower())


def _index(nodes: list[dict], aliases: list[dict]) -> dict[str, str]:
    """名称 → 实体 id。别名指向代表实体，所以「国电集团」也能查到 id。"""
    idx: dict[str, str] = {}
    for node in nodes:
        keys = {norm(node.get("name") or ""), norm(node.get("norm_name") or "")}
        for key in keys:
            if key:
                idx.setdefault(key, str(node["id"]))
    for alias in aliases:
        key = norm(alias.get("alias_norm") or alias.get("alias_name") or "")
        if key:
            idx.setdefault(key, str(alias["entity_id"]))
    return idx


def adjacency(nodes: list[dict], edges: list[dict]) -> dict[str, set[str]]:
    ids = {str(n["id"]) for n in nodes}
    adj: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        s, t = str(edge["source"]), str(edge["target"])
        if s in ids and t in ids and s != t:
            adj[s].add(t)
            adj[t].add(s)
    return adj


def components(nodes: list[dict], adj: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    out: list[list[str]] = []
    for node in nodes:
        start = str(node["id"])
        if start in seen or start not in adj:
            # 孤立点也各自算一个分量，孤立率高时"最大分量占比"才有意义
            if start not in seen:
                seen.add(start)
                out.append([start])
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        out.append(group)
    return out


def _reachable(adj: dict[str, set[str]], a: str, b: str, max_hops: int = 2) -> bool:
    """两点是否在 max_hops 跳内连通：只要求"图上能连上"，不要求边名完全对。"""
    if a == b:
        return True
    frontier, seen = {a}, {a}
    for _ in range(max_hops):
        nxt: set[str] = set()
        for cur in frontier:
            for peer in adj.get(cur, ()):
                if peer == b:
                    return True
                if peer not in seen:
                    seen.add(peer)
                    nxt.add(peer)
        if not nxt:
            return False
        frontier = nxt
    return False


def evaluate(
    nodes: list[dict],
    edges: list[dict],
    gold: dict,
    aliases: list[dict] | None = None,
) -> dict:
    """算一整套指标。gold 里没给的项自动跳过（返回 None）。"""
    aliases = aliases or []
    idx = _index(nodes, aliases)
    adj = adjacency(nodes, edges)
    comps = components(nodes, adj)

    report: dict = {
        "entities": len(nodes),
        "relations": len(edges),
        "relations_per_entity": round(len(edges) / len(nodes), 2) if nodes else 0.0,
        "isolated_ratio": round(
            sum(1 for n in nodes if not adj.get(str(n["id"]))) / len(nodes), 3
        )
        if nodes
        else 0.0,
        "largest_component_ratio": round(
            max((len(c) for c in comps), default=0) / len(nodes), 3
        )
        if nodes
        else 0.0,
    }

    gold_entities: list[str] = gold.get("entities") or []
    if gold_entities:
        hits = [name for name in gold_entities if norm(name) in idx]
        misses = [name for name in gold_entities if norm(name) not in idx]
        report["entity_recall"] = round(len(hits) / len(gold_entities), 3)
        report["entity_misses"] = misses

    gold_aliases: list[list[str]] = gold.get("aliases") or []
    if gold_aliases:
        merged, unresolved, missing = [], [], []
        for pair in gold_aliases:
            a, b = pair[0], pair[1]
            ia, ib = idx.get(norm(a)), idx.get(norm(b))
            if not ia or not ib:
                missing.append(pair)
            elif ia == ib:
                merged.append(pair)
            else:
                unresolved.append(pair)
        report["alias_merged"] = len(merged)
        report["alias_recall"] = round(len(merged) / len(gold_aliases), 3)
        report["alias_unmerged"] = unresolved
        report["alias_missing"] = missing

    gold_relations: list[list[str]] = gold.get("relations") or []
    if gold_relations:
        linked, broken, absent = [], [], []
        for pair in gold_relations:
            a, b = pair[0], pair[1]
            ia, ib = idx.get(norm(a)), idx.get(norm(b))
            if not ia or not ib:
                absent.append(pair)
            elif _reachable(adj, ia, ib, max_hops=int(gold.get("max_hops", 2))):
                linked.append(pair)
            else:
                broken.append(pair)
        report["relation_linked"] = len(linked)
        report["relation_recall"] = round(len(linked) / len(gold_relations), 3)
        report["relation_unlinked"] = broken
        report["relation_absent"] = absent

    return report
