"""知识图谱的查询。

从 graph_store.py 拆出(S5)。与 KgWriteStore（写入）同为 mixin，挂到
MemoryStore 上共享连接池。本模块负责取图、搜实体、邻域扩展、取实体所在
切片与统计。

设计取舍：
- 实体按「文件内唯一」（file_id, norm_name, type）去重，跨文件同名不合并；
- 关系额外存 chunk_seq，前端点边可以直接定位到原文切片；
- 邻域扩展在 Python 层做 BFS 而不是递归 CTE：单文件实体量在几十到几百，
  逐层查一轮比一条复杂的双向递归 SQL 更好读、也更好限制节点预算。
"""

from __future__ import annotations

import uuid

from pgvector import Vector

from ai_agent.kb.kg_match import (
    _ALIAS_IN_QUERY_MIN,
    _norm_text,
    _query_tokens,
    _rank_candidates,
    _score_candidate,
)


class KgStore:
    # ——— 查询 ———


    # ——— 实体对齐（图谱质量 P2） ———

    def list_file_entities(self, file_id: uuid.UUID) -> list[dict]:
        """对齐用：把一个文件的实体连同名信息一起取出（不带向量，向量在库里算）。"""
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT id, name, norm_name, type, description, mention_count
                FROM agent_kg_entity
                WHERE file_id = %s
                ORDER BY mention_count DESC
                """,
                (file_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def entity_similar_pairs(
        self,
        file_id: uuid.UUID,
        *,
        min_sim: float = 0.86,
        neighbors: int = 4,
    ) -> list[dict]:
        """文件内实体的向量近邻对（pgvector 侧算，返回 [{src,dst,sim}]）。

        每个实体只取最近的 neighbors 个：全量两两比较在 400 个实体时是 8 万对，
        而真正可能是同一实体的对子一定在最近邻里。
        """
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT a.id AS src, n.id AS dst,
                       1 - (a.embedding <=> n.embedding) AS sim
                FROM agent_kg_entity a
                CROSS JOIN LATERAL (
                    SELECT b.id, b.embedding
                    FROM agent_kg_entity b
                    WHERE b.file_id = a.file_id
                      AND b.id <> a.id
                      AND b.embedding IS NOT NULL
                    ORDER BY a.embedding <=> b.embedding
                    LIMIT %(neighbors)s
                ) n
                WHERE a.file_id = %(file_id)s
                  AND a.embedding IS NOT NULL
                  AND 1 - (a.embedding <=> n.embedding) >= %(min_sim)s
                """,
                {"file_id": file_id, "neighbors": max(1, neighbors), "min_sim": min_sim},
            ).fetchall()
        return [dict(r) for r in rows]

    def get_align_status(self, file_id: uuid.UUID) -> dict:
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT align_status, alias_count, graph_entity_count
                FROM agent_kb_file WHERE id = %s
                """,
                (file_id,),
            ).fetchone()
        if row is None:
            return {"align_status": "", "alias_count": 0, "entity_count": 0}
        return {
            "align_status": row["align_status"] or "",
            "alias_count": int(row["alias_count"] or 0),
            "entity_count": int(row["graph_entity_count"] or 0),
        }

    def list_aliases(self, file_id: uuid.UUID) -> list[dict]:
        """某个文件里被合并掉的写法（前端用来解释"这两个怎么变一个了"）。"""
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT a.alias_name, a.alias_norm, a.method, a.score,
                       e.id AS entity_id, e.name AS canonical_name, e.type
                FROM agent_kg_alias a
                JOIN agent_kg_entity e ON e.id = a.entity_id
                WHERE a.file_id = %s
                ORDER BY e.name, a.alias_name
                """,
                (file_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ——— 读取 ———

    def get_file_graph_status(self, file_id: uuid.UUID) -> dict:
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT graph_status, graph_entity_count,
                       graph_batch_done, graph_batch_total, align_status
                FROM agent_kb_file
                WHERE id = %s
                """,
                (file_id,),
            ).fetchone()
        if row is None:
            return {
                "graph_status": "",
                "entity_count": 0,
                "batch_done": 0,
                "batch_total": 0,
                "align_status": "",
            }
        return {
            "graph_status": row["graph_status"] or "",
            "entity_count": int(row["graph_entity_count"] or 0),
            "batch_done": int(row["graph_batch_done"] or 0),
            "batch_total": int(row["graph_batch_total"] or 0),
            "align_status": row["align_status"] or "",
        }


    def get_file_graph(
        self, file_id: uuid.UUID, *, limit: int = 200
    ) -> dict:
        """单个文件的图谱：节点 + 边。

        候选抓取量放大到 max(limit*3, 400)：单文件也能抽出几百个实体，
        前端力导向是 O(n²)，全量渲染会卡；显示集由枢纽分层 + 点击下钻裁。
        这里如果按 limit 截断，被提及数挡在门外的实体连下钻都展开不出来。

        边只取两端都在候选集里的，并给节点带上 degree，前端挑枢纽时不必再数一遍。
        """
        fetch = max(limit * 3, 400)
        with self._get_pool().connection() as conn:
            nodes = conn.execute(
                """
                SELECT e.id, e.name, e.norm_name, e.type, e.description,
                       e.mention_count, e.file_id, f.filename
                FROM agent_kg_entity e
                JOIN agent_kb_file f ON f.id = e.file_id
                WHERE e.file_id = %(file_id)s
                ORDER BY e.mention_count DESC, e.name
                LIMIT %(limit)s
                """,
                {"file_id": file_id, "limit": fetch},
            ).fetchall()
            if not nodes:
                return {"nodes": [], "edges": []}
            edge_rows = conn.execute(
                """
                SELECT r.id, r.source_id, r.target_id, r.relation,
                       r.description, r.weight, r.chunk_seq
                FROM agent_kg_relation r
                WHERE r.file_id = %(file_id)s
                  AND r.source_id = ANY(%(ids)s::uuid[])
                  AND r.target_id = ANY(%(ids)s::uuid[])
                LIMIT %(lim)s
                """,
                {
                    "file_id": file_id,
                    "ids": [n["id"] for n in nodes],
                    "lim": fetch * 4,
                },
            ).fetchall()
        node_list = [dict(n) for n in nodes]
        edge_list = [dict(e) for e in edge_rows]
        degree: dict[uuid.UUID, int] = {}
        for edge in edge_list:
            degree[edge["source_id"]] = degree.get(edge["source_id"], 0) + 1
            degree[edge["target_id"]] = degree.get(edge["target_id"], 0) + 1
        for node in node_list:
            node["degree"] = degree.get(node["id"], 0)
        return {"nodes": node_list, "edges": edge_list}

    def get_global_graph(
        self, *, limit: int = 160, file_ids: list[uuid.UUID] | None = None
    ) -> dict:
        """全库图谱：先取提及次数最高的实体，再取它们之间的边。

        限制节点数是因为前端力导向是 O(n²)，节点过百会明显掉帧。
        跨文件同名实体在这里合并成一个节点（详见 _merge_by_norm）。

        候选实体抓取量放大到 max(limit*3, 400)：文件多了以后「枢纽分层 +
        点击下钻」靠前端裁剪显示，这里如果也按 limit 截断，被提及数挡在
        门外的实体连下钻都展开不出来。显示集由前端决定，返回多带不伤体验。

        节点带 degree（在返回边集里的连接数）：前端据此隐藏孤独点、
        挑枢纽节点，省得再用返回的边自己数一遍。

        `file_ids` 非空时只取这些文件的实体（管理员勾选同一行业的文件
        看合并视图），LIMIT 在 WHERE 之后取，配额不会漏进范围外的文件。
        """
        fetch = max(limit * 3, 400)
        with self._get_pool().connection() as conn:
            nodes = conn.execute(
                """
                SELECT e.id, e.name, e.norm_name, e.type, e.description,
                       e.mention_count, e.file_id, f.filename
                FROM agent_kg_entity e
                JOIN agent_kb_file f ON f.id = e.file_id
                WHERE f.status = 'ready'
                  AND (%(file_ids)s::uuid[] IS NULL
                       OR e.file_id = ANY(%(file_ids)s::uuid[]))
                ORDER BY e.mention_count DESC, e.name
                LIMIT %(limit)s
                """,
                {"limit": fetch, "file_ids": file_ids or None},
            ).fetchall()
            if not nodes:
                return {"nodes": [], "edges": []}
            edge_rows = conn.execute(
                """
                SELECT r.id, r.source_id, r.target_id, r.relation,
                       r.description, r.weight, r.chunk_seq
                FROM agent_kg_relation r
                WHERE r.source_id = ANY(%(ids)s::uuid[])
                  AND r.target_id = ANY(%(ids)s::uuid[])
                LIMIT %(lim)s
                """,
                {"ids": [n["id"] for n in nodes], "lim": fetch * 4},
            ).fetchall()
        merged = self._merge_by_norm(list(nodes), list(edge_rows))
        degree: dict[uuid.UUID, int] = {}
        for edge in merged["edges"]:
            degree[edge["source_id"]] = degree.get(edge["source_id"], 0) + 1
            degree[edge["target_id"]] = degree.get(edge["target_id"], 0) + 1
        for node in merged["nodes"]:
            node["degree"] = degree.get(node["id"], 0)
        return merged

    @staticmethod
    def _merge_by_norm(nodes: list[dict], edges: list[dict]) -> dict:
        """同名实体合并：保留提及次数最高的那条做代表，其余并进来。

        知识库是共享文档库，同一家公司/技术在每份文档里各存一条实体，
        全库图上会出现多个同名节点，看起来像重复绘制；合并后按 id 重映射边。
        """
        representative: dict[str, dict] = {}
        id_map: dict[uuid.UUID, uuid.UUID] = {}
        merged: list[dict] = []
        for node in nodes:
            key = node["norm_name"]
            rep = representative.get(key)
            if rep is None:
                rep = dict(node)
                rep["mention_count"] = int(node["mention_count"] or 0)
                rep["filename"] = node.get("filename") or ""
                representative[key] = rep
                merged.append(rep)
            else:
                rep["mention_count"] += int(node["mention_count"] or 0)
                # 同一个实体可能横跨多份文档，标注出来给前端展示来源
                names = {n for n in (rep.get("filename") or "").split("、") if n}
                if node.get("filename"):
                    names.add(node["filename"])
                rep["filename"] = "、".join(sorted(names))
            id_map[node["id"]] = rep["id"]

        kept_ids = {n["id"] for n in merged}
        out_edges: list[dict] = []
        seen: set[tuple] = set()
        for edge in edges:
            src = id_map.get(edge["source_id"])
            dst = id_map.get(edge["target_id"])
            if src is None or dst is None or src == dst:
                continue
            if src not in kept_ids or dst not in kept_ids:
                continue
            key = (src, dst, edge["relation"])
            if key in seen:
                continue
            seen.add(key)
            out_edges.append({**edge, "source_id": src, "target_id": dst})
        return {"nodes": merged, "edges": out_edges}

    def search_entities(
        self,
        query_embedding: list[float],
        *,
        query: str = "",
        top_k: int = 6,
        file_ids: list[uuid.UUID] | None = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """实体检索：向量召回 + 字面依据加权。

        同一实体在每份文档里各存一条，这里按归一化名合并，否则前端会把
        「HBM」列出两三次，扩展出来的子图也是重复节点。

        合并时**代表记录选提及次数最多的那条**，与 `_merge_by_norm`（全库图
        合并）保持同一套规则；分数取组内最高，用来做阈值和排序。早前这里按
        「向量分最高」选代表，和画布上的节点 id 对不上，前端按 id 高亮就会
        整片不亮。

        `min_score` 是「强命中」的门槛：有字面依据、或向量分过线的候选算强命中；
        一条强命中都没有就整体返回空（这一问根本没问到库里的东西）。只要有一条，
        其余候选的分数线就放宽一档（`WEAK_SCORE_DROP`），避免把相关但描述里没写
        这个词的实体也切掉。

        对齐会删掉「亚厦股份」只留全称，写法进 alias 表。名称 LIKE 对不上
        「亚厦股份 ⊂ 浙江亚厦装饰…」这种非子串简称，所以第三路按别名召回。
        """
        emb = Vector(query_embedding)
        query_norm = _norm_text(query)
        tokens = _query_tokens(query)
        pool = max(top_k * 6, 24)

        columns = """
                SELECT e.id, e.name, e.norm_name, e.type, e.description, e.file_id,
                       f.filename, e.mention_count,
                       1 - (e.embedding <=> %(emb)s) AS score
                FROM agent_kg_entity e
                JOIN agent_kb_file f ON f.id = e.file_id
        """
        alias_hit: dict[str, str] = {}
        with self._get_pool().connection() as conn:
            rows = list(
                conn.execute(
                    columns
                    + """
                WHERE e.embedding IS NOT NULL
                  AND (%(file_ids)s::uuid[] IS NULL
                       OR e.file_id = ANY(%(file_ids)s::uuid[]))
                ORDER BY e.embedding <=> %(emb)s
                LIMIT %(pool)s
                """,
                    {"emb": emb, "file_ids": file_ids or None, "pool": pool},
                ).fetchall()
            )
            # 名称命中不一定在向量 top N 里（比如查询是名称的一部分，向量反而
            # 被同话题的长描述挤掉），单独按名字捞一遍再合并，保证「搜名字就能搜到」
            if query_norm:
                like = "%" + query_norm.replace("\\", "\\\\").replace("%", "\\%") + "%"
                params = {
                    "emb": emb,
                    "like": like,
                    "query_norm": query_norm,
                    "file_ids": file_ids or None,
                    "pool": pool,
                    "alias_min": _ALIAS_IN_QUERY_MIN,
                }
                rows.extend(
                    conn.execute(
                        columns
                        + """
                WHERE e.norm_name LIKE %(like)s ESCAPE '\\'
                  AND (%(file_ids)s::uuid[] IS NULL
                       OR e.file_id = ANY(%(file_ids)s::uuid[]))
                ORDER BY e.mention_count DESC
                LIMIT %(pool)s
                        """,
                        params,
                    ).fetchall()
                )
                # 别名召回：对齐后简称不在实体表里。两条互补——
                # alias LIKE 查询（搜「亚厦股份」命中别名「亚厦股份」）；
                # 查询包含别名（搜「亚厦股份的上下游」命中别名「亚厦股份」）。
                alias_rows = conn.execute(
                    """
                SELECT e.id, e.name, e.norm_name, e.type, e.description, e.file_id,
                       f.filename, e.mention_count,
                       1 - (e.embedding <=> %(emb)s) AS score,
                       a.alias_name
                FROM agent_kg_entity e
                JOIN agent_kb_file f ON f.id = e.file_id
                JOIN agent_kg_alias a ON a.entity_id = e.id
                WHERE (
                        a.alias_norm LIKE %(like)s ESCAPE '\\'
                        OR (
                            length(a.alias_norm) >= %(alias_min)s
                            AND strpos(%(query_norm)s, a.alias_norm) > 0
                        )
                      )
                  AND (%(file_ids)s::uuid[] IS NULL
                       OR e.file_id = ANY(%(file_ids)s::uuid[]))
                ORDER BY e.mention_count DESC
                LIMIT %(pool)s
                    """,
                    params,
                ).fetchall()
                rows.extend(alias_rows)
                for row in alias_rows:
                    rid = str(row["id"])
                    alias_name = str(row.get("alias_name") or "")
                    if not alias_name:
                        continue
                    prev = alias_hit.get(rid, "")
                    if not prev or _norm_text(alias_name) == query_norm:
                        alias_hit[rid] = alias_name

        # 两轮查询会有重叠，先按 id 去重
        unique: dict[str, dict] = {}
        for row in rows:
            unique.setdefault(str(row["id"]), dict(row))

        scored = [
            _score_candidate(
                query_norm, tokens, row, alias_hit.get(str(row["id"]), "")
            )
            for row in unique.values()
        ]

        # 先看这一问到底「像不像在问库里的事」，再决定留下哪些候选
        return _rank_candidates(scored, min_score=min_score, top_k=top_k)

    def entity_ids_by_norms(
        self,
        norms: list[str],
        *,
        file_ids: list[uuid.UUID] | None = None,
    ) -> dict[str, list[uuid.UUID]]:
        """规范名 → 该实体的所有记录 id（跨文件）。

        检索命中的是一个「名称」，它在每份文档里各有一条记录，正文证据也就
        散在各份文件里。只取代表记录的话，两份文件都提到的 HBM 只会回一份
        的原文，看起来像漏了。
        """
        if not norms:
            return {}
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT e.id, e.norm_name
                FROM agent_kg_entity e
                WHERE e.norm_name = ANY(%(norms)s::text[])
                  AND (%(file_ids)s::uuid[] IS NULL
                       OR e.file_id = ANY(%(file_ids)s::uuid[]))
                """,
                {"norms": norms, "file_ids": file_ids or None},
            ).fetchall()
        grouped: dict[str, list[uuid.UUID]] = {}
        for row in rows:
            grouped.setdefault(row["norm_name"], []).append(row["id"])
        return grouped

    def expand_subgraph(
        self,
        seed_ids: list[uuid.UUID],
        *,
        depth: int = 1,
        max_nodes: int = 60,
    ) -> dict:
        """从种子实体出发按跳数扩展，返回子图（节点 + 边）。

        每层只查这一层新增节点的边，节点数超预算就停，避免图爆炸。
        """
        if not seed_ids:
            return {"nodes": [], "edges": []}
        visited: dict[uuid.UUID, None] = {sid: None for sid in seed_ids}
        frontier = list(seed_ids)
        edges: list[dict] = []
        seen_edges: set[uuid.UUID] = set()

        with self._get_pool().connection() as conn:
            for _ in range(max(1, depth)):
                if not frontier:
                    break
                rows = conn.execute(
                    """
                    SELECT id, source_id, target_id, relation, description,
                           weight, chunk_seq, file_id
                    FROM agent_kg_relation
                    WHERE source_id = ANY(%(ids)s::uuid[])
                       OR target_id = ANY(%(ids)s::uuid[])
                    LIMIT %(lim)s
                    """,
                    {"ids": frontier, "lim": max_nodes * 6},
                ).fetchall()
                next_frontier: list[uuid.UUID] = []
                for row in rows:
                    if row["id"] in seen_edges:
                        continue
                    seen_edges.add(row["id"])
                    edges.append(row)
                    for side in (row["source_id"], row["target_id"]):
                        if side not in visited:
                            visited[side] = None
                            next_frontier.append(side)
                if len(visited) >= max_nodes:
                    break
                frontier = next_frontier[: max_nodes - len(visited) + len(next_frontier)]

            if not visited:
                return {"nodes": [], "edges": []}
            nodes = conn.execute(
                """
                SELECT e.id, e.name, e.norm_name, e.type, e.description,
                       e.mention_count, e.file_id, f.filename
                FROM agent_kg_entity e
                JOIN agent_kb_file f ON f.id = e.file_id
                WHERE e.id = ANY(%s::uuid[])
                """,
                (list(visited.keys()),),
            ).fetchall()
        node_ids = {n["id"] for n in nodes}
        edges = [
            e for e in edges if e["source_id"] in node_ids and e["target_id"] in node_ids
        ]
        return {"nodes": nodes, "edges": edges}

    def get_entity_chunks(
        self, entity_ids: list[uuid.UUID], *, per_entity: int = 3
    ) -> dict[str, list[dict]]:
        """实体 → 出现过的切片原文（按命中次数排序），供图谱检索返回证据。"""
        if not entity_ids:
            return {}
        with self._get_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT m.entity_id, m.chunk_seq, m.hits, c.page, c.page_end,
                       c.section, c.kind, c.text,
                       f.filename, m.file_id
                FROM agent_kg_mention m
                JOIN agent_kb_chunk c
                  ON c.file_id = m.file_id AND c.seq = m.chunk_seq
                JOIN agent_kb_file f ON f.id = m.file_id
                WHERE m.entity_id = ANY(%s::uuid[])
                ORDER BY m.hits DESC, m.chunk_seq
                """,
                (entity_ids,),
            ).fetchall()
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            bucket = grouped.setdefault(str(row["entity_id"]), [])
            if len(bucket) < per_entity:
                bucket.append(
                    {
                        "file_id": row["file_id"],
                        "chunk_seq": row["chunk_seq"],
                        "page": row["page"],
                        "page_end": row.get("page_end"),
                        "section": row.get("section"),
                        "kind": row["kind"],
                        "text": row["text"],
                        "filename": row["filename"],
                    }
                )
        return grouped

    def graph_stats(self, *, file_ids: list[uuid.UUID] | None = None) -> dict:
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT
                    (SELECT count(*) FROM agent_kg_entity e
                     WHERE %(file_ids)s::uuid[] IS NULL
                        OR e.file_id = ANY(%(file_ids)s::uuid[])) AS entities,
                    (SELECT count(*) FROM agent_kg_relation r
                     WHERE %(file_ids)s::uuid[] IS NULL
                        OR r.file_id = ANY(%(file_ids)s::uuid[])) AS relations,
                    (SELECT count(DISTINCT file_id) FROM agent_kg_entity e
                     WHERE %(file_ids)s::uuid[] IS NULL
                        OR e.file_id = ANY(%(file_ids)s::uuid[])) AS files
                """,
                {"file_ids": file_ids or None},
            ).fetchone()
        return {
            "entities": int(row["entities"]),
            "relations": int(row["relations"]),
            "files": int(row["files"]),
        }
