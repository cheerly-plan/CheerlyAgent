"""知识图谱的 DB 存取。

与 KbStore 同为 mixin，挂到 MemoryStore 上共享连接池。

设计取舍：
- 实体按「文件内唯一」（file_id, norm_name, type）去重。跨文件同名实体**不合并**——
  合并需要全局实体表 + 冲突消解，而当前知识库是共享文档库，同一实体在不同文档里
  的表述和关系并不一致，合并反而会串味；前端按文件名过滤即可。
- 关系额外存 chunk_seq，前端点边可以直接定位到原文切片。
- 邻域扩展在 Python 层做 BFS 而不是递归 CTE：单文件实体量在几十到几百，
  逐层查一轮比一条复杂的双向递归 SQL 更好读、也更好限制节点预算。
"""

import re
import uuid

from pgvector import Vector


# ——— 检索打分：向量分 + 字面依据 ———
#
# 向量模型对完全不相关的词也会给 0.45~0.52 的相似度（模型基线就这么高），
# 只靠绝对阈值切不干净：搜「天气预报」照样能召回一堆芯片公司。所以在向量分
# 之上再叠一层字面证据，并让「没有字面证据」的候选自己过一道更高的门槛：
#   · 名称完全相同        +0.30（搜「HBM」命中实体「HBM」）
#   · 名称互相包含        +0.18（搜「长鑫」命中「长鑫存储」）
#   · 检索词 token 命中名称或描述  最多 +0.12
# 有字面证据的候选直接保留，避免因为描述写法不同漏掉真正相关的实体。

_STRIP_PUNCT = re.compile(
    r"""[\s\-_/·・,，。.、:：;；!！?？'"“”‘’()（）\[\]【】<>《》]+"""
)
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_WORD = re.compile(r"[a-z0-9][a-z0-9+#.]*")

NAME_BOOST_EXACT = 0.30
NAME_BOOST_PART = 0.18
KEYWORD_BOOST_MAX = 0.12
# token 命中到这个比例即视为「有字面依据」，不再受相似度门槛约束
KEYWORD_HIT_RATIO = 0.5
# 确认「问到点子上了」之后，其余候选的分数线可以比 min_score 低这么多
WEAK_SCORE_DROP = 0.12
# 同时和最强候选比：差得太远的（名字对上了但只是同话题）也一并切掉
RELATIVE_BAND = 0.3


def _norm_text(text: str) -> str:
    """与实体 norm_name 同口径：去掉空白与常见标点、转小写。"""
    return _STRIP_PUNCT.sub("", (text or "").lower())


def _query_tokens(query: str) -> list[str]:
    """检索词切词：英文数字按词切，中文按二元组（单字太泛、整串太严）。"""
    text = (query or "").lower()
    tokens = _LATIN_WORD.findall(text)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return [t for t in tokens if t]


def _literal_match(
    query_norm: str, tokens: list[str], name: str, description: str
) -> tuple[float, str, float]:
    """字面匹配：返回（加分, 依据类型, 关键词命中比例）。"""
    name_norm = _norm_text(name)
    if not query_norm or not name_norm:
        return 0.0, "", 0.0
    if query_norm == name_norm:
        return NAME_BOOST_EXACT, "name", 1.0
    if len(name_norm) >= 2 and (query_norm in name_norm or name_norm in query_norm):
        return NAME_BOOST_PART, "name", 1.0
    if not tokens:
        return 0.0, "", 0.0
    haystack = name_norm + _norm_text(description)
    ratio = sum(1 for token in tokens if token in haystack) / len(tokens)
    if ratio <= 0:
        return 0.0, "", 0.0
    return min(KEYWORD_BOOST_MAX, KEYWORD_BOOST_MAX * ratio), "keyword", ratio


def _rank_candidates(
    scored: list[dict], *, min_score: float, top_k: int
) -> list[dict]:
    """把打分后的候选挑成种子列表。

    三条规则：
    1. 整批都没有「问到点子上」的证据（字面命中，或向量分过 min_score）→ 返回空。
       向量对无关词也会给 0.45~0.52，不这样切，搜「天气预报」也能拉出一串芯片公司。
    2. 一旦确认问到了，字面命中的全留；纯向量的候选过一条双重线：
       不低于 `min_score - WEAK_SCORE_DROP`，也不比最强候选低超过 `RELATIVE_BAND`。
       前者保证「相关但描述里没写这个词」的实体不被切掉，后者砍掉只是同话题的尾部。
    3. 同一实体在每份文档里各有一条记录，按 norm_name 合并；代表记录取提及次数
       最多的那条（与全库图合并同一规则），分数取组内最高。
    """
    if not scored:
        return []
    if not any(r["literal"] or float(r["vector"]) >= min_score for r in scored):
        return []

    best_vector = max(float(r["vector"]) for r in scored)
    floor = max(min_score - WEAK_SCORE_DROP, best_vector - RELATIVE_BAND)

    grouped: dict[str, list[dict]] = {}
    for row in scored:
        if not row["literal"] and float(row["vector"]) < floor:
            continue
        grouped.setdefault(row["norm_name"] or row["name"], []).append(row)

    picked: list[dict] = []
    for group in grouped.values():
        best = max(
            group,
            key=lambda r: (int(r["mention_count"] or 0), r["name"] or ""),
        )
        top = max(group, key=lambda r: float(r["score"] or 0))
        picked.append({**best, "score": top["score"], "matched": top["matched"]})

    picked.sort(key=lambda r: float(r["score"] or 0), reverse=True)
    return picked[:top_k]


class KgStore:
    # ——— 写入 ———

    def replace_file_graph(
        self,
        file_id: uuid.UUID,
        entities: list[dict],
        relations: list[dict],
        mentions: list[dict],
    ) -> None:
        """整文件覆盖写入（先删后插，重跑幂等）。

        删除走 agent_kg_entity → relation / mention 的级联；
        另外单独删一次 relation，兼容「实体全被过滤掉但关系还在」的边界情况。
        """
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM agent_kg_relation WHERE file_id = %s", (file_id,))
                cur.execute("DELETE FROM agent_kg_entity WHERE file_id = %s", (file_id,))
                if entities:
                    cur.executemany(
                        """
                        INSERT INTO agent_kg_entity
                            (id, file_id, name, norm_name, type, description,
                             embedding, mention_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (file_id, norm_name, type) DO NOTHING
                        """,
                        [
                            (
                                e["id"],
                                file_id,
                                e["name"],
                                e["norm_name"],
                                e["type"],
                                e.get("description", ""),
                                Vector(e["embedding"]) if e.get("embedding") else None,
                                e.get("mention_count", 1),
                            )
                            for e in entities
                        ],
                    )
                if relations:
                    cur.executemany(
                        """
                        INSERT INTO agent_kg_relation
                            (id, file_id, source_id, target_id, relation,
                             description, weight, chunk_seq)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                r["id"],
                                file_id,
                                r["source_id"],
                                r["target_id"],
                                r["relation"],
                                r.get("description", ""),
                                r.get("weight", 1.0),
                                r.get("chunk_seq"),
                            )
                            for r in relations
                        ],
                    )
                if mentions:
                    cur.executemany(
                        """
                        INSERT INTO agent_kg_mention (entity_id, file_id, chunk_seq, hits)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (entity_id, chunk_seq)
                        DO UPDATE SET hits = agent_kg_mention.hits + EXCLUDED.hits
                        """,
                        [
                            (m["entity_id"], file_id, m["chunk_seq"], m.get("hits", 1))
                            for m in mentions
                        ],
                    )
            conn.commit()

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

    def apply_alignment(self, file_id: uuid.UUID, merges: list[dict]) -> dict:
        """按 merge 列表把实体并成一份：提及/关系/描述并到代表实体上。

        顺序有讲究：先把提及搬到代表实体（否则删源实体时级联删掉就丢了），
        再把关系重新指向代表，最后才删源实体（它的关系/提及随级联清空）。
        全程一个事务，中途失败不会留下"关系指向已删实体"的半成品。
        """
        if not merges:
            return {"merged": 0, "relations": 0, "aliases": 0}

        merged = 0
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                for merge in merges:
                    src = uuid.UUID(str(merge["from_id"]))
                    dst = uuid.UUID(str(merge["to_id"]))
                    cur.execute(
                        """
                        INSERT INTO agent_kg_mention (entity_id, file_id, chunk_seq, hits)
                        SELECT %s, file_id, chunk_seq, hits
                        FROM agent_kg_mention WHERE entity_id = %s
                        ON CONFLICT (entity_id, chunk_seq)
                        DO UPDATE SET hits = agent_kg_mention.hits + EXCLUDED.hits
                        """,
                        (dst, src),
                    )
                    cur.execute(
                        "UPDATE agent_kg_relation SET source_id = %s WHERE source_id = %s",
                        (dst, src),
                    )
                    cur.execute(
                        "UPDATE agent_kg_relation SET target_id = %s WHERE target_id = %s",
                        (dst, src),
                    )
                    # 描述取更长的那个、提及数累加，再删源实体
                    cur.execute(
                        """
                        UPDATE agent_kg_entity d
                        SET mention_count = d.mention_count + COALESCE(s.mention_count, 0),
                            description = CASE
                                WHEN length(COALESCE(s.description, ''))
                                     > length(COALESCE(d.description, ''))
                                THEN s.description ELSE d.description END
                        FROM agent_kg_entity s
                        WHERE d.id = %s AND s.id = %s
                        """,
                        (dst, src),
                    )
                    cur.execute(
                        """
                        INSERT INTO agent_kg_alias
                            (id, file_id, entity_id, alias_name, alias_norm, method, score)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (file_id, alias_norm) DO NOTHING
                        """,
                        (
                            uuid.uuid4(),
                            file_id,
                            dst,
                            merge["alias_name"],
                            merge["alias_norm"],
                            "llm",
                            float(merge.get("score") or 0),
                        ),
                    )
                    cur.execute("DELETE FROM agent_kg_entity WHERE id = %s", (src,))
                    merged += 1

                # 合并后会出现自环（A→B 与 B→A 同时存在）和重复边，一并清掉
                cur.execute(
                    "DELETE FROM agent_kg_relation WHERE file_id = %s AND source_id = target_id",
                    (file_id,),
                )
                cur.execute(
                    """
                    DELETE FROM agent_kg_relation a
                    USING agent_kg_relation b
                    WHERE a.file_id = %s
                      AND a.file_id = b.file_id
                      AND a.source_id = b.source_id
                      AND a.target_id = b.target_id
                      AND a.relation = b.relation
                      AND a.ctid > b.ctid
                    """,
                    (file_id,),
                )
                # 连接池是 dict_row，按列别名取，别用下标
                remaining = cur.execute(
                    "SELECT count(*) AS n FROM agent_kg_relation WHERE file_id = %s",
                    (file_id,),
                ).fetchone()["n"]
                total = cur.execute(
                    "SELECT count(*) AS n FROM agent_kg_entity WHERE file_id = %s",
                    (file_id,),
                ).fetchone()["n"]
                aliases = cur.execute(
                    "SELECT count(*) AS n FROM agent_kg_alias WHERE file_id = %s",
                    (file_id,),
                ).fetchone()["n"]
            conn.commit()
        return {
            "merged": merged,
            "entities": int(total),
            "relations": int(remaining),
            "aliases": int(aliases),
        }

    def set_align_status(
        self,
        file_id: uuid.UUID,
        status: str,
        *,
        alias_count: int | None = None,
        entity_count: int | None = None,
    ) -> None:
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                UPDATE agent_kb_file
                SET align_status = %s,
                    alias_count = COALESCE(%s, alias_count),
                    graph_entity_count = COALESCE(%s, graph_entity_count),
                    updated_at = now()
                WHERE id = %s
                """,
                (status, alias_count, entity_count, file_id),
            )
            conn.commit()

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

    def update_graph_status(
        self, file_id: uuid.UUID, status: str, *, entity_count: int | None = None
    ) -> None:
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                UPDATE agent_kb_file
                SET graph_status = %s,
                    graph_entity_count = COALESCE(%s, graph_entity_count),
                    updated_at = now()
                WHERE id = %s
                """,
                (status, entity_count, file_id),
            )
            conn.commit()

    def reset_stuck_graphs(self) -> int:
        """启动时把上次异常退出留下的 building 标记为 failed，重跑由用户触发。"""
        with self._get_pool().connection() as conn:
            cur = conn.execute(
                """
                UPDATE agent_kb_file
                SET graph_status = 'failed', updated_at = now()
                WHERE graph_status = 'building'
                """
            )
            conn.commit()
        return cur.rowcount

    # ——— 读取 ———

    def get_file_graph_status(self, file_id: uuid.UUID) -> dict:
        with self._get_pool().connection() as conn:
            row = conn.execute(
                """
                SELECT graph_status, graph_entity_count
                FROM agent_kb_file
                WHERE id = %s
                """,
                (file_id,),
            ).fetchone()
        if row is None:
            return {"graph_status": "", "entity_count": 0}
        return {
            "graph_status": row["graph_status"] or "",
            "entity_count": int(row["graph_entity_count"] or 0),
        }


    def get_file_graph(
        self, file_id: uuid.UUID, *, limit: int = 200
    ) -> dict:
        """单个文件的图谱：节点 + 边 + 实体总数。"""
        with self._get_pool().connection() as conn:
            nodes = conn.execute(
                """
                SELECT e.id, e.name, e.norm_name, e.type, e.description,
                       e.mention_count, e.file_id, f.filename
                FROM agent_kg_entity e
                JOIN agent_kb_file f ON f.id = e.file_id
                WHERE e.file_id = %s
                ORDER BY e.mention_count DESC, e.name
                LIMIT %s
                """,
                (file_id, limit),
            ).fetchall()
            edges = conn.execute(
                """
                SELECT r.id, r.source_id, r.target_id, r.relation,
                       r.description, r.weight, r.chunk_seq
                FROM agent_kg_relation r
                WHERE r.file_id = %s
                LIMIT %s
                """,
                (file_id, limit * 4),
            ).fetchall()
        return {"nodes": nodes, "edges": edges}

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
                        {
                            "emb": emb,
                            "like": like,
                            "file_ids": file_ids or None,
                            "pool": pool,
                        },
                    ).fetchall()
                )

        # 两轮查询会有重叠，先按 id 去重
        unique: dict[str, dict] = {}
        for row in rows:
            unique.setdefault(str(row["id"]), dict(row))

        scored: list[dict] = []
        for row in unique.values():
            vector = float(row["score"] or 0)
            boost, basis, ratio = _literal_match(
                query_norm, tokens, row["name"], row["description"] or ""
            )
            literal = basis == "name" or ratio >= KEYWORD_HIT_RATIO
            scored.append(
                {
                    **row,
                    "score": min(1.0, vector + boost),
                    "vector": vector,
                    "literal": literal,
                    "matched": basis or "vector",
                }
            )

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
