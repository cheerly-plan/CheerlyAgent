"""知识图谱的写入与维护。

从 graph_store.py 拆出(S5)。与 KgStore（查询）同为 mixin，挂到 MemoryStore
上共享连接池。本模块负责把抽取/对齐的结果写回图谱表：整图替换、应用实体
对齐合并、更新构建与对齐状态、清理卡住的构建任务。

跨文件同名实体**不合并**：合并需要全局实体表 + 冲突消解，而当前知识库是共享
文档库，同一实体在不同文档里的表述和关系并不一致，合并反而会串味。
"""

from __future__ import annotations

import uuid

from pgvector import Vector


class KgWriteStore:
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

    def update_graph_status(
        self,
        file_id: uuid.UUID,
        status: str,
        *,
        entity_count: int | None = None,
        batch_done: int | None = None,
        batch_total: int | None = None,
    ) -> None:
        with self._get_pool().connection() as conn:
            conn.execute(
                """
                UPDATE agent_kb_file
                SET graph_status = %s,
                    graph_entity_count = COALESCE(%s, graph_entity_count),
                    graph_batch_done = COALESCE(%s, graph_batch_done),
                    graph_batch_total = COALESCE(%s, graph_batch_total),
                    updated_at = now()
                WHERE id = %s
                """,
                (status, entity_count, batch_done, batch_total, file_id),
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
