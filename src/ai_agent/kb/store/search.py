"""检索查询：文件/切片检索、关键词命中、词频统计。（KbStore 的 mixin 之一，见 kb/store.py）。"""

from __future__ import annotations

import re
import uuid
from pgvector import Vector
from ai_agent.core.config import settings

from ai_agent.kb.store._helpers import (
    _INDICATOR_LIST,
    _SUBJECT_BUCKET,
    _chunk_meta_where,
    _keyword_hit,
)

# 方法体只需 self._get_pool()（来自 Database），不依赖其他 mixin。


class KbSearchStore:

    def search_kb_files(
        self,
        query_embedding: list[float],
        *,
        file_ids: list[uuid.UUID] | None = None,
        top_k: int = 6,
        max_distance: float | None = None,
    ) -> list[dict]:
        """文档级检索：在摘要向量上找「哪些文件整体上与这个问题相关」。

        与 search_kb_chunks 的分工：它出片段，这里出文件池。file_ids 非空时
        在这个子集里挑（会话已限定范围时不去摸范围外的文件）；max_distance
        是路由的门槛，超线的文件连候选都不该进。
        """
        emb = Vector(query_embedding)
        with self._get_pool().connection() as conn:
            return conn.execute(
                """
                SELECT id AS file_id, filename, subject, period, summary,
                       summary_embedding <=> %(emb)s AS distance
                FROM agent_kb_file
                WHERE summary_embedding IS NOT NULL
                  AND status = 'ready'
                  AND (%(file_ids)s::uuid[] IS NULL
                       OR id = ANY(%(file_ids)s::uuid[]))
                  AND (%(max_distance)s::float8 IS NULL
                       OR summary_embedding <=> %(emb)s <= %(max_distance)s)
                ORDER BY distance
                LIMIT %(top_k)s
                """,
                {
                    "emb": emb,
                    "file_ids": file_ids or None,
                    "max_distance": max_distance,
                    "top_k": top_k,
                },
            ).fetchall()


    def search_kb_chunks(
        self,
        query_embedding: list[float],
        *,
        file_ids: list[uuid.UUID] | None = None,
        top_k: int = 3,
        max_distance: float | None = None,
        kinds: list[str] | None = None,
        subjects: list[str] | None = None,
        periods: list[str] | None = None,
        scopes: list[str] | None = None,
        sections: list[str] | None = None,
    ) -> list[dict]:
        """全库检索（共享语义，不按 user_id 过滤）。

        `max_distance` 是余弦距离上限：超出的直接不返回。向量模型对无关
        内容也会给出 0.55~0.65 的基线距离，不设门槛时 top_k 会硬凑满，
        把"话题沾边但没用"的片段喂给模型，反而稀释了真正相关的内容。

        名额按主体（``subject``）保底：每个主体的头名只要与全局最佳命中
        的距离差在 ``kb_subject_quota_band`` 内，就先各占一个名额，剩下的
        再按距离竞争。这样一次调用里圈了两家公司的文档时两家都出得来
        （否则它们共用一个 query 向量，离混合向量更近的那家会把名额吃满）；
        而只问一家时，别家头名差得远就不保底，不会白白稀释上下文。
        """
        # 查询向量必须包成 Vector：裸 list 会被 psycopg 适配成
        # double precision[]，与 vector 的 <=> 运算符类型不匹配直接报错
        emb = Vector(query_embedding)
        params = {
            "emb": emb,
            # 空列表退化为 NULL（= 不过滤）：否则 ANY('{}') 恒假，
            # 调用方误传空列表会静默返回 0 条
            "file_ids": file_ids or None,
            "max_distance": max_distance,
            "top_k": top_k,
            "band": settings.kb_subject_quota_band,
        }
        # chunk 元数据过滤：列名硬编码、值参数化，无注入面；空列表/NULL 退化为不过滤
        meta_where = _chunk_meta_where(
            kinds=kinds,
            subjects=subjects,
            periods=periods,
            scopes=scopes,
            sections=sections,
            params=params,
        )
        with self._get_pool().connection() as conn:
            return conn.execute(
                f"""
                WITH hit AS (
                    SELECT c.id, c.file_id, c.seq, c.page, c.page_end, c.section,
                           c.kind, c.scope, c.text, f.filename, c.subject,
                           {_INDICATOR_LIST} AS indicators, c.period,
                           c.embedding <=> %(emb)s AS distance
                    FROM agent_kb_chunk c
                    JOIN agent_kb_file f ON f.id = c.file_id
                    WHERE (%(file_ids)s::uuid[] IS NULL
                           OR c.file_id = ANY(%(file_ids)s::uuid[]))
                      AND (%(max_distance)s::float8 IS NULL
                           OR c.embedding <=> %(emb)s <= %(max_distance)s)
                      {meta_where}
                ),
                ranked AS (
                    SELECT hit.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY {_SUBJECT_BUCKET} ORDER BY distance
                           ) AS rn,
                           MIN(distance) OVER (PARTITION BY {_SUBJECT_BUCKET})
                               AS subject_best,
                           MIN(distance) OVER () AS global_best
                    FROM hit
                )
                SELECT id, file_id, seq, page, page_end, section,
                       kind, scope, text, filename, subject, indicators, period,
                       distance
                FROM ranked
                ORDER BY (rn = 1 AND subject_best <= global_best + %(band)s) DESC,
                         distance
                LIMIT %(top_k)s
                """,
                params,
            ).fetchall()


    def search_kb_chunks_keyword(
        self,
        keywords: list[str],
        *,
        file_ids: list[uuid.UUID] | None = None,
        top_k: int = 12,
        kinds: list[str] | None = None,
        subjects: list[str] | None = None,
        periods: list[str] | None = None,
        scopes: list[str] | None = None,
        sections: list[str] | None = None,
    ) -> list[dict]:
        """关键词直查（混合检索的第二路召回）。

        向量对"制度编号、专有名词、精确表述"不敏感（同义改写会把编号
        拉远），这里按关键词子串匹配召回：命中的关键词个数越多排越前，
        同分按出现位置靠前的先出。返回字段与 search_kb_chunks 一致
        （无 distance，用 keyword_score 代替），便于上层统一融合。

        名额同样按主体保底（与向量路同一套规则，带宽为 1 个关键词）：
        关键词路也会被「营业收入」这种通用词灌满单一主体的切片。
        """
        if not keywords:
            return []
        # 每个关键词一列 CASE：命中 +1。列数受限于调用方的关键词上限
        # （默认 6 个），不会拼出超长 SQL；关键词只作参数，无注入面
        score_sql = " + ".join(_keyword_hit(i) for i in range(len(keywords)))
        params: dict = {
            "file_ids": file_ids or None,
            "top_k": top_k,
            # 关键词分是「命中了几个关键词」的整数，带宽 1 = 只给「少命中
            # 一个词以内」的主体保底；差两个词以上的主体硬塞只会灌水
            "band": 1,
        }
        for i, kw in enumerate(keywords):
            params[f"kw{i}"] = f"%{kw}%"
        # chunk 元数据过滤：列名硬编码、值参数化，无注入面；空列表/NULL 退化为不过滤
        meta_where = _chunk_meta_where(
            kinds=kinds,
            subjects=subjects,
            periods=periods,
            scopes=scopes,
            sections=sections,
            params=params,
        )
        with self._get_pool().connection() as conn:
            return conn.execute(
                f"""
                WITH hit AS (
                    SELECT c.id, c.file_id, c.seq, c.page, c.page_end, c.section,
                           c.kind, c.scope, c.text, f.filename, c.subject,
                           {_INDICATOR_LIST} AS indicators, c.period,
                           ({score_sql})::float8 AS keyword_score
                    FROM agent_kb_chunk c
                    JOIN agent_kb_file f ON f.id = c.file_id
                    WHERE (%(file_ids)s::uuid[] IS NULL
                           OR c.file_id = ANY(%(file_ids)s::uuid[]))
                      AND ({score_sql}) > 0
                      {meta_where}
                ),
                ranked AS (
                    SELECT hit.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY {_SUBJECT_BUCKET}
                               ORDER BY keyword_score DESC, file_id, seq
                           ) AS rn,
                           MAX(keyword_score) OVER (PARTITION BY {_SUBJECT_BUCKET})
                               AS subject_best,
                           MAX(keyword_score) OVER () AS global_best
                    FROM hit
                )
                SELECT id, file_id, seq, page, page_end, section,
                       kind, scope, text, filename, subject, indicators, period,
                       keyword_score
                FROM ranked
                ORDER BY (rn = 1 AND subject_best >= global_best - %(band)s) DESC,
                         keyword_score DESC, file_id, seq
                LIMIT %(top_k)s
                """,
                params,
            ).fetchall()


    def keyword_doc_freq(
        self,
        keywords: list[str],
        *,
        file_ids: list[uuid.UUID] | None = None,
    ) -> tuple[int, dict[str, int]]:
        """返回（切片总数, {关键词: 命中切片数}）。

        ILIKE 子串计数没有 IDF：「营业收入」这种词在财务表里遍地都是，
        命中不代表相关。一次聚合把 df 取出来，调用方据此只保留稀有词
        （编号、专名），关键词路才不会被通用词灌满。
        """
        if not keywords:
            return 0, {}
        cols = ", ".join(
            f"count(*) FILTER (WHERE c.text ILIKE %(kw{i})s) AS df{i}"
            for i in range(len(keywords))
        )
        params: dict = {"file_ids": file_ids or None}
        for i, kw in enumerate(keywords):
            params[f"kw{i}"] = f"%{kw}%"
        with self._get_pool().connection() as conn:
            row = conn.execute(
                f"""
                SELECT count(*) AS total, {cols}
                FROM agent_kb_chunk c
                WHERE (%(file_ids)s::uuid[] IS NULL
                       OR c.file_id = ANY(%(file_ids)s::uuid[]))
                """,
                params,
            ).fetchone()
        total = int(row["total"] or 0)
        return total, {kw: int(row[f"df{i}"] or 0) for i, kw in enumerate(keywords)}


    def keyword_boundary_freq(
        self,
        keywords: list[str],
        *,
        file_ids: list[uuid.UUID] | None = None,
    ) -> tuple[int, dict[str, int]]:
        """返回（切片总数, {关键词: 后面不再是汉字的切片数}）。

        「合并利润」在库里全是「合并利润表」的前缀，子串计数会把它当成真词。
        四个字后面还是汉字的不计；标点、括号、数字或片段结束才算。
        """
        if not keywords:
            return 0, {}
        cols = ", ".join(
            f"count(*) FILTER (WHERE c.text ~ %(kw{i})s) AS df{i}"
            for i in range(len(keywords))
        )
        params: dict = {"file_ids": file_ids or None}
        for i, kw in enumerate(keywords):
            params[f"kw{i}"] = re.escape(kw) + "([^\\u4e00-\\u9fff]|$)"
        with self._get_pool().connection() as conn:
            row = conn.execute(
                f"""
                SELECT count(*) AS total, {cols}
                FROM agent_kb_chunk c
                WHERE (%(file_ids)s::uuid[] IS NULL
                       OR c.file_id = ANY(%(file_ids)s::uuid[]))
                """,
                params,
            ).fetchone()
        total = int(row["total"] or 0)
        return total, {kw: int(row[f"df{i}"] or 0) for i, kw in enumerate(keywords)}
