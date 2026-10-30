"""知识图谱构建入口：把抽取结果累加成图并落库。

从原 graph.py 拆出(S5)。切片分批与 LLM 调用在 graph_extract.py，实体名的
归一化/可用性判据在 graph_names.py，输出结构与容错解析在 graph_schema.py；
本模块只负责两件事：
- _GraphAccumulator：把各批次结果归并成实体表、关系表、提及表，
  跨批的同名实体合并、关系去重；
- build_file_graph：编排批次划分、并发调用、分批落库与进度上报。

抽取失败不抛给上层：图谱是增强能力，失败只是没有图，不能拖垮已经可检索
的文件。每 kb_graph_flush_batches 批落一次库，中断也留着部分图。
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.chunking import Chunk
from ai_agent.kb.doc_profiles import (
    DEFAULT_LEVEL,
    build_system_prompt,
    normalize_level,
    resolve_profile,
)
from ai_agent.kb.graph_extract import (
    _batch_bounds,
    _extract_all,
    _worth_extracting,
)
from ai_agent.kb.graph_names import (
    _claimed_offset,
    _clean_type,
    _locate_chunk,
    _subject_from_filename,
    _usable_name,
    display_name,
    normalize_name,
)
from ai_agent.kb.graph_schema import _Extraction, _Relation
from ai_agent.memory.embeddings import embed_texts_parallel
from ai_agent.memory.store import store

logger = get_logger(__name__)


# 泛称自称（「本公司」「本基金」「股东」…）不单独拦字典：模型把它写在
# 关系端点上时由 _resolve 改写成本文档主体全称，写在实体里时按 _usable_name
# 的通用判据处理。曾经这里维护过一份端点噪声名单，但它只服务于补桩
# （已移除，见 finalize），留着就是一份没人看的字典。

# 文档自指名。模型在关系里大量写「本公司持股…」「公司实现营业收入…」：
# 这些**既不能建实体**（一个泛称会把所有子图连成一片），**也不能直接丢**
# （丢的是整条关系）。单文件图谱里它们几乎总是指这份文件的主体，所以按
# 文件名推断出的主体全称落地——实测三家季报的「本公司」都指封面上的全称。
_SELF_REF = {
    "本公司",
    "公司",
    "该公司",
    "本集团",
    "该集团",
    "发行人",
    "上市公司",
    "本行",
    "本基金",
    "本单位",
    "本企业",
    "上述公司",
    "本公司股份",
}


class _GraphAccumulator:
    """跨批增量合并 + 分次落库。

    为什么要增量：700 多个切片是几十批、几十分钟的活。全部抽完才写库的话，
    中途超时或被重启就一点都不剩——用户感受到的是"等了一小时，白等"。
    每 N 批覆盖写一次，进程被杀也留着部分图，前端轮询的实体数还能往上走。

    合并策略与一次性合并完全一致：同名（归一化后）实体合并，type 取首次
    出现的值，mention_count 累计，描述保留更长的那个。
    """

    def __init__(self, file_id: uuid.UUID, subject: str | None = None) -> None:
        self.file_id = file_id
        # 文件主体全称（由文件名推断）：自指名端点落到它身上，见 _resolve
        self.subject_name = subject
        self.subject_norm = normalize_name(subject) if subject else None
        # 被改写的端点数与「没有主体可落、只能丢」的端点数，收尾时打一行日志
        self.self_ref_hits = 0
        self.entities: dict[str, dict] = {}
        # (source_norm, target_norm, relation) -> relation dict，天然去重
        self.relations: dict[tuple[str, str, str], dict] = {}
        self.mentions: dict[tuple[str, int], int] = {}
        # norm -> 向量：分次落库时只给新实体算向量，别整份重算
        self._vectors: dict[str, list[float]] = {}
        # 端点尚未出现的待定关系：(关系, 批起始, 批长度)
        self._pending: list[tuple[_Relation, int, int]] = []

    def add(self, start: int, batch: list[Chunk], result: _Extraction) -> None:
        # 只有整批「实体和关系都没有」才算空批。曾经这里只判 entities，
        # 于是「本批没抽到新实体、只有指向别批实体的关系」的批会被整批丢掉
        if not result.entities and not result.relations:
            return

        # 真实段号：建图前会过滤掉低信息切片，批内下标和库里的 seq 不是一回事，
        # 拿下标当 seq 写 mention，实体详情里的「原文片段」就会指到同批别的片上
        seqs = [c.seq if c.seq is not None else start + i for i, c in enumerate(batch)]
        count = len(batch)
        # 每批只归一化一次，供「这个名字出现在哪一片」的定位复用
        haystack = [normalize_name(c.text) for c in batch]

        for item in result.entities:
            norm = normalize_name(item.name)
            # 自指名不是实体：它指的就是主体，留着只会生成一个把整张图
            # 串成一片的万能节点
            if norm in _SELF_REF and self.subject_norm:
                continue
            if not _usable_name(item.name, norm):
                continue
            entity = self.entities.get(norm)
            if entity is None:
                entity = {
                    "name": display_name(item.name),
                    "norm": norm,
                    "type": _clean_type(item.type),
                    "description": (item.description or "").strip()[:200],
                }
                self.entities[norm] = entity
            else:
                if len(item.description or "") > len(entity["description"]):
                    entity["description"] = item.description.strip()[:200]
                    # 描述变了，向量也得重算（它在 _vectors 里已经过期）
                    self._vectors.pop(norm, None)
            # 提及记录该实体真正落在哪一片：先按名字在本批里找，找不到才信
            # 模型自报的位置（越界就夹回本批范围）
            offset = _locate_chunk(
                item.name, haystack, _claimed_offset(item.chunk, start, count)
            )
            key = (norm, seqs[offset])
            self.mentions[key] = self.mentions.get(key, 0) + 1

        for rel in result.relations:
            # 关系同样按端点名定位：用它报的位置，引用会指到无关的片上去
            rel.chunk = _locate_chunk(
                rel.source, haystack, _claimed_offset(rel.chunk, start, count)
            )
            if not self._add_relation(rel, seqs):
                # 端点还没出现：先挂起，等后面某一批把它抽出来再补上。
                # 直接丢会让「关系指向后文才出现的实体」整条消失——这正
                # 是「实体都在图上却互不相连」的另一个来源
                self._pending.append((rel, seqs))
        if self._pending:
            still_pending: list[tuple[_Relation, list[int]]] = []
            for item in self._pending:
                if not self._add_relation(*item):
                    still_pending.append(item)
            self._pending = still_pending

    def finalize(self) -> None:
        """收尾：把仍挂起的关系再整体试一次，端点仍不是实体的直接丢边。

        挂起队列在每批 add() 里重试，但**最后一批之后没有 add 了**——指向
        "整份文档都没抽成实体、只在关系里露过名字"的那些关系，到收尾时
        还躺在 _pending 里，flush() 不会碰它，随进程一起消失。表现是
        "关系数总比实体关系对不上"，且很难从日志里看出原因。所以收尾要
        再整体重试一次：最后几批可能刚把端点补出来。

        重试完仍然缺端点的，**丢弃整条关系，不给它补桩**。这里原来会给
        缺失端点补一个 type=other 的桩节点（描述写"仅作为关系端点出现"），
        实测三份季报补出 145 个这样的点，代价大于收益：
        - 桩没有正文依据，也没有 mention，前端点开是一片空白，检索命中后
          引用不出任何原文；
        - 名字来自模型在关系里随手写的一串字，模型笔误（截断、带标点、
          抄 prompt 示例）会被固化成图上节点，且永远不会被修正；
        - 它们撑起来的边是不可验证的——"某实体与某个查不到出处的名字之间
          存在某关系"，问一句"依据在哪"就答不上来。
        缺一条没有实体支撑的边，比留一个凭空捏出来的点更容易判断图的可信度。

        收尾是终态：连不上的就到此为止，别留在队列里假装"下次还能补"。
        """
        dropped: list[tuple[_Relation, list[int]]] = []
        for item in self._pending:
            if not self._add_relation(*item):
                dropped.append(item)
        # 收尾是终态：连不上的就到此为止，别留在队列里假装"下次还能补"
        self._pending = []
        if dropped or self.self_ref_hits:
            names = sorted({n for rel, _ in dropped for n in (rel.source, rel.target)})
            logger.info(
                "kb 图谱收尾：%d 条关系因端点未抽成实体被丢弃，自指名端点改写 %d 个"
                "（主体=%s）；缺失端点样例：%s",
                len(dropped),
                self.self_ref_hits,
                self.subject_name or "未识别",
                "、".join(names[:8]) or "无",
            )

    def _resolve(self, raw: str) -> str:
        """把自指名端点换成主体全称；换不了就原样返回（后面照常挡掉）。"""
        norm = normalize_name(raw)
        if norm in _SELF_REF and self.subject_norm:
            self._ensure_subject()
            return self.subject_norm
        return norm

    def _ensure_subject(self) -> None:
        """主体全称是图上必需的一个点：自指名端点指过来时按需补出来。

        不放在 __init__ 里预置：整份文档一条关系都没有时，凭空多出来的
        孤立主体只会拉高孤立率。描述留给模型——后续批抽到同名实体时，
        更长的描述会覆盖掉这里的默认值。
        """
        if not self.subject_norm or self.subject_norm in self.entities:
            return
        self.entities[self.subject_norm] = {
            "name": self.subject_name,
            "norm": self.subject_norm,
            "type": "org",
            "description": "本文档主体",
        }

    def _add_relation(self, rel: _Relation, seqs: list[int]) -> bool:
        """写入一条关系。端点缺失返回 False（调用方决定挂起还是丢弃）。"""
        src = self._resolve(rel.source)
        dst = self._resolve(rel.target)
        # 记「有多少条关系是靠自指名兜底才留下来的」：这就是要不要保留
        # 这套兜底的判断依据（重试与去重都走 return True，不会重复计数）
        resolved = src != normalize_name(rel.source) or dst != normalize_name(rel.target)
        # 端点查全局实体表，不查「本批」：主体常在前一批被抽出，本批只是
        # 顺带提到它（"公司向亚厦控股采购…"），只在本批内找的话这类关系
        # 会被整条丢掉，最后表现为「实体都在图上却互不相连」的孤立点
        if src == dst:
            return True  # 自环，没有意义，也不必再重试
        if src not in self.entities or dst not in self.entities:
            return False
        rel_name = (rel.relation or "相关").strip()[:20] or "相关"
        key = (src, dst, rel_name)
        if key in self.relations:
            return True
        self.relations[key] = {
            "source": src,
            "target": dst,
            "relation": rel_name,
            "description": (rel.description or "").strip()[:200],
            "chunk_seq": seqs[min(max(rel.chunk, 0), len(seqs) - 1)],
        }
        if resolved:
            self.self_ref_hits += 1
        return True

    def flush(self) -> dict:
        """把当前累积结果整文件覆盖写一次，返回计数（幂等，可反复调）。"""
        if not self.entities:
            return {"entities": 0, "relations": 0}

        norms = list(self.entities.keys())
        missing = [n for n in norms if n not in self._vectors]
        if missing:
            # 实体名+描述一起向量化：只向量化名字的话，"长鑫存储" 和
            # "DRAM 制造" 这类语义相关的实体在向量空间里挨不上
            vectors = embed_texts_parallel(
                [
                    f"{self.entities[n]['name']}（{self.entities[n]['type']}）："
                    f"{self.entities[n]['description']}"
                    for n in missing
                ],
                concurrency=settings.kb_embedding_concurrency,
            )
            self._vectors.update(zip(missing, vectors))

        entity_rows: list[dict] = []
        id_by_norm: dict[str, uuid.UUID] = {}
        mention_count: dict[str, int] = {}
        for norm, _seq in self.mentions:
            mention_count[norm] = mention_count.get(norm, 0) + 1
        for norm in norms:
            entity = self.entities[norm]
            eid = uuid.uuid5(
                uuid.NAMESPACE_URL, f"kg:{self.file_id}:{norm}:{entity['type']}"
            )
            id_by_norm[norm] = eid
            entity_rows.append(
                {
                    "id": eid,
                    "name": entity["name"],
                    "norm_name": norm,
                    "type": entity["type"],
                    "description": entity["description"],
                    "embedding": self._vectors[norm],
                    "mention_count": mention_count.get(norm, 1),
                }
            )

        relation_rows = [
            {
                "id": uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"kgr:{self.file_id}:{rel['source']}:{rel['target']}:"
                    f"{rel['relation']}",
                ),
                "source_id": id_by_norm[rel["source"]],
                "target_id": id_by_norm[rel["target"]],
                "relation": rel["relation"],
                "description": rel["description"],
                "weight": 1.0,
                "chunk_seq": rel["chunk_seq"],
            }
            for rel in self.relations.values()
        ]
        mention_rows = [
            {"entity_id": id_by_norm[norm], "chunk_seq": seq, "hits": hits}
            for (norm, seq), hits in self.mentions.items()
            if norm in id_by_norm
        ]

        store.replace_file_graph(
            self.file_id, entity_rows, relation_rows, mention_rows
        )
        return {"entities": len(entity_rows), "relations": len(relation_rows)}


def build_file_graph(
    file_id: uuid.UUID,
    chunks: list[Chunk],
    *,
    doc_type: str | None = None,
    level: str = DEFAULT_LEVEL,
    progress: Callable[[dict], None] | None = None,
    doc_name: str | None = None,
) -> dict:
    """抽取并写入某个文件的图谱，返回计数。

    doc_type 决定注入哪套文档配置（招股书/年报/法律意见书/通用…），
    未知或为空时走通用配置；level 是同一配置下的轻重粒度（完整/轻量）。
    两者都只影响 prompt，抽取与合并逻辑一份代码通吃。

    doc_name 是文件名，会写进每批 prompt 的开头。财报的表格片段通篇没有
    公司全称，没有它模型只能给「持股/每股收益」猜一个主语。

    progress 每完成一批回调一次（带累计实体数和 已完成批/总批），
    上层拿它更新 graph_status，前端就能看到百分比。
    """
    profile = resolve_profile(doc_type)
    system_prompt = build_system_prompt(profile.key, level)
    kept = [c for c in chunks if _worth_extracting(c.text)]
    if len(kept) < len(chunks):
        logger.info(
            "kb 图谱跳过 %d/%d 个低信息切片（短于 %d 字或纯数字）",
            len(chunks) - len(kept),
            len(chunks),
            settings.kb_graph_min_chunk_chars,
        )
    if not kept:
        logger.info("kb 图谱无可用切片 file_id=%s", file_id)
        return {"entities": 0, "relations": 0}

    batch_size = max(1, settings.kb_graph_batch_chunks)
    flush_every = max(1, settings.kb_graph_flush_batches)
    # 批数要按「章节组批」后的实际批数算：进度百分比和分次落库的节奏都
    # 依赖它，用 ceil(片数/批大小) 会在小章节合批后偏高
    total_batches = len(_batch_bounds(kept, batch_size))
    accumulator = _GraphAccumulator(file_id, _subject_from_filename(doc_name))
    started = time.perf_counter()

    def report(done: int) -> None:
        if progress is None:
            return
        progress(
            {
                "entities": len(accumulator.entities),
                "relations": len(accumulator.relations),
                "batches": done,
                "total_batches": total_batches,
            }
        )

    report(0)

    def sink(index: int, start: int, batch: list[Chunk], result: _Extraction) -> None:
        accumulator.add(start, batch, result)
        report(index + 1)
        if (index + 1) % flush_every == 0 and index + 1 < total_batches:
            counts = accumulator.flush()
            logger.info(
                "kb 图谱阶段写入 file_id=%s %d/%d 批 entities=%d relations=%d %.0fs",
                file_id,
                index + 1,
                total_batches,
                counts["entities"],
                counts["relations"],
                time.perf_counter() - started,
            )

    _extract_all(kept, batch_size, system_prompt, sink=sink, doc_name=doc_name)
    # 收尾必须先于 flush：挂起的关系只有在最后一批之后再试一次才可能落地，
    # 否则 flush 写出去的图里会缺掉「端点出现在更早/更晚批」的整条关系
    accumulator.finalize()
    counts = accumulator.flush()
    logger.info(
        "kb 图谱抽取完成 file_id=%s doc_type=%s level=%s batches=%d chunks=%d(过滤前 %d) "
        "实体=%d 关系=%d %.0fs",
        file_id,
        profile.key,
        normalize_level(level),
        total_batches,
        len(kept),
        len(chunks),
        counts["entities"],
        counts["relations"],
        time.perf_counter() - started,
    )
    if progress is not None:
        progress(
            {
                "entities": counts["entities"],
                "relations": counts["relations"],
                "batches": total_batches,
                "total_batches": total_batches,
            }
        )
    return counts
