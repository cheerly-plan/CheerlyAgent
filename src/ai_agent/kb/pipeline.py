"""知识库后台处理管道：解析 → 切片 → 批量嵌入 → 灌库。

单进程部署，沿用 turn-finalizer 的线程池模式：lifespan 启动时
recover_stuck()，关停时 shutdown() 等待在途任务完成。进度即
agent_kb_file.status（parsing → embedding → ready / failed），
前端轮询文件列表即可看到，不单独建 job 表。
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.align import align_file_graph
from ai_agent.kb.chunking import Chunk, chunk_segments, embedding_input
from ai_agent.kb.graph import build_file_graph
from ai_agent.kb.parsing import parse_segments
from ai_agent.memory.embeddings import embed_texts_parallel
from ai_agent.memory.store import store

logger = get_logger(__name__)


class KbPipeline:
    """两条独立线程池：解析/向量化（短任务）与图谱抽取（长任务）。

    分池是必要的：一次图谱抽取是几十分钟的活，跟解析/向量化挤同一个池子时，
    管理员点了「建图」之后上传的新文件只能排队等它跑完——用户看到的现象是
    「切片生成也变慢了」，实际是队列被长任务占住了。
    """

    def __init__(self, workers: int = 2, graph_workers: int = 1) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="kb-pipeline"
        )
        self._graph_pool = ThreadPoolExecutor(
            max_workers=max(1, graph_workers), thread_name_prefix="kb-graph"
        )

    def recover_stuck(self) -> None:
        stuck = store.recover_stuck_kb_files()
        if stuck:
            logger.warning("kb 启动恢复：重置 %d 个中断文件为 failed", stuck)
        # 图谱构建是切片 ready 之后的增强步骤，卡住只标记图谱失败，文件仍可用
        graphs = store.reset_stuck_graphs()
        if graphs:
            logger.warning("kb 启动恢复：%d 个文件图谱构建中断，标记 failed", graphs)

    def submit(self, file_id: uuid.UUID, user_id: str, path: Path) -> None:
        self._pool.submit(self._process, file_id, user_id, path)

    def submit_graph(self, file_id: uuid.UUID) -> None:
        """只重建图谱（切片已在库里，不重复解析源文件）。"""
        self._graph_pool.submit(self._rebuild_graph, file_id)

    def submit_align(self, file_id: uuid.UUID) -> None:
        """实体对齐（P2）：图谱建成后把同一实体的多种写法合并。

        与建图共用图谱池：两者都读写同一份图谱，串行才不会出现
        "对齐正在搬关系，建图把实体整份删了重建"的竞态。
        """
        self._graph_pool.submit(self._align, file_id)

    def submit_reembed(self, file_id: uuid.UUID) -> None:
        """改名后按新文件名重嵌全部切片（后台）。"""
        self._pool.submit(self._reembed, file_id)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=False)
        self._graph_pool.shutdown(wait=wait, cancel_futures=False)

    def _process(self, file_id: uuid.UUID, user_id: str, path: Path) -> None:
        try:
            stage = time.perf_counter()
            store.update_kb_status(
                file_id, "parsing", chunk_count=0, text_count=0, table_count=0
            )
            segments = parse_segments(path)
            parse_seconds = time.perf_counter() - stage
            chunks = chunk_segments(
                segments,
                max_chars=settings.kb_chunk_size,
                overlap=settings.kb_chunk_overlap,
                within_page=settings.kb_chunk_within_page,
            )
            if not chunks:
                store.update_kb_status(file_id, "failed", error="文档内容为空")
                return
            store.update_kb_status(file_id, "embedding")
            # 嵌入时带上「文档名 + 页码/表号」的语境前缀（见 chunking.context_prefix）：
            # 存储的正文不变，只是送进向量模型的那份文本更有辨识度。
            # 批间并行（EMBEDDING_CONCURRENCY），结果顺序与切片一致
            filename = (store.get_kb_file(file_id) or {}).get("filename") or path.name
            embed_stage = time.perf_counter()
            vectors = embed_texts_parallel(
                [
                    embedding_input(
                        chunk.text,
                        filename=filename,
                        page=chunk.page,
                        kind=chunk.kind,
                        section=chunk.section,
                    )
                    for chunk in chunks
                ],
                concurrency=settings.kb_embedding_concurrency,
            )
            embed_seconds = time.perf_counter() - embed_stage
            db_stage = time.perf_counter()
            done = 0
            batch = settings.kb_embedding_batch
            for seq_start in range(0, len(chunks), batch):
                done += store.insert_kb_chunks(
                    file_id,
                    user_id,
                    chunks[seq_start : seq_start + batch],
                    vectors[seq_start : seq_start + batch],
                    seq_start,
                )
            text_n = sum(1 for c in chunks if c.kind == "text")
            table_n = sum(1 for c in chunks if c.kind == "table")
            store.update_kb_status(
                file_id,
                "ready",
                chunk_count=done,
                text_count=text_n,
                table_count=table_n,
            )
            logger.info(
                "kb 处理完成 file_id=%s chunks=%d text=%d table=%d bytes=%d "
                "parse=%.1fs embed=%.1fs db=%.1fs",
                file_id,
                done,
                text_n,
                table_n,
                path.stat().st_size,
                parse_seconds,
                embed_seconds,
                time.perf_counter() - db_stage,
            )
            # 图谱不再自动生成：管理员在知识库页人工核对/修订切片后，
            # 自己选文件触发（POST /v1/kg/graph/build）——图谱是贵的 LLM
            # 操作，建完再改切片就得整份重抽，先审后建才不浪费
        except Exception as e:
            # 原因写进 file.error 给前端展示，日志留全栈
            logger.exception("kb 处理失败 file_id=%s", file_id)
            store.update_kb_status(file_id, "failed", error=str(e)[:300])

    def _rebuild_graph(self, file_id: uuid.UUID) -> None:
        """用库里的切片重建图谱（不重新解析源文件）。

        为什么不再 parse：切片已经是「解析 + 人工核对」后的结果，重新解析
        不仅要再跑一遍（PDF 还要再渲染 + 视觉识别 20 页，几分钟），
        还可能因为 chunk 参数变化导致 seq 与库里的切片对不上——
        mention 指向错的原文比没有 mention 更糟。
        """
        try:
            store.update_graph_status(
                file_id, "building", entity_count=0, batch_done=0, batch_total=0
            )
            rows = store.list_kb_chunk_inputs(file_id)
            chunks = [
                Chunk(
                    text=r["text"],
                    page=r["page"],
                    kind=r["kind"],
                    page_end=r.get("page_end"),
                    section=r.get("section"),
                )
                for r in rows
            ]
            if not chunks:
                raise RuntimeError("文件没有可用的切片，请先重新解析")
            # 抽取配置跟着文件走：doc_type 决定注入哪套配置块，level 决定轻重。
            # 空 doc_type（老文件/未指定）= 全局默认的通用配置
            record = store.get_kb_file(file_id) or {}
            doc_type = (record.get("doc_type") or "").strip() or settings.kb_default_doc_type
            level = (record.get("extract_level") or "").strip() or "full"
            counts = build_file_graph(
                file_id,
                chunks,
                doc_type=doc_type,
                level=level,
                # 文件名进 prompt：表格片段里没有公司全称，模型要靠它认主体
                doc_name=record.get("filename"),
                progress=lambda c: store.update_graph_status(
                    file_id,
                    "building",
                    entity_count=c["entities"],
                    batch_done=c.get("batches"),
                    batch_total=c.get("total_batches"),
                ),
            )
            # 实体对齐并进建图流程：建完即对齐好，前端不再有手动「对齐」入口。
            # 对齐失败不把 ready 打回 failed（图仍可用，只是多几个重复点），
            # entity_count 回退用抽取的原始数
            aligned = self._align(file_id)
            store.update_graph_status(
                file_id,
                "ready",
                entity_count=aligned if aligned is not None else counts["entities"],
            )
        except Exception:
            logger.exception("kb 图谱重建失败 file_id=%s", file_id)
            store.update_graph_status(file_id, "failed")

    def _align(self, file_id: uuid.UUID) -> int | None:
        """实体对齐：合并同一实体的多种写法，进度写 align_status。

        返回对齐后的实体数（entities - merged）；失败返回 None——
        图谱本身还是可用的，只是多几个重复点，不该因此把 graph_status
        也打成失败让用户以为图没了。
        """
        try:
            store.set_align_status(file_id, "aligning")
            counts = align_file_graph(
                file_id,
                progress=lambda c: store.set_align_status(
                    file_id, "aligning", alias_count=c["merged"]
                ),
            )
            store.set_align_status(
                file_id,
                "ready",
                alias_count=counts["merged"],
                entity_count=counts["entities"] - counts["merged"],
            )
            return counts["entities"] - counts["merged"]
        except Exception:
            logger.exception("kb 实体对齐失败 file_id=%s", file_id)
            store.set_align_status(file_id, "failed")
            return None

    def _reembed(self, file_id: uuid.UUID) -> None:
        """改名重嵌：语境前缀《文件名》在嵌入时就烧进向量里（不落库），
        改名后同一份文件的向量语境还停在旧名，必须整份重算才干净。
        只换向量不动文本；失败只影响语境新鲜度，检索照常可用。"""
        try:
            record = store.get_kb_file(file_id)
            if record is None:
                return
            filename = record["filename"]
            chunks = store.list_kb_chunk_inputs(file_id)
            vectors = embed_texts_parallel(
                [
                    embedding_input(
                        c["text"],
                        filename=filename,
                        page=c["page"],
                        kind=c["kind"],
                        # 与首次写入路径保持一致：漏传 section 会让改名后
                        # 的向量和首次解析的向量不同源
                        section=c.get("section"),
                    )
                    for c in chunks
                ],
                concurrency=settings.kb_embedding_concurrency,
            )
            done = 0
            batch = settings.kb_embedding_batch
            for start in range(0, len(chunks), batch):
                part = chunks[start : start + batch]
                done += store.update_kb_chunk_vectors(
                    list(zip((c["id"] for c in part), vectors[start : start + batch]))
                )
            logger.info("kb 改名重嵌完成 file_id=%s chunks=%d", file_id, done)
        except Exception:
            logger.exception("kb 改名重嵌失败 file_id=%s", file_id)


kb_pipeline = KbPipeline(
    workers=settings.kb_pipeline_workers, graph_workers=settings.kb_graph_workers
)
