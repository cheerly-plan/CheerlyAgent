"""知识库后台处理管道：解析 → 切片 → 批量嵌入 → 灌库。

单进程部署，沿用 turn-finalizer 的线程池模式：lifespan 启动时
recover_stuck()，关停时 shutdown() 等待在途任务完成。进度即
agent_kb_file.status（parsing → embedding → ready / failed），
前端轮询文件列表即可看到，不单独建 job 表。
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.chunking import chunk_segments, embedding_input
from ai_agent.kb.graph import build_file_graph
from ai_agent.kb.parsing import parse_segments
from ai_agent.memory.embeddings import embed_texts
from ai_agent.memory.store import store

logger = get_logger(__name__)


class KbPipeline:
    def __init__(self, workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="kb-pipeline"
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

    def submit_graph(self, file_id: uuid.UUID, path: Path) -> None:
        """只重建图谱（切片已在库里，不重复入库）。"""
        self._pool.submit(self._rebuild_graph, file_id, path)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=False)

    def _process(self, file_id: uuid.UUID, user_id: str, path: Path) -> None:
        try:
            store.update_kb_status(file_id, "parsing")
            segments = parse_segments(path)
            chunks = chunk_segments(
                segments,
                max_chars=settings.kb_chunk_size,
                overlap=settings.kb_chunk_overlap,
            )
            if not chunks:
                store.update_kb_status(file_id, "failed", error="文档内容为空")
                return
            store.update_kb_status(file_id, "embedding")
            # 嵌入时带上「文档名 + 页码/表号」的语境前缀（见 chunking.context_prefix）：
            # 存储的正文不变，只是送进向量模型的那份文本更有辨识度
            filename = (store.get_kb_file(file_id) or {}).get("filename") or path.name
            done = 0
            batch = settings.kb_embedding_batch
            for seq_start in range(0, len(chunks), batch):
                part = chunks[seq_start : seq_start + batch]
                vectors = embed_texts(
                    [
                        embedding_input(
                            chunk.text,
                            filename=filename,
                            page=chunk.page,
                            kind=chunk.kind,
                        )
                        for chunk in part
                    ]
                )
                done += store.insert_kb_chunks(
                    file_id, user_id, part, vectors, seq_start
                )
            store.update_kb_status(file_id, "ready", chunk_count=done)
            logger.info(
                "kb 处理完成 file_id=%s chunks=%d bytes=%d",
                file_id,
                done,
                path.stat().st_size,
            )
            # 图谱不再自动生成：管理员在知识库页人工核对/修订切片后，
            # 自己选文件触发（POST /v1/kg/graph/build）——图谱是贵的 LLM
            # 操作，建完再改切片就得整份重抽，先审后建才不浪费
        except Exception as e:
            # 原因写进 file.error 给前端展示，日志留全栈
            logger.exception("kb 处理失败 file_id=%s", file_id)
            store.update_kb_status(file_id, "failed", error=str(e)[:300])

    def _rebuild_graph(self, file_id: uuid.UUID, path: Path) -> None:
        """重新解析源文件并重建图谱（切片不重新入库）。"""
        try:
            store.update_graph_status(file_id, "building")
            segments = parse_segments(path)
            chunks = chunk_segments(
                segments,
                max_chars=settings.kb_chunk_size,
                overlap=settings.kb_chunk_overlap,
            )
            counts = build_file_graph(file_id, chunks)
            store.update_graph_status(
                file_id, "ready", entity_count=counts["entities"]
            )
        except Exception:
            logger.exception("kb 图谱重建失败 file_id=%s", file_id)
            store.update_graph_status(file_id, "failed")


kb_pipeline = KbPipeline(workers=settings.kb_pipeline_workers)
