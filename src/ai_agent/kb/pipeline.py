"""知识库后台处理管道：解析 → 切片 → 批量嵌入 → 灌库。

默认（``kb_ingest_inline=false``）：API 的 ``submit`` 只把文件留在 ``uploaded``，
由独立进程 ``almost-agent ingest-worker`` 用 ``FOR UPDATE SKIP LOCKED`` 抢任务。
Worker 内再 spawn 子进程跑解析／向量化——OCR SIGSEGV 只死子进程，HTTP 与
常驻 worker 都还在。

``kb_ingest_inline=true``：兼容单进程，API 内直接 spawn（测试或不想起 worker）。

图谱抽取仍是 LLM IO，留在本进程线程池。进度即 agent_kb_file.status，
前端轮询即可，不单独建 job 表。
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.graph.align import align_file_graph
from ai_agent.kb.ingest.chunking import Chunk, chunk_segments, embedding_input
from ai_agent.kb.graph.graph import build_file_graph
from ai_agent.kb.ingest.meta import (
    assign_chunk_subjects,
    extract_aliases,
    pick_chunk_subject,
    resolve_file_identity,
)
from ai_agent.kb.ingest.summary import index_file_summary
from ai_agent.kb.ingest.parsing import parse_segments
from ai_agent.storage.embeddings import embed_texts_parallel
from ai_agent.storage.store import store

logger = get_logger(__name__)


def _kb_parse_job(file_id: str, user_id: str, path: str) -> None:
    """子进程入口：解析 + 向量化。必须是顶层函数，供 spawn 序列化。"""
    from ai_agent.core.logging import setup_logging

    setup_logging()
    kb_pipeline._process(uuid.UUID(file_id), user_id, Path(path))


def _kb_reembed_job(file_id: str) -> None:
    """子进程入口：改名重嵌。"""
    from ai_agent.core.logging import setup_logging

    setup_logging()
    kb_pipeline._reembed(uuid.UUID(file_id))


def _mark_subprocess_crash(file_id: uuid.UUID, exitcode: int | None) -> None:
    """子进程非 0 退出时，若文件仍停在中间态，标 failed 供前端展示。"""
    row = store.get_kb_file(file_id) or {}
    if str(row.get("status") or "") not in {"uploaded", "parsing", "embedding"}:
        return
    code = "?" if exitcode is None else str(exitcode)
    store.update_kb_status(
        file_id,
        "failed",
        error=f"解析子进程异常退出 (exit={code})，可手动重试",
    )


class KbPipeline:
    """解析/向量化走子进程；图谱抽取走本进程线程池。

    分池是必要的：一次图谱抽取是几十分钟的活，跟解析/向量化挤同一个池子时，
    管理员点了「建图」之后上传的新文件只能排队等它跑完——用户看到的现象是
    「切片生成也变慢了」，实际是队列被长任务占住了。
    """

    def __init__(self, workers: int = 2, graph_workers: int = 1) -> None:
        # 监督线程：每个 worker 同时只盯一个 spawn 子进程，上限即并发解析数
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="kb-pipeline"
        )
        self._graph_pool = ThreadPoolExecutor(
            max_workers=max(1, graph_workers), thread_name_prefix="kb-graph"
        )
        self._mp_ctx = mp.get_context("spawn")
        # 同一 file_id 只允许一条解析在途，避免重投把 (file_id, seq) 插重
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()

    def recover_stuck(self, *, requeue: bool | None = None) -> None:
        """恢复中断的入库任务。

        ``requeue=True``（ingest-worker / inline）：把磁盘还在的中断文件退回
        ``uploaded`` 队列。``requeue=False``（API 热重载）：只处理无源文件与
        图谱卡住，**不要**动正在 parsing 的行——否则会把 worker 已领取的任务
        又打回队列，造成「改了 uploaded 却像没解析」或重复抢单。
        """
        do_requeue = settings.kb_ingest_inline if requeue is None else requeue
        # 图谱卡住与解析无关，API／worker 都可清。
        graphs = store.reset_stuck_graphs()
        if graphs:
            logger.warning("kb 启动恢复：%d 个文件图谱构建中断，标记 failed", graphs)
        if not do_requeue:
            # API 热重载：绝不能动 parsing／embedding——ingest-worker 可能正在跑，
            # 误标「源文件已不在」会导致研报把 failed 当结束、不等解析就出稿。
            return
        # 解析／向量化中断且磁盘 PDF 还在：清半成品切片后重投，免删了重传。
        requeued = self._requeue_interrupted_from_disk()
        if requeued:
            logger.warning(
                "kb 启动恢复：重投 %d 个中断文件（磁盘仍在，免重传）",
                len(requeued),
            )
        stuck = store.recover_stuck_kb_files(
            exclude_ids=[uuid.UUID(item) for item in requeued]
        )
        if stuck:
            logger.warning(
                "kb 启动恢复：%d 个中断文件无源文件，已标 failed（需重新上传）", stuck
            )
        # uploaded + 磁盘已有 PDF：热重载丢掉了 ThreadPool 任务，主动重投
        try:
            from ai_agent.kb.fetch_reports import requeue_uploaded_files

            restarted = requeue_uploaded_files()
            if restarted:
                logger.warning(
                    "kb 启动恢复：重投 %d 个仍停在 uploaded 的文件", len(restarted)
                )
        except Exception:
            logger.exception("kb 启动恢复：重投 uploaded 失败")

    def _requeue_interrupted_from_disk(self) -> list[str]:
        """parsing／embedding／可重试 failed → 磁盘源文件还在则重投。"""
        from ai_agent.kb.ingest.parsing import supported

        rows = list(store.list_interrupted_kb_files())
        # 自愈：曾被 API 热重载误标「源文件已不在」、但 PDF 其实还在
        rows.extend(store.list_false_missing_source_kb_files())
        seen: set[str] = set()
        restarted: list[str] = []
        for row in rows:
            file_id = str(row.get("file_id") or "")
            if not file_id or file_id in seen:
                continue
            seen.add(file_id)
            try:
                uid = uuid.UUID(file_id)
            except ValueError:
                continue
            folder = Path(settings.kb_upload_dir) / file_id
            if not folder.is_dir():
                continue
            sources = sorted(
                p for p in folder.iterdir() if p.is_file() and supported(p.suffix)
            )
            if not sources:
                continue
            user_id = str(row.get("user_id") or "")
            if not user_id:
                continue
            # 半成品切片清掉，避免 (file_id, seq) 冲突；状态先回 uploaded
            store.clear_kb_chunks(uid)
            store.update_kb_status(
                uid, "uploaded", error="", chunk_count=0, text_count=0, table_count=0
            )
            self.submit(uid, user_id, sources[0])
            restarted.append(file_id)
            logger.info(
                "kb 重投中断文件 file_id=%s name=%s prior=%s",
                file_id,
                row.get("filename"),
                row.get("status"),
            )
        return restarted

    def submit(self, file_id: uuid.UUID, user_id: str, path: Path) -> None:
        """投递解析任务。

        默认只保证状态为 ``uploaded``（队列），由 ingest-worker 领取；
        ``kb_ingest_inline`` 时在本进程 spawn 子进程执行（旧行为）。
        """
        # path 留给调用方落盘校验；worker 按 upload_dir/file_id 再找源文件
        _ = path
        if not settings.kb_ingest_inline:
            store.update_kb_status(
                file_id, "uploaded", error="", chunk_count=0, text_count=0, table_count=0
            )
            logger.info(
                "kb 入队 uploaded file_id=%s user=%s（等待 ingest-worker）",
                file_id,
                user_id,
            )
            return

        key = str(file_id)
        with self._inflight_lock:
            if key in self._inflight:
                logger.info("kb 跳过重复投递 file_id=%s", key)
                return
            self._inflight.add(key)

        def _run() -> None:
            try:
                self._run_in_subprocess(
                    _kb_parse_job,
                    (str(file_id), user_id, str(path)),
                    file_id=file_id,
                    label="parse",
                )
            finally:
                with self._inflight_lock:
                    self._inflight.discard(key)

        self._pool.submit(_run)

    def process_claimed(
        self, file_id: uuid.UUID, user_id: str, path: Path
    ) -> None:
        """ingest-worker 已抢到任务后调用：子进程内跑解析，崩了只标该文件。"""
        self._run_in_subprocess(
            _kb_parse_job,
            (str(file_id), user_id, str(path)),
            file_id=file_id,
            label="parse",
        )

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
        """改名后按新文件名重嵌全部切片（后台，子进程）。"""
        self._pool.submit(
            self._run_in_subprocess,
            _kb_reembed_job,
            (str(file_id),),
            file_id,
            "reembed",
        )

    def _run_in_subprocess(
        self,
        target,
        args: tuple,
        file_id: uuid.UUID,
        label: str,
    ) -> None:
        """spawn 子进程跑重活；异常／超时只标该文件 failed，不拖死监督线程。"""
        # OCR／pymupdf 偶发挂死：无超时 join 会占满 worker 槽，后面 uploaded 永远不跑。
        timeout = max(float(getattr(settings, "kb_ingest_job_timeout_seconds", 900)), 60.0)
        proc = self._mp_ctx.Process(
            target=target,
            args=args,
            name=f"kb-{label}-{file_id}",
            daemon=True,
        )
        logger.info("kb 子进程启动 label=%s file_id=%s timeout=%.0fs", label, file_id, timeout)
        proc.start()
        proc.join(timeout=timeout)
        if proc.is_alive():
            logger.error(
                "kb 子进程超时 label=%s file_id=%s timeout=%.0fs，强制终止",
                label,
                file_id,
                timeout,
            )
            proc.terminate()
            proc.join(timeout=15)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
            if label == "parse":
                store.update_kb_status(
                    file_id,
                    "failed",
                    error=f"解析子进程超时（>{int(timeout)}s），可手动重试",
                )
            return
        if proc.exitcode == 0:
            return
        logger.error(
            "kb 子进程异常退出 label=%s file_id=%s exit=%s",
            label,
            file_id,
            proc.exitcode,
        )
        if label == "parse":
            _mark_subprocess_crash(file_id, proc.exitcode)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=False)
        self._graph_pool.shutdown(wait=wait, cancel_futures=False)

    def _process(self, file_id: uuid.UUID, user_id: str, path: Path) -> None:
        try:
            stage = time.perf_counter()
            store.update_kb_status(file_id, "parsing", chunk_count=0, text_count=0, table_count=0)
            segments = parse_segments(path)
            parse_seconds = time.perf_counter() - stage
            chunks = chunk_segments(
                segments,
                max_chars=settings.kb_chunk_size,
                overlap=settings.kb_chunk_overlap,
                within_page=settings.kb_chunk_within_page,
                table_group_mode=settings.kb_table_group_mode,
            )
            if not chunks:
                store.update_kb_status(file_id, "failed", error="文档内容为空")
                return
            store.update_kb_status(file_id, "embedding")
            # 嵌入时带上「文档名 + 页码/表号」的语境前缀（见 chunking.context_prefix）：
            # 存储的正文不变，只是送进向量模型的那份文本更有辨识度。
            # 批间并行（EMBEDDING_CONCURRENCY），结果顺序与切片一致
            record = store.get_kb_file(file_id) or {}
            filename = record.get("filename") or path.name
            # 文件名规则优先；封面补公司名和报告期。doc_type 只在还没人工指定时写入。
            file_meta = resolve_file_identity(filename, _cover_text(segments))
            if not (record.get("doc_type") or "").strip() and file_meta["doc_type"]:
                store.set_kb_extract_config(
                    file_id,
                    doc_type=file_meta["doc_type"],
                    level=(record.get("extract_level") or "").strip() or "full",
                )
            store.set_kb_file_meta(
                file_id,
                subject=file_meta["subject"],
                subject_aliases=file_meta["subject_aliases"],
                period=file_meta["period"],
                meta=file_meta["meta"],
            )
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
            library = [
                (row["subject"], list(row["subject_aliases"] or []))
                for row in store.list_kb_subjects(user_id)
            ]
            chunk_subjects = assign_chunk_subjects(
                [chunk.text for chunk in chunks],
                file_meta["subject"],
                library,
            )
            # 重解析前清掉旧切片：否则 (file_id, seq=0) 唯一约束直接炸
            cleared = store.clear_kb_chunks(file_id)
            if cleared:
                logger.info("kb 重解析清旧切片 file_id=%s n=%d", file_id, cleared)
            done = 0
            batch = settings.kb_embedding_batch
            for seq_start in range(0, len(chunks), batch):
                done += store.insert_kb_chunks(
                    file_id,
                    user_id,
                    chunks[seq_start : seq_start + batch],
                    vectors[seq_start : seq_start + batch],
                    seq_start,
                    subject=file_meta["subject"],
                    period=file_meta["period"],
                    subjects=chunk_subjects[seq_start : seq_start + batch],
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
            # 文档摘要 + 摘要向量：切片入库后才有素材。它在检索前回答
            # 「该查哪几份文件」（见 kb/router.py），失败只让这份文件走
            # 老路径（模型指定 files），不影响文件本身可用
            if settings.kb_summary_enabled:
                index_file_summary(file_id)
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
                    # seq 必须带：建图会过滤低信息切片，下标不再等于库里的序号
                    seq=r.get("seq"),
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
            # 建图/对齐后才有实体，多主体文档的切片归属放这一步做
            self._attribute_subjects(file_id)
            # 图谱实体在这之后才有，摘要重算一次把「讲了哪些主体」吸收进去
            if settings.kb_summary_enabled:
                index_file_summary(file_id)
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
            _write_back_subject_aliases(file_id)
            return counts["entities"] - counts["merged"]
        except Exception:
            logger.exception("kb 实体对齐失败 file_id=%s", file_id)
            store.set_align_status(file_id, "failed")
            return None

    def _attribute_subjects(self, file_id: uuid.UUID) -> None:
        """多主体文档（行业研报等）的切片级主体归属。

        一份文件只讲一家公司时，切片直接继承文件主体就够了。但「新能源行业
        研报」这种一份文件讲十家的，切片全继承「新能源行业」会让检索分桶
        形同虚设——问「宁德时代营收」时所有切片都在同一个桶里抢名额。

        候选主体 = 知识库里已知的公司主体 + 本文件图谱里像公司的实体名；
        每个切片按提及次数归属到最像的那个主体，提不到的退回文件主体。
        纯规则、可重跑：图谱变了再跑一遍就刷新，不留隐式状态。
        """
        try:
            record = store.get_kb_file(file_id) or {}
            user_id = record.get("user_id") or ""
            default = (record.get("subject") or "").strip()
            candidates: list[tuple[str, list[str]]] = [
                (r["subject"], list(r["subject_aliases"] or []))
                for r in store.list_kb_subjects(user_id)
            ]
            known = {item[0] for item in candidates}
            for name in store.list_kb_entity_subjects(file_id):
                if name not in known:
                    candidates.append((name, extract_aliases(name)))
                    known.add(name)
            if default and default not in known:
                candidates.append((default, extract_aliases(default)))
            if len(candidates) < 2:
                return  # 只有一个主体可归，切片继承文件主体即可
            pairs = []
            for row in store.list_kb_chunk_inputs(file_id):
                subject = pick_chunk_subject(row["text"], candidates, default=default)
                # 切片入库时继承的就是文件主体，只有归属不同的才需要回写
                if subject != default:
                    pairs.append((row["seq"], subject))
            moved = store.reassign_chunk_subjects(file_id, pairs)
            if moved:
                logger.info("kb 切片主体归属 file_id=%s 覆盖 %d 片", file_id, moved)
        except Exception:
            # 归属失败不影响图谱与检索：切片仍带文件主体，只是分桶粗一些
            logger.exception("kb 切片主体归属失败 file_id=%s", file_id)

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


_GENERIC_ALIAS = {"公司", "集团", "股份", "控股", "银行", "证券", "保险", "基金"}


def _cover_text(segments, pages: int = 2) -> str:
    """取前两页正文，供文件名里没有的公司名和报告期补缺。"""
    if not any(getattr(seg, "page", None) for seg in segments):
        return "\n".join(seg.text or "" for seg in segments)[:4000]
    seen: list[int] = []
    parts: list[str] = []
    for seg in segments:
        page = seg.page
        if not page:
            continue
        if page not in seen:
            if len(seen) >= pages:
                continue
            seen.append(page)
        if page in seen:
            parts.append(seg.text or "")
    return "\n".join(parts)[:4000]


def _write_back_subject_aliases(file_id: uuid.UUID) -> None:
    """把对齐得到的主体简称写回文件别名，检索按整词命中。"""
    record = store.get_kb_file(file_id) or {}
    subject = str(record.get("subject") or "").strip()
    if not subject:
        return
    try:
        aligned = store.list_aligned_aliases(file_id, subject)
    except Exception:
        logger.warning("主体别名回写失败 file_id=%s", file_id, exc_info=True)
        return
    merged = extract_aliases(subject)
    for name in list(record.get("subject_aliases") or []) + list(aligned):
        name = str(name or "").strip()
        if not name or name in merged or name in _GENERIC_ALIAS or len(name) < 2:
            continue
        merged.append(name)
    store.set_kb_subject_aliases(file_id, merged)


kb_pipeline = KbPipeline(
    workers=settings.kb_pipeline_workers, graph_workers=settings.kb_graph_workers
)
