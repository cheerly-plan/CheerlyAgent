"""常驻知识库入库进程：从 DB 抢 ``uploaded`` 任务并解析。

与 API 进程分离，避免 OCR／嵌入原生库拖死 HTTP。用法：

    almost-agent ingest-worker

``dev.sh`` 会与 API 一同拉起。多实例靠 ``claim_next_uploaded_kb_file`` 的
``FOR UPDATE SKIP LOCKED`` 互斥。
"""

from __future__ import annotations

import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger, setup_logging
from ai_agent.kb.ingest.parsing import supported
from ai_agent.kb.pipeline import kb_pipeline
from ai_agent.storage.store import store

logger = get_logger(__name__)


def _source_path(file_id: uuid.UUID) -> Path | None:
    folder = Path(settings.kb_upload_dir) / str(file_id)
    if not folder.is_dir():
        return None
    sources = sorted(
        p for p in folder.iterdir() if p.is_file() and supported(p.suffix)
    )
    return sources[0] if sources else None


def _handle_claimed(row: dict) -> None:
    file_id = uuid.UUID(str(row["file_id"]))
    user_id = str(row.get("user_id") or "")
    path = _source_path(file_id)
    if path is None:
        store.update_kb_status(
            file_id,
            "failed",
            error="服务重启导致处理中断，源文件已不在，请重新上传",
        )
        logger.warning("ingest-worker 无源文件 file_id=%s", file_id)
        return
    if not user_id:
        store.update_kb_status(file_id, "failed", error="缺少 user_id，无法解析")
        return
    logger.info(
        "ingest-worker 开始解析 file_id=%s name=%s path=%s",
        file_id,
        row.get("filename"),
        path.name,
    )
    kb_pipeline.process_claimed(file_id, user_id, path)


def run_ingest_worker(*, poll_seconds: float | None = None) -> None:
    """阻塞运行，直到收到 SIGINT／SIGTERM。"""
    setup_logging()
    interval = (
        float(settings.kb_ingest_poll_seconds)
        if poll_seconds is None
        else max(float(poll_seconds), 0.2)
    )
    workers = max(1, int(settings.kb_pipeline_workers))
    logger.info(
        "ingest-worker 启动 workers=%d poll=%.1fs upload_dir=%s",
        workers,
        interval,
        settings.kb_upload_dir,
    )
    # 把上次卡在 parsing／embedding 的、磁盘还在的文件退回 uploaded，本进程接着啃
    try:
        kb_pipeline.recover_stuck(requeue=True)
    except Exception:
        logger.exception("ingest-worker 启动恢复失败")

    stop = threading.Event()

    def _stop(*_args) -> None:
        logger.info("ingest-worker 收到停止信号")
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    inflight = 0
    lock = threading.Lock()

    def _run(row: dict) -> None:
        nonlocal inflight
        try:
            _handle_claimed(row)
        except Exception:
            logger.exception(
                "ingest-worker 任务异常 file_id=%s", row.get("file_id")
            )
            try:
                fid = uuid.UUID(str(row["file_id"]))
                store.update_kb_status(
                    fid, "failed", error="ingest-worker 任务异常，可手动重试"
                )
            except Exception:
                logger.exception("ingest-worker 标 failed 失败")
        finally:
            with lock:
                inflight -= 1

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="ingest-claim"
    ) as pool:
        while not stop.is_set():
            dispatched = 0
            with lock:
                slots = workers - inflight
            for _ in range(max(slots, 0)):
                row = store.claim_next_uploaded_kb_file()
                if row is None:
                    break
                with lock:
                    inflight += 1
                pool.submit(_run, row)
                dispatched += 1
            if dispatched == 0:
                stop.wait(timeout=interval)

    logger.info("ingest-worker 正在收尾…")
    kb_pipeline.shutdown(wait=True)
    store.close()
    logger.info("ingest-worker 已退出")
