"""研报生成的后台任务表（可选异步入口）。

设计边界：

* **不改同步端点**：``POST /v1/report`` 仍同步跑完并返回完整 payload；本模块只给
  ``POST /v1/report/jobs`` 一组新端点使用，任务里跑的是同一条
  ``run_report`` + ``_payload`` 链路，结果逐字一致；
* **进度复用既有通道**：``tools/progress.emit_progress`` 用 ContextVar 回传，
  主流水线已有「理解需求 / 补齐原文 / 抽数 / 校验 / 渲染 / 深度 / 落盘」等阶段；
  worker 线程里 ``set_progress`` 就能把这些阶段如实写进任务状态，不需要另造
  一套轮询心跳或改流水线签名；
* **纯内存、按用户隔离**：任务表不落盘，进程重启后未完成的任务会丢失；
  已生成的报告文件仍落在 ``report_output_dir``，通过审核队列与文件端点取回，
  不依赖任务表。查询任务时必须 owner 匹配，避免越权读到别人的研报正文；
* **有界**：同时运行的任务用信号量限制，保留条数有上限，优先淘汰旧的已完成任务。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_agent.core.logging import get_logger
from ai_agent.report.progress import advance_progress
from ai_agent.storage.db import user_scope
from ai_agent.tools.progress import reset_progress, set_progress

logger = get_logger(__name__)

Runner = Callable[[], dict[str, Any]]

_TERMINAL = frozenset({"succeeded", "failed"})


@dataclass
class ReportJob:
    job_id: str
    owner: str
    request_text: str
    status: str = "queued"  # queued / running / succeeded / failed
    progress: int = 0
    stage: str = "排队中"
    messages: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def elapsed_seconds(self) -> float:
        end = self.finished_at or time.time()
        return round(end - (self.started_at or self.created_at), 1)

    def to_dict(self, *, include_result: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "messages": list(self.messages[-30:]),
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds(),
            "request": self.request_text[:120],
            "poll_url": f"/v1/report/jobs/{self.job_id}",
            "result_url": f"/v1/report/jobs/{self.job_id}/result",
        }
        if include_result:
            payload["result"] = self.result
        return payload


class ReportJobManager:
    """内存任务表。所有读写都在同一把 RLock 下，worker 只改自己那一条记录。"""

    def __init__(self, *, max_jobs: int = 100, max_running: int = 2) -> None:
        self._jobs: dict[str, ReportJob] = {}
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(max(1, max_running))
        self._max_jobs = max(1, max_jobs)

    def submit(self, *, owner: str, request_text: str, runner: Runner) -> ReportJob:
        job = ReportJob(
            job_id=uuid.uuid4().hex[:12], owner=owner, request_text=request_text
        )
        with self._lock:
            self._prune_locked()
            self._jobs[job.job_id] = job
        threading.Thread(
            target=self._execute,
            args=(job, runner),
            name=f"report-job-{job.job_id}",
            daemon=True,
        ).start()
        return job

    def get(self, job_id: str, owner: str) -> ReportJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job if job is not None and job.owner == owner else None

    def find_running(self, *, owner: str, request_text: str) -> ReportJob | None:
        """同一用户同一需求的在跑任务：聊天侧重复提交时复用它，别重复烧模型。"""
        with self._lock:
            for job in self._jobs.values():
                if (
                    job.owner == owner
                    and job.request_text == request_text
                    and job.status in {"queued", "running"}
                ):
                    return job
        return None

    def _execute(self, job: ReportJob, runner: Runner) -> None:
        # 信号量满时状态保持 queued，前端轮询看到的就是真实的排队。
        with self._slots:
            self._update(job.job_id, status="running", started_at=time.time(),
                         stage="正在启动研报流水线…")
            # ContextVar 是线程本地的：必须在这个 worker 线程里注册回调。
            token = set_progress(lambda message: self._on_progress(job.job_id, message))
            try:
                # 任务线程不继承请求线程的 ContextVar：显式按 owner 设好用户作用域，
                # 否则任务里任何 current_user_id() 都会静默退回 "default"。
                with user_scope(job.owner):
                    payload = runner()
            except Exception as exc:
                logger.exception("研报后台任务失败 job_id=%s", job.job_id)
                self._update(
                    job.job_id, status="failed", stage="生成失败",
                    error=f"{type(exc).__name__}: {exc}", finished_at=time.time(),
                )
            else:
                self._update(
                    job.job_id, status="succeeded", stage="已完成", progress=100,
                    result=payload, finished_at=time.time(),
                )
            finally:
                reset_progress(token)

    def _on_progress(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in _TERMINAL:
                return
            job.stage = message
            job.progress = advance_progress(job.progress, message)
            if not job.messages or job.messages[-1] != message:
                job.messages.append(message)
            del job.messages[:-30]

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                setattr(job, key, value)

    def _prune_locked(self) -> None:
        finished = sorted(
            (job for job in self._jobs.values() if job.status in _TERMINAL),
            key=lambda job: job.finished_at or job.created_at,
        )
        for job in finished:
            if len(self._jobs) < self._max_jobs:
                break
            self._jobs.pop(job.job_id, None)


job_manager = ReportJobManager()
