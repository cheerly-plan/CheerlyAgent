"""可选异步研报入口：任务表状态、进度回传、失败、按用户隔离与路由契约。"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from ai_agent.api.app import create_app
from ai_agent.api.routers import report as router
from ai_agent.api.schemas import ReportRequest
from ai_agent.report.jobs import ReportJob, ReportJobManager
from ai_agent.report.deep.pipeline import ENTRY_CHAPTER_CONCURRENCY
from ai_agent.storage.db import UserRecord
from ai_agent.tools.progress import emit_progress

_ALICE = UserRecord(user_id="u-1", username="alice", display_name="alice")
_BOB = UserRecord(user_id="u-2", username="bob", display_name="bob")


def _wait(manager: ReportJobManager, job_id: str, owner: str) -> ReportJob:
    for _ in range(400):
        job = manager.get(job_id, owner)
        assert job is not None
        if job.status in {"succeeded", "failed"}:
            return job
        time.sleep(0.005)
    raise AssertionError("后台任务未在预期时间内结束")


def test_manager_tracks_progress_from_existing_emit_progress():
    manager = ReportJobManager()

    def runner():
        emit_progress("正在抽取财务指标…")
        emit_progress("正在校验数字与出处…")
        emit_progress("正在渲染 PDF…")
        return {"report_id": "abc123"}

    job = manager.submit(owner="alice", request_text="生成甲公司研报", runner=runner)
    done = _wait(manager, job.job_id, "alice")
    assert done.status == "succeeded" and done.progress == 100
    assert done.result == {"report_id": "abc123"}
    assert "正在校验数字与出处…" in done.messages
    assert "正在渲染 PDF…" in done.messages
    assert done.to_dict(include_result=True)["result"] == {"report_id": "abc123"}


def test_manager_records_failure_without_leaking_stack():
    manager = ReportJobManager()

    def runner():
        raise RuntimeError("知识库不可用")

    job = manager.submit(owner="alice", request_text="x", runner=runner)
    done = _wait(manager, job.job_id, "alice")
    assert done.status == "failed"
    assert "RuntimeError" in done.error and "知识库不可用" in done.error
    assert done.result is None


def test_manager_isolates_jobs_by_owner():
    manager = ReportJobManager()
    job = manager.submit(owner="alice", request_text="x", runner=lambda: {"ok": 1})
    assert manager.get(job.job_id, "alice") is job
    assert manager.get(job.job_id, "bob") is None
    _wait(manager, job.job_id, "alice")


def _queued_job(manager: ReportJobManager) -> ReportJob:
    job = ReportJob(job_id="job-fixed-1", owner="alice", request_text="x")
    manager._jobs[job.job_id] = job
    return job


def test_job_endpoints_enforce_owner_and_completion(monkeypatch):
    manager = ReportJobManager()
    monkeypatch.setattr(router, "job_manager", manager)
    job = _queued_job(manager)

    waiting = router.get_report_job(job.job_id, _ALICE)
    assert waiting["status"] == "queued" and waiting["progress"] == 0
    assert waiting["poll_url"] == f"/v1/report/jobs/{job.job_id}"

    with pytest.raises(HTTPException) as denied:
        router.get_report_job(job.job_id, _BOB)
    assert denied.value.status_code == 404

    with pytest.raises(HTTPException) as early:
        router.get_report_job_result(job.job_id, _ALICE)
    assert early.value.status_code == 409

    manager._update(job.job_id, status="succeeded", result={"report_id": "r1"},
                    progress=100)
    assert router.get_report_job_result(job.job_id, _ALICE) == {"report_id": "r1"}
    assert router.get_report_job(job.job_id, _ALICE)["result"] == {"report_id": "r1"}


def test_submit_endpoint_shares_sync_pipeline_and_enables_concurrency(monkeypatch):
    captured: dict = {}

    def fake_run(req: ReportRequest):
        captured["req"] = req
        return SimpleNamespace(marker="result")

    monkeypatch.setattr(router, "_run_request", fake_run)
    monkeypatch.setattr(router, "_payload", lambda result: {"marker": result.marker})
    manager = ReportJobManager()
    monkeypatch.setattr(router, "job_manager", manager)

    req = ReportRequest(request="生成甲公司 2026Q3 研报", deep="full", write_pdf=False)
    submitted = router.submit_report_job(req, _ALICE)
    assert submitted["status"] in {"queued", "running", "succeeded"}
    job = _wait(manager, submitted["job_id"], "alice")
    assert job.result == {"marker": "result"}
    assert captured["req"] is req


def test_api_entry_uses_entry_concurrency_for_deep_options(monkeypatch):
    """同步端点与后台任务共用 _run_request，对外入口必须显式开并发。"""
    seen: dict = {}

    def fake_run_report(request, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(marker="ok")

    monkeypatch.setattr(router, "run_report", fake_run_report)
    router._run_request(ReportRequest(request="x", deep="full", write_pdf=False))
    assert seen["deep"].chapter_concurrency == ENTRY_CHAPTER_CONCURRENCY
    assert seen["deep"].enabled is True
    assert seen["deep"].pdf_mode == "separate"


def test_job_routes_are_registered():
    # app.routes 里有 _IncludedRouter 包装层，openapi() 才是展开后的真实路径表。
    paths = set(create_app().openapi()["paths"])
    assert "/v1/report/jobs" in paths
    assert "/v1/report/jobs/{job_id}" in paths
    assert "/v1/report/jobs/{job_id}/result" in paths
    # 既有同步端点没被替换
    assert "/v1/report" in paths
