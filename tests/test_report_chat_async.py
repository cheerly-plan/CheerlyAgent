"""聊天入口异步研报：提交即返回 job_id，结果靠 report_job_status 取回。

深度层 5–8 分钟，同步等待会把 SSE 拖到网关超时。改成「提交后台任务 + 轮询」
后：聊天流几秒结束、断线不丢活、重复提交复用同一任务、任务按 owner 隔离。
"""

from __future__ import annotations

import re
import threading
import time
from types import SimpleNamespace

import ai_agent.tools.report.report_generate as rg
import ai_agent.tools.report.report_status as rs
from ai_agent.report.jobs import job_manager
from ai_agent.storage.db import user_scope

JOB_RE = re.compile(r"job_id=([0-9a-f]{6,})")


def fake_result():
    return SimpleNamespace(
        document=SimpleNamespace(title="中国银行股份有限公司 2026Q3 财报深度分析"),
        status="ok",
        deliverable="pdf",
        pdf_status="generated",
        report_id="abc1234567",
        review_status="finalized",
        validation=SimpleNamespace(
            summary="校验通过，但仍有 1 条需人工审核。", findings=[], review_queue=[]
        ),
        artifacts=SimpleNamespace(
            pdf="bank.pdf", markdown="bank.md",
            provenance="bank.provenance.md", review="bank.review.json",
        ),
        deep_artifacts={
            "pdf": "bank.deep.pdf",
            "markdown": "bank.deep.md",
            "json": "bank.deep.json",
        },
        deep_error="",
        deep_pdf_error="",
        deep_pdf_review_id="abc1234567-deep",
        pdf_error="",
        gaps=[],
    )


def wait_job(job_id: str, timeout: float = 5.0):
    """等任务到终态；任务表是内存单例，测试里直接查它，别用 sleep 猜。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = job_manager.get(job_id, "default")
        if job is not None and job.status in {"succeeded", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"任务 {job_id} 未在 {timeout}s 内结束")


def _patch_run(monkeypatch, fn):
    monkeypatch.setattr(rg, "run_report", fn)


def test_generate_report_returns_immediately_with_job_id(monkeypatch):
    """工具不在调用线程里等生成：立刻给出 job_id，且不给下载链接。"""
    release = threading.Event()

    def slow(*_a, **_k):
        release.wait(5)
        return fake_result()

    _patch_run(monkeypatch, slow)
    start = time.perf_counter()
    out = rg.generate_report_tool.run(
        request="生成中国银行2026Q3财报深度分析PDF研报-立即返回"
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, f"提交耗时 {elapsed:.2f}s，说明又变成同步等待了"
    job_id = JOB_RE.search(out).group(1)
    assert "已提交研报生成任务" in out
    # 报告还没生成，不能提前给链接（否则用户会白点）
    assert "/v1/report/files/" not in out
    assert "report_job_status" in out

    # 运行中：状态如实说是「仍在生成」，不编结果
    running = rs.report_job_status_tool.run(job_id=job_id)
    assert "仍在生成" in running
    assert "/v1/report/files/" not in running

    release.set()
    job = wait_job(job_id)
    assert job.status == "succeeded"

    done = rs.report_job_status_tool.run(job_id=job_id)
    assert "/v1/report/files/bank.pdf" in done
    assert "/v1/report/files/bank.deep.pdf" in done
    assert "校验通过" in done


def test_job_stage_follows_pipeline_progress(monkeypatch):
    """任务进度复用流水线的 emit_progress：job.stage 不是永远停在「启动中」。"""
    release = threading.Event()

    def slow(*_a, **_k):
        release.wait(5)
        return fake_result()

    _patch_run(monkeypatch, slow)
    out = rg.generate_report_tool.run(request="生成中国银行2026Q3财报深度分析PDF研报-进度")
    job_id = JOB_RE.search(out).group(1)

    deadline = time.monotonic() + 3
    stage = ""
    while time.monotonic() < deadline:
        job = job_manager.get(job_id, "default")
        stage = job.stage if job else ""
        if "底稿与深度" in stage:
            break
        time.sleep(0.02)
    release.set()
    wait_job(job_id)
    assert "底稿与深度" in stage, stage


def test_duplicate_submit_reuses_running_job(monkeypatch):
    """模型重复调用同一需求：复用同一任务，不重复烧模型。"""
    release = threading.Event()

    def slow(*_a, **_k):
        release.wait(5)
        return fake_result()

    _patch_run(monkeypatch, slow)
    first = rg.generate_report_tool.run(request="生成中国银行2026Q3财报深度分析PDF研报-去重")
    second = rg.generate_report_tool.run(request="生成中国银行2026Q3财报深度分析PDF研报-去重")
    try:
        assert JOB_RE.search(first).group(1) == JOB_RE.search(second).group(1)
        assert "已复用" in second
    finally:
        release.set()
        wait_job(JOB_RE.search(first).group(1))


def test_failed_job_reports_reason_not_links(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("知识库暂不可用")

    _patch_run(monkeypatch, boom)
    out = rg.generate_report_tool.run(request="生成中国银行2026Q3财报深度分析PDF研报-失败")
    job = wait_job(JOB_RE.search(out).group(1))
    assert job.status == "failed"
    status = rs.report_job_status_tool.run(job_id=job.job_id)
    assert "失败" in status
    assert "知识库暂不可用" in status
    assert "/v1/report/files/" not in status


def test_job_is_isolated_by_owner(monkeypatch):
    release = threading.Event()

    def slow(*_a, **_k):
        release.wait(5)
        return fake_result()

    _patch_run(monkeypatch, slow)
    with user_scope("alice"):
        out = rg.generate_report_tool.run(request="生成中国银行2026Q3财报深度分析PDF研报-隔离")
    job_id = JOB_RE.search(out).group(1)
    try:
        # 别人的 job_id 一律当不存在：不回显任务、更不回显报告内容
        with user_scope("bob"):
            status = rs.report_job_status_tool.run(job_id=job_id)
        assert "没有找到" in status
        assert "/v1/report/files/" not in status
        # owner 自己能查到
        with user_scope("alice"):
            assert "仍在生成" in rs.report_job_status_tool.run(job_id=job_id)
    finally:
        release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = job_manager.get(job_id, "alice")
            if job is not None and job.status in {"succeeded", "failed"}:
                break
            time.sleep(0.02)


def test_unknown_job_id_does_not_pretend_success():
    out = rs.report_job_status_tool.run(job_id="deadbeef0000")
    assert "没有找到" in out
    assert "/v1/report/files/" not in out
    assert "审核队列" in out


def test_generate_tool_schema_has_no_path_params():
    from ai_agent.tools.registry import TOOLS_MAP

    tool = TOOLS_MAP["generate_report"]
    props = tool.schema()["function"]["parameters"]["properties"]
    assert set(props) == {"request", "report_type", "write_pdf"}
    status_props = TOOLS_MAP["report_job_status"].schema()["function"]["parameters"]["properties"]
    assert set(status_props) == {"job_id"}


def test_job_stage_carries_deep_chapter_progress(monkeypatch):
    """深度层的章节进度要能落到任务状态上：否则轮询永远停在「启动中」。"""
    from ai_agent.tools.progress import emit_progress

    release = threading.Event()

    def fake(*_a, **_k):
        emit_progress("深度分析章节进度 3/7（第4章 ROE）…")
        release.wait(5)
        return fake_result()

    _patch_run(monkeypatch, fake)
    out = rg.generate_report_tool.run(request="生成中国银行2026Q3财报深度分析PDF研报-深度进度")
    job_id = JOB_RE.search(out).group(1)
    deadline = time.monotonic() + 3
    stage, progress = "", 0
    while time.monotonic() < deadline:
        job = job_manager.get(job_id, "default")
        stage, progress = (job.stage, job.progress) if job else ("", 0)
        if "深度分析章节进度" in stage:
            break
        time.sleep(0.02)
    release.set()
    wait_job(job_id)
    assert "深度分析章节进度" in stage, stage
    assert progress >= 76, progress


def test_job_runs_under_owner_scope(monkeypatch):
    """任务线程要带 owner 的用户作用域，否则任务里的 current_user_id() 会退回 default。"""
    from ai_agent.storage.db import current_user_id

    seen: list[str] = []

    def fake(*_a, **_k):
        seen.append(current_user_id())
        return fake_result()

    _patch_run(monkeypatch, fake)
    with user_scope("carol"):
        out = rg.generate_report_tool.run(request="生成中国银行2026Q3财报深度分析PDF研报-作用域")
    job_id = JOB_RE.search(out).group(1)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = job_manager.get(job_id, "carol")
        if job is not None and job.status in {"succeeded", "failed"}:
            break
        time.sleep(0.02)
    assert seen == ["carol"]


def test_job_id_input_is_normalized():
    """模型常把「job_id=xxx」整句传回来，不能因此判成任务不存在。"""
    from ai_agent.tools.report.report_status import _normalize_job_id

    assert _normalize_job_id(" job_id=abc12345 ") == "abc12345"
    assert _normalize_job_id("`abc12345`") == "abc12345"
    assert _normalize_job_id("abc12345") == "abc12345"


def test_generate_report_emits_report_job_sse_event(monkeypatch):
    """前端要渲染任务卡片，得先拿到 job_id：工具通过结构化事件把它抛出去。"""
    from ai_agent.tools.progress import reset_event_emitter, set_event_emitter

    _patch_run(monkeypatch, lambda *_a, **_k: fake_result())
    events: list[dict] = []
    token = set_event_emitter(events.append)
    try:
        out = rg.generate_report_tool.run(request="生成中国银行2026Q3财报深度分析PDF研报-事件")
    finally:
        reset_event_emitter(token)
    job_id = JOB_RE.search(out).group(1)
    assert events == [
        {
            "type": "report_job",
            "job_id": job_id,
            "poll_url": f"/v1/report/jobs/{job_id}",
        }
    ]
    wait_job(job_id)
