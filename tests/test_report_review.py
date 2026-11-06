"""M2：审核队列持久化 + 定稿硬门禁。

覆盖状态机、权限、内容指纹、端点门禁与索引损坏，全部走真实的
``run_report`` 产物（不 mock 门禁本身），避免「测试通过但真链路没接上」。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi import HTTPException

import ai_agent.api.routers.report as router
import ai_agent.report.pipeline as pipeline
from ai_agent.api.schemas import ReportDecisionRequest
from ai_agent.core.config import settings
from ai_agent.memory.db import UserRecord
from ai_agent.report import review
from ai_agent.report.extract import Evidence, ExtractResult, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan, SourceFile

_USER = UserRecord(user_id="u-1", username="tester", display_name="tester")
_REVIEWER = UserRecord(user_id="u-2", username="reviewer", display_name="reviewer")
_SUBJECT = "中国能源建设股份有限公司"
_FILENAME = "中国能源建设2026年第三季度报告.pdf"
_VALUE = "100,924,954"


@pytest.fixture
def env(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        monkeypatch.setattr(settings, "report_output_dir", out)
        monkeypatch.setattr(settings, "admin_usernames", "reviewer")
        yield out


def _plan() -> ReportPlan:
    return ReportPlan(
        title=f"{_SUBJECT} 2026Q3 财报深度分析",
        kind="company",
        deliverable="pdf",
        subjects=[_SUBJECT],
        period="2026Q3",
        dimensions=["同比"],
        indicators=["营业收入"],
        outline=[OutlineItem("本期结论", ""), OutlineItem("缺失与口径", "")],
        files=[SourceFile("file-1", _FILENAME, _SUBJECT, "2026Q3", "本期")],
    )


def _fact(*, page: int | None = 12, quote_ok: bool = True) -> IndicatorFact:
    return IndicatorFact(
        subject=_SUBJECT,
        period="2026Q3",
        name="营业收入",
        value=_VALUE,
        unit="元",
        column="本报告期",
        scope="consolidated",
        evidence=Evidence(
            file_id="file-1",
            filename=_FILENAME,
            page=page,
            page_end=page,
            section="主要会计数据",
            quote=f"营业收入 | {_VALUE}" if quote_ok else "营业收入 | 123",
        ),
    )


def _report(env, monkeypatch, *, page=12, quote_ok=True, write_pdf=False):
    plan = _plan()
    fact = _fact(page=page, quote_ok=quote_ok)
    monkeypatch.setattr(pipeline, "parse_request", lambda _text: plan)
    monkeypatch.setattr(
        pipeline,
        "extract_indicators",
        lambda *_a, **_k: ExtractResult(facts=[fact]),
    )
    return pipeline.run_report(
        "生成中国能源建设2026Q3财报深度分析PDF研报",
        output_dir=env,
        write_pdf=write_pdf,
    )


def _expect_http(code, fn, *args, **kwargs):
    with pytest.raises(HTTPException) as exc:
        fn(*args, **kwargs)
    assert exc.value.status_code == code
    return exc.value


def test_clean_report_auto_finalizes_and_allows_pdf_download(env, monkeypatch):
    result = _report(env, monkeypatch, write_pdf=True)

    assert result.validation.passed is True
    assert result.validation.review_queue == []
    assert result.pdf_status == "generated"
    assert result.review_status == review.FINALIZED

    record = review.get_record(result.report_id)
    assert record is not None
    assert record.status == review.FINALIZED
    assert record.auto_finalized is True
    assert record.content_hash
    # 快照与索引一致，且是完整审核记录（不再是 M1 的裸数组）
    snapshot = json.loads((env / result.artifacts.review).read_text(encoding="utf-8"))
    assert snapshot["report_id"] == result.report_id
    assert snapshot["status"] == review.FINALIZED

    assert router.download_report(result.artifacts.pdf, _USER).status_code == 200
    assert router.download_final(result.report_id, _USER).status_code == 200
    assert router.get_review(result.report_id, _USER)["finalizable"] is True


def test_pending_items_block_pdf_until_approved_and_finalized(env, monkeypatch):
    # 缺页码只进审核队列，不判 error：正好是「需要人工看一眼」的场景
    result = _report(env, monkeypatch, page=None, write_pdf=True)
    assert result.validation.passed is True
    assert result.review_status == review.PENDING
    assert result.pdf_status == "generated"

    record = review.get_record(result.report_id)
    assert len(record.items) == 1
    assert record.items[0].state == review.PENDING

    err = _expect_http(409, router.download_report, result.artifacts.pdf, _USER)
    assert "未定稿" in err.detail
    assert _expect_http(409, router.finalize_review, result.report_id, _REVIEWER).status_code == 409
    assert _expect_http(409, router.download_final, result.report_id, _USER).status_code == 409

    approved = router.decide_review(
        result.report_id,
        ReportDecisionRequest(action="approve", note="页码已人工核对"),
        _REVIEWER,
    )
    assert approved["status"] == review.APPROVED
    assert approved["items"][0]["reviewer"] == "reviewer"
    assert approved["items"][0]["note"] == "页码已人工核对"
    assert approved["finalizable"] is True

    finalized = router.finalize_review(result.report_id, _REVIEWER)
    assert finalized["status"] == review.FINALIZED
    assert finalized["finalized_by"] == "reviewer"
    assert router.download_report(result.artifacts.pdf, _USER).status_code == 200
    # 审批链留在记录里，可审计
    assert [item["action"] for item in finalized["decisions"]] == ["approve", "finalize"]


def test_validation_error_blocks_rendering_and_cannot_be_approved_away(env, monkeypatch):
    result = _report(env, monkeypatch, quote_ok=False, write_pdf=True)

    assert result.validation.passed is False
    assert result.pdf_status == "blocked"
    assert result.pdf is None
    assert result.artifacts.pdf == ""
    assert "禁止渲染 PDF" in result.pdf_error
    assert result.review_status == review.PENDING

    record = review.get_record(result.report_id)
    assert record.validation_passed is False
    assert record.error_count >= 1
    assert record.items  # 错误也进审核队列
    assert review.gate(record)[0] is False

    approved = review.decide(result.report_id, "approve", "reviewer")
    assert all(item.state == review.APPROVED for item in approved.items)
    assert approved.status == review.PENDING  # 人工批准不能豁免数值错误
    err = _expect_http(409, router.finalize_review, result.report_id, _REVIEWER)
    assert "校验未通过" in err.detail


def test_reject_then_reset_then_approve(env, monkeypatch):
    result = _report(env, monkeypatch, page=None, write_pdf=True)

    rejected = review.decide(result.report_id, "reject", "reviewer", note="出处存疑")
    assert rejected.status == review.REJECTED
    assert review.gate(rejected)[0] is False
    _expect_http(409, router.download_report, result.artifacts.pdf, _USER)

    reset = review.decide(result.report_id, "reset", "reviewer")
    assert reset.status == review.PENDING
    assert all(item.state == review.PENDING for item in reset.items)

    # 退回待审但还没批准，不能直接定稿
    with pytest.raises(review.ReviewForbidden):
        review.finalize(result.report_id, "reviewer")
    review.decide(result.report_id, "approve", "reviewer")
    assert review.finalize(result.report_id, "reviewer").status == review.FINALIZED


def test_tampering_after_finalization_invalidates_approval(env, monkeypatch):
    result = _report(env, monkeypatch, write_pdf=True)
    assert result.review_status == review.FINALIZED

    (env / result.artifacts.markdown).write_text("被人改过的正文", encoding="utf-8")
    record = review.get_record(result.report_id)
    assert record.status == review.PENDING
    assert record.finalized_by == ""
    assert any(item["action"] == "invalidate" for item in record.decisions)

    err = _expect_http(409, router.download_report, result.artifacts.pdf, _USER)
    assert "未定稿" in err.detail
    _expect_http(409, router.download_final, result.report_id, _USER)


def test_snapshot_is_read_only_index_is_authoritative(env, monkeypatch):
    result = _report(env, monkeypatch, write_pdf=True)
    (env / result.artifacts.review).write_text('{"status": "approved"}', encoding="utf-8")

    assert review.get_record(result.report_id).status == review.FINALIZED
    assert router.download_report(result.artifacts.pdf, _USER).status_code == 200


def test_non_admin_cannot_review_or_finalize(env, monkeypatch):
    result = _report(env, monkeypatch, page=None, write_pdf=True)
    err = _expect_http(
        403,
        router.decide_review,
        result.report_id,
        ReportDecisionRequest(action="approve"),
        _USER,
    )
    assert "管理员" in err.detail
    _expect_http(403, router.finalize_review, result.report_id, _USER)
    assert review.get_record(result.report_id).status == review.PENDING


def test_queue_listing_detail_and_unknown_ids(env, monkeypatch):
    pending = _report(env, monkeypatch, page=None, write_pdf=True)
    final = _report(env, monkeypatch, write_pdf=True)

    all_items = router.list_reviews(None, _USER)["items"]
    assert {item["report_id"] for item in all_items} == {
        pending.report_id,
        final.report_id,
    }
    pending_list = router.list_reviews("pending", _USER)["items"]
    assert [item["report_id"] for item in pending_list] == [pending.report_id]
    assert pending_list[0]["pending_items"] == 1
    assert pending_list[0]["finalizable"] is False

    detail = router.get_review(pending.report_id, _USER)
    assert detail["queue_url"] == f"/v1/report/reviews/{pending.report_id}"
    _expect_http(404, router.get_review, "deadbeef00", _USER)
    _expect_http(404, router.download_final, "deadbeef00", _USER)


def test_legacy_pdf_without_review_record_is_blocked(env, monkeypatch):
    monkeypatch.setattr(settings, "report_output_dir", env)
    (env / "legacy_abcdef1234.pdf").write_bytes(b"%PDF-1.4")
    err = _expect_http(409, router.download_report, "legacy_abcdef1234.pdf", _USER)
    assert "没有审核记录" in err.detail


def test_corrupted_index_is_never_silently_overwritten(env, monkeypatch):
    index = review.index_path(env)
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("{ broken json", encoding="utf-8")
    monkeypatch.setattr(pipeline, "parse_request", lambda _text: _plan())
    monkeypatch.setattr(
        pipeline,
        "extract_indicators",
        lambda *_a, **_k: ExtractResult(facts=[_fact()]),
    )

    # 登记审核会抛错：宁可让这次生成失败，也不能悄悄重建索引丢掉历史审批
    with pytest.raises(review.ReviewError):
        pipeline.run_report("生成研报", output_dir=env, write_pdf=False)
    assert index.read_text(encoding="utf-8") == "{ broken json"
