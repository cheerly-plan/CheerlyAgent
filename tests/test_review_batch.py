"""审核列表可读性 + 跨报告批量确认。用临时目录，不碰真实审核记录。"""

from pathlib import Path

import pytest

from ai_agent.report import review as R


@pytest.fixture
def outdir(tmp_path: Path) -> Path:
    return tmp_path


def _entry(reason: str, item_id: str = "i1") -> R.ReviewEntry:
    return R.ReviewEntry(id=item_id, reason=reason)


def _record(report_id: str, items: list[R.ReviewEntry], owner: str = "u1") -> R.ReviewRecord:
    now = "2026-11-15T00:00:00"
    return R.ReviewRecord(
        report_id=report_id,
        title="测试研报",
        owner=owner,
        report_status="ok",
        status="pending",
        created_at=now,
        updated_at=now,
        # 与 decide() 的完整性校验对齐：hash 不一致会被判「内容已变更」
        content_hash="",
        validation_passed=True,
        error_count=0,
        artifacts={},
        items=items,
        decisions=[],
        finalized_by="",
        finalized_at="",
        auto_finalized=False,
    )


# ——— 条目分类 ———

def test_item_kind_classifies_reasons():
    assert R.item_kind(_entry(R.DEEP_REVIEW_REASON)) == R.ITEM_KIND_DEEP_PDF
    assert R.item_kind(_entry("正文出现未核对数字 688361")) == R.ITEM_KIND_NUMBER
    assert (
        R.item_kind(_entry("正文写了 营业收入=1，但看不到出处标记"))
        == R.ITEM_KIND_CITATION
    )
    assert R.item_kind(_entry("别的什么")) == R.ITEM_KIND_OTHER


def test_entry_payload_marks_batchable():
    deep = R.entry_payload(_entry(R.DEEP_REVIEW_REASON))
    assert deep["kind"] == "deep_pdf" and deep["batchable"] is True
    num = R.entry_payload(_entry("正文出现未核对数字 123"))
    assert num["batchable"] is False


# ——— 列表带上待审内容 ———

def test_list_records_include_pending_list(outdir):
    """列表要能在不点进详情的情况下判断「在审什么」。"""
    R._write_index(
        {r.report_id: r.to_dict() for r in [_record("r1", [_entry(R.DEEP_REVIEW_REASON)])]},
        outdir,
    )
    rows = R.list_records(output_dir=outdir)
    assert rows[0]["pending_items"] == 1
    assert rows[0]["pending_list"][0]["kind"] == "deep_pdf"
    assert rows[0]["batch_approvable"] is True


def test_list_records_marks_non_batchable(outdir):
    R._write_index(
        {r.report_id: r.to_dict() for r in [_record("r2", [_entry("正文出现未核对数字 123")])]},
        outdir,
    )
    rows = R.list_records(output_dir=outdir)
    assert rows[0]["batch_approvable"] is False


# ——— 批量确认 ———

def test_approve_batch_only_touches_batchable(outdir):
    """含具体数字问题的报告不能被批量批准，必须逐条核对。"""
    R._write_index(
        {
            "deep1": _record("deep1", [_entry(R.DEEP_REVIEW_REASON)]).to_dict(),
            "num1": _record("num1", [_entry("正文出现未核对数字 123")]).to_dict(),
        },
        outdir,
    )
    result = R.approve_batch(["deep1", "num1"], "tester", output_dir=outdir)
    assert result["approved"] == ["deep1"]
    skipped_ids = [item["report_id"] for item in result["skipped"]]
    assert "num1" in skipped_ids
    reason = next(i["reason"] for i in result["skipped"] if i["report_id"] == "num1")
    assert "逐条核对" in reason

    after = {r["report_id"]: r for r in R.list_records(output_dir=outdir)}
    assert after["num1"]["pending_items"] == 1  # 未被改动


def test_approve_batch_reports_unknown_ids(outdir):
    result = R.approve_batch(["nope"], "tester", output_dir=outdir)
    assert result["approved"] == []
    assert result["skipped"][0]["reason"] == "没有该审核记录"


def test_approve_batch_skips_records_without_pending(outdir):
    R._write_index({"r3": _record("r3", []).to_dict()}, outdir)
    result = R.approve_batch(["r3"], "tester", output_dir=outdir)
    assert result["approved"] == []
    assert result["skipped"][0]["reason"] == "没有待审条目"


# ——— 按报告聚合 ———

def test_groups_pair_base_and_deep(outdir):
    """底稿与推断层合成一条：用户眼里是同一份报告。"""
    R._write_index(
        {
            "r1": _record("r1", []).to_dict(),
            "r1-deep": _record("r1-deep", [_entry(R.DEEP_REVIEW_REASON)]).to_dict(),
        },
        outdir,
    )
    # 底稿定稿，推断层仍在待审
    rows = R.list_report_groups(output_dir=outdir)
    assert len(rows) == 1
    group = rows[0]
    assert group["report_id"] == "r1"
    assert group["base"]["status"] == "pending"
    assert group["deep"]["status"] == "pending"
    assert group["deep"]["pending_items"] == 1
    assert group["next_action"] == "approve_deep"


def test_group_next_action_finalize_when_deep_approved(outdir):
    deep = _record("r2-deep", [])
    deep.status = R.APPROVED
    R._write_index({"r2": _record("r2", []).to_dict(), "r2-deep": deep.to_dict()}, outdir)
    rows = R.list_report_groups(output_dir=outdir)
    assert rows[0]["next_action"] == "finalize"
    assert rows[0]["finalizable"] is True


def test_group_next_action_rejected(outdir):
    deep = _record("r3-deep", [])
    deep.status = R.REJECTED
    R._write_index({"r3": _record("r3", []).to_dict(), "r3-deep": deep.to_dict()}, outdir)
    assert R.list_report_groups(output_dir=outdir)[0]["next_action"] == "rejected"


def test_group_without_deep_layer(outdir):
    """只有底稿（单 PDF 档）也要能正常成组。"""
    R._write_index({"r4": _record("r4", []).to_dict()}, outdir)
    rows = R.list_report_groups(output_dir=outdir)
    assert len(rows) == 1
    assert rows[0]["deep"] is None
    assert rows[0]["base"] is not None
