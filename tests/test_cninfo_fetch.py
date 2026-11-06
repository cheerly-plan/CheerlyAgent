"""巨潮补齐：全 mock 网络，不连外网。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_agent.core.config import settings
from ai_agent.kb import cninfo
from ai_agent.kb import fetch_reports as fr
from ai_agent.report.ensure_filings import ensure_filings_for_report
from ai_agent.report.parse import OutlineItem, ReportPlan
from ai_agent.tools.registry import TOOLS_MAP


@pytest.fixture(autouse=True)
def _enable_and_clear(monkeypatch):
    monkeypatch.setattr(settings, "enable_cninfo_fetch", True)
    monkeypatch.setattr(settings, "cninfo_request_interval_seconds", 0.0)
    monkeypatch.setattr(settings, "cninfo_ingest_wait_seconds", 1)
    cninfo.clear_stock_cache()
    yield
    cninfo.clear_stock_cache()


def _stock_payload() -> dict:
    return {
        "stockList": [
            {
                "code": "601988",
                "orgId": "jjxt0000028",
                "zwjc": "中国银行",
                "category": "A股",
            },
            {
                "code": "000001",
                "orgId": "gssz0000001",
                "zwjc": "平安银行",
                "category": "A股",
            },
        ]
    }


def _announcements_payload(*, full_size: int = 2_000_000) -> dict:
    return {
        "totalAnnouncement": 3,
        "totalpages": 1,
        "announcements": [
            {
                "announcementId": "1",
                "announcementTitle": "中国银行股份有限公司2025年第三季度报告",
                "announcementTime": 1761696000000,
                "adjunctUrl": "finalpage/2025-10-29/1224754192.PDF",
                "adjunctSize": full_size,
                "secCode": "601988",
                "secName": "中国银行",
            },
            {
                "announcementId": "2",
                "announcementTitle": "中国银行2025年第三季度报告摘要",
                "adjunctUrl": "finalpage/2025-10-29/summary.PDF",
                "adjunctSize": 400_000,
                "secCode": "601988",
                "secName": "中国银行",
            },
            {
                "announcementId": "3",
                "announcementTitle": "H股公告-2025年第三季度报告",
                "adjunctUrl": "finalpage/2025-10-29/hshare.PDF",
                "secCode": "601988",
                "secName": "中国银行",
            },
        ],
    }


def test_period_helpers():
    assert cninfo.period_label("2025Q3") == "2025年第三季度报告"
    assert cninfo.category_for_period("2025Q3") == "category_sjdbg_szsh"
    assert cninfo.category_for_period("2025") == "category_ndbg_szsh"
    assert cninfo.market_for_code("601988") == ("sse", "sh")
    assert cninfo.market_for_code("000001") == ("szse", "sz")
    assert cninfo.market_for_code("920001") == ("bj", "bj")


def test_resolve_code_by_name(monkeypatch):
    monkeypatch.setattr(cninfo.http, "request_json", lambda *a, **k: _stock_payload())
    stock = cninfo.resolve_code_by_name("中国银行股份有限公司")
    assert stock is not None
    assert stock.code == "601988"
    assert cninfo.resolve_code_by_name("601988").code == "601988"


def test_resolve_codes_in_text(monkeypatch):
    monkeypatch.setattr(cninfo.http, "request_json", lambda *a, **k: _stock_payload())
    rows = cninfo.resolve_codes_in_text("生成中国银行2025Q3财报深度分析研报")
    assert [item.code for item in rows] == ["601988"]


def test_search_keeps_small_full_and_filters_hshare(monkeypatch):
    monkeypatch.setattr(
        cninfo.http, "request_json", lambda *a, **k: _stock_payload()
    )
    monkeypatch.setattr(
        cninfo.http,
        "request_form",
        lambda *a, **k: _announcements_payload(full_size=2_000_000),
    )
    rows = cninfo.search_announcements("601988", "2025Q3")
    assert len(rows) == 1
    assert rows[0].announcement_id == "1"
    assert "摘要" not in rows[0].title
    assert rows[0].pdf_url.startswith("https://static.cninfo.com.cn/")


def test_search_switches_to_summary_when_full_too_large(monkeypatch):
    monkeypatch.setattr(
        cninfo.http, "request_json", lambda *a, **k: _stock_payload()
    )
    monkeypatch.setattr(
        cninfo.http,
        "request_form",
        lambda *a, **k: _announcements_payload(full_size=20_000_000),
    )
    monkeypatch.setattr(settings, "cninfo_full_max_mb", 4.0)
    rows = cninfo.search_announcements("601988", "2025Q3")
    assert len(rows) == 1
    assert rows[0].announcement_id == "2"
    assert "摘要" in rows[0].title


def test_half_year_uses_summary_when_full_huge(monkeypatch):
    monkeypatch.setattr(cninfo.http, "request_json", lambda *a, **k: _stock_payload())

    def form(*_a, **_k):
        return {
            "totalAnnouncement": 3,
            "totalpages": 1,
            "announcements": [
                {
                    "announcementId": "full",
                    "announcementTitle": "中国银行股份有限公司2025年半年度报告",
                    "adjunctUrl": "finalpage/2025-08-01/full.PDF",
                    "adjunctSize": 25_000_000,
                    "secCode": "601988",
                    "secName": "中国银行",
                },
                {
                    "announcementId": "sum",
                    "announcementTitle": "中国银行股份有限公司2025年半年度报告摘要",
                    "adjunctUrl": "finalpage/2025-08-01/sum.PDF",
                    "adjunctSize": 800_000,
                    "secCode": "601988",
                    "secName": "中国银行",
                },
                {
                    "announcementId": "h",
                    "announcementTitle": "H股公告-2025年半年度报告摘要",
                    "adjunctUrl": "finalpage/2025-08-01/h.PDF",
                    "secCode": "601988",
                    "secName": "中国银行",
                },
            ],
        }

    monkeypatch.setattr(cninfo.http, "request_form", form)
    monkeypatch.setattr(settings, "cninfo_full_max_mb", 4.0)
    rows = cninfo.search_announcements("601988", "2025H1")
    assert len(rows) == 1
    assert rows[0].announcement_id == "sum"
    assert "摘要" in rows[0].title


def test_pdf_host_whitelist():
    with pytest.raises(ValueError, match="白名单"):
        cninfo.assert_allowed_pdf_url("https://evil.example/a.pdf")
    with pytest.raises(ValueError, match="https"):
        cninfo.assert_allowed_pdf_url("http://static.cninfo.com.cn/a.pdf")
    cninfo.assert_allowed_pdf_url(
        "https://static.cninfo.com.cn/finalpage/2025-10-29/1224754192.PDF"
    )


def test_canonical_stem_has_no_announce_date_prefix():
    stem = cninfo._canonical_stem("中国银行股份有限公司", "2025Q3")
    assert stem == "中国银行股份有限公司2025年第三季度报告"
    assert (
        cninfo._canonical_stem("中国银行", "2025Q1", summary=True)
        == "中国银行2025年第一季度报告摘要"
    )


def test_fetch_ingests_and_submits(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "kb_upload_dir", tmp_path)
    monkeypatch.setattr(cninfo.http, "request_json", lambda *a, **k: _stock_payload())
    monkeypatch.setattr(
        cninfo.http, "request_form", lambda *a, **k: _announcements_payload()
    )
    monkeypatch.setattr(
        cninfo.http,
        "request_bytes",
        lambda *a, **k: (b"%PDF-1.4 mock", "application/pdf"),
    )

    created: dict = {}

    def fake_create(user_id, filename, size_bytes, mime, *, tags=None):
        import uuid

        file_id = uuid.uuid4()
        created["file_id"] = file_id
        created["filename"] = filename
        created["tags"] = tags
        created["size"] = size_bytes
        return file_id

    submitted: dict = {}

    monkeypatch.setattr(fr.store, "create_kb_file", fake_create)
    monkeypatch.setattr(fr.store, "find_kb_file_by_hash", lambda *a, **k: None)
    monkeypatch.setattr(fr.store, "update_kb_file_hash", lambda *a, **k: None)
    monkeypatch.setattr(fr.store, "delete_kb_file", lambda *a, **k: True)
    monkeypatch.setattr(
        fr.kb_pipeline,
        "submit",
        lambda file_id, user_id, path: submitted.update(
            {"file_id": file_id, "path": Path(path)}
        ),
    )

    result = fr.fetch_missing_reports("601988", "2025Q3", user_id="admin-1")
    assert not result.errors
    assert len(result.files) == 1
    item = result.files[0]
    assert item.status == "parsing"
    # 默认 mock 全文约 2MB < 8MB 阈值，仍取全文
    assert item.filename == "中国银行股份有限公司2025年第三季度报告"
    assert created["filename"] == item.filename
    assert "cninfo:1" in created["tags"]
    assert submitted["path"].exists()
    assert submitted["path"].read_bytes().startswith(b"%PDF")


def test_duplicate_content_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "kb_upload_dir", tmp_path)
    monkeypatch.setattr(cninfo.http, "request_json", lambda *a, **k: _stock_payload())
    monkeypatch.setattr(
        cninfo.http, "request_form", lambda *a, **k: _announcements_payload()
    )
    monkeypatch.setattr(
        cninfo.http,
        "request_bytes",
        lambda *a, **k: (b"%PDF-1.4 same", "application/pdf"),
    )
    monkeypatch.setattr(
        fr.store,
        "find_kb_file_by_hash",
        lambda *a, **k: {
            "file_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "filename": "已有文件",
        },
    )

    result = fr.fetch_missing_reports("601988", "2025Q3")
    assert result.files == []
    assert any("内容相同" in item for item in result.skipped)


def test_ensure_filings_fetches_missing_period(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "kb_upload_dir", tmp_path)
    monkeypatch.setattr(cninfo.http, "request_json", lambda *a, **k: _stock_payload())
    monkeypatch.setattr(
        cninfo.http, "request_form", lambda *a, **k: _announcements_payload()
    )
    monkeypatch.setattr(
        cninfo.http,
        "request_bytes",
        lambda *a, **k: (b"%PDF-1.4 mock", "application/pdf"),
    )
    monkeypatch.setattr(
        "ai_agent.report.ensure_filings._files_for", lambda *_a, **_k: []
    )

    created_ids: list[str] = []

    def fake_create(user_id, filename, size_bytes, mime, *, tags=None):
        import uuid

        file_id = uuid.uuid4()
        created_ids.append(str(file_id))
        return file_id

    monkeypatch.setattr(fr.store, "create_kb_file", fake_create)
    monkeypatch.setattr(fr.store, "find_kb_file_by_hash", lambda *a, **k: None)
    monkeypatch.setattr(fr.store, "update_kb_file_hash", lambda *a, **k: None)
    monkeypatch.setattr(fr.store, "delete_kb_file", lambda *a, **k: True)
    monkeypatch.setattr(fr.kb_pipeline, "submit", lambda *a, **k: None)

    def fake_get(file_id):
        return {"id": file_id, "status": "ready", "filename": "x", "period": "2025Q3"}

    monkeypatch.setattr(fr.store, "get_kb_file", fake_get)

    plan = ReportPlan(
        title="t",
        kind="company",
        deliverable="markdown",
        subjects=["中国银行股份有限公司"],
        period="2025Q3",
        dimensions=["同比"],
        indicators=["净息差"],
        outline=[OutlineItem("本期结论", "")],
    )
    ensured = ensure_filings_for_report(plan, request_text="生成中国银行2025Q3研报")
    assert ensured.fetched_any
    assert created_ids
    assert any("巨潮补齐" in note for note in ensured.notes)


def test_chat_tool_not_registered():
    assert "fetch_missing_reports" not in TOOLS_MAP


def test_url_ingest_rejects_bad_host(monkeypatch):
    monkeypatch.setattr(settings, "enable_cninfo_fetch", True)
    with pytest.raises(ValueError, match="白名单"):
        fr.ingest_pdf_url(
            "https://example.com/a.pdf",
            filename_stem="中国银行股份有限公司2025年第三季度报告",
        )
