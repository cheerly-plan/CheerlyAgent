"""研报产物清理：同标题去重 + 按天过期。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ai_agent.report.cleanup import cleanup_reports, prune_older_editions
from ai_agent.report.review import list_index_records, remove_records


def _touch(path: Path, *, age_days: float = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    if age_days:
        stamped = time.time() - age_days * 86400
        path.touch()
        import os

        os.utime(path, (stamped, stamped))


def _edition(root: Path, title: str, report_id: str, *, age_days: float = 0) -> None:
    stem = f"{title}_{report_id}"
    for name in (f"{stem}.md", f"{stem}.pdf", f"{stem}.provenance.md", f"{stem}.review.json"):
        _touch(root / name, age_days=age_days)
    index = root / "_review" / "index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    if index.is_file():
        records = json.loads(index.read_text(encoding="utf-8")).get("records", {})
    records[report_id] = {
        "report_id": report_id,
        "title": title,
        "status": "finalized",
        "artifacts": {
            "markdown": f"{stem}.md",
            "pdf": f"{stem}.pdf",
            "provenance": f"{stem}.provenance.md",
            "review": f"{stem}.review.json",
        },
    }
    index.write_text(
        json.dumps({"version": 1, "records": records}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_keep_latest_per_title_deletes_older_editions(tmp_path: Path):
    title = "中国银行股份有限公司 2026Q3 财报深度分析"
    _edition(tmp_path, title, "aaaaaaaaaa", age_days=2)
    _edition(tmp_path, title, "bbbbbbbbbb", age_days=0)
    _edition(tmp_path, "另一家公司 2026Q3 财报深度分析", "cccccccccc", age_days=0)

    result = cleanup_reports(tmp_path, keep_latest=1, retention_days=0)
    assert result.deleted_report_ids == ["aaaaaaaaaa"]
    assert not (tmp_path / f"{title}_aaaaaaaaaa.md").exists()
    assert (tmp_path / f"{title}_bbbbbbbbbb.pdf").exists()
    assert (tmp_path / "另一家公司 2026Q3 财报深度分析_cccccccccc.md").exists()
    assert "aaaaaaaaaa" not in list_index_records(tmp_path)
    assert "bbbbbbbbbb" in list_index_records(tmp_path)


def test_retention_days_deletes_stale_reports(tmp_path: Path):
    _edition(tmp_path, "旧报告", "aaaaaaaa01", age_days=40)
    _edition(tmp_path, "新报告", "bbbbbbbb01", age_days=1)

    result = cleanup_reports(tmp_path, keep_latest=0, retention_days=30)
    assert result.deleted_report_ids == ["aaaaaaaa01"]
    assert not (tmp_path / "旧报告_aaaaaaaa01.md").exists()
    assert (tmp_path / "新报告_bbbbbbbb01.md").exists()


def test_prune_older_editions_only_affects_same_title(tmp_path: Path, monkeypatch):
    title = "中国银行股份有限公司 2026Q3 财报深度分析"
    _edition(tmp_path, title, "aaaaaaaa01", age_days=1)
    _edition(tmp_path, title, "bbbbbbbb01", age_days=0)
    _edition(tmp_path, "旁路报告", "cccccccc01", age_days=0)

    monkeypatch.setattr(
        "ai_agent.report.cleanup.settings.report_keep_latest_per_title", 1
    )
    result = prune_older_editions(
        title, keep_report_id="bbbbbbbb01", output_dir=tmp_path
    )
    assert "aaaaaaaa01" in result.deleted_report_ids
    assert (tmp_path / f"{title}_bbbbbbbb01.md").exists()
    assert (tmp_path / "旁路报告_cccccccc01.md").exists()


def test_remove_records_is_idempotent(tmp_path: Path):
    _edition(tmp_path, "x", "deadbeef01")
    assert remove_records(["deadbeef01"], tmp_path) == 1
    assert remove_records(["deadbeef01"], tmp_path) == 0


def _deep_edition(root: Path, title: str, report_id: str, *, age_days: float = 0) -> None:
    """主记录 + 三份 .deep.* 文件 + `<id>-deep` 审核记录（文件名里是主 id）。"""
    _edition(root, title, report_id, age_days=age_days)
    # 深度产物名比正文多一个「报告」后缀（pipeline.deep_report_title），
    # 和真实数据一致；_scan 必须用正文标题归一化版本标题。
    stem = f"{title}报告_{report_id}"
    deep_pdf = root / f"{stem}.deep.pdf"
    deep_md = root / f"{stem}.deep.md"
    deep_json = root / f"{stem}.deep.json"
    for path in (deep_pdf, deep_md, deep_json):
        _touch(path, age_days=age_days)
    index = root / "_review" / "index.json"
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["records"][f"{report_id}-deep"] = {
        "report_id": f"{report_id}-deep",
        "title": f"{title}报告（推断层）",
        "status": "pending",
        "artifacts": {
            "markdown": deep_md.name,
            "provenance": deep_json.name,
            "pdf": deep_pdf.name,
            "review": "",
        },
    }
    index.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_cleanup_keeps_deep_record_whose_files_still_exist(tmp_path: Path):
    """<id>-deep 的文件名写的是主 id，_scan 配不到 edition；不能当残留删掉。

    历史 bug：删掉后 .deep.pdf 因「没有审核记录」既不能下载也不能预览。
    """
    title = "中国银行 2026Q3 财报深度分析"
    _deep_edition(tmp_path, title, "aaaaaaaa11")

    cleanup_reports(tmp_path, keep_latest=1, retention_days=30)

    records = list_index_records(tmp_path)
    assert "aaaaaaaa11" in records
    assert "aaaaaaaa11-deep" in records
    assert (tmp_path / f"{title}报告_aaaaaaaa11.deep.pdf").exists()


def test_cleanup_deletes_deep_record_with_its_main_edition(tmp_path: Path):
    title = "中国银行 2026Q3 财报深度分析"
    _deep_edition(tmp_path, title, "aaaaaaaa12", age_days=2)
    _deep_edition(tmp_path, title, "bbbbbbbb12", age_days=0)

    result = cleanup_reports(tmp_path, keep_latest=1, retention_days=0)

    assert "aaaaaaaa12" in result.deleted_report_ids
    records = list_index_records(tmp_path)
    assert "aaaaaaaa12-deep" not in records
    assert "bbbbbbbb12-deep" in records
