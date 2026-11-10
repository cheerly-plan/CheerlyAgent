"""ingest-worker 抢任务与 API 入队行为。"""

from __future__ import annotations

import uuid
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.kb import ingest_worker as iw
from ai_agent.kb.pipeline import KbPipeline


def test_submit_enqueues_uploaded_when_not_inline(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "kb_ingest_inline", False)
    pipe = KbPipeline(workers=1, graph_workers=1)
    updates: list[tuple] = []

    def fake_update(file_id, status, **kwargs):
        updates.append((file_id, status, kwargs))

    monkeypatch.setattr("ai_agent.kb.pipeline.store.update_kb_status", fake_update)
    fid = uuid.uuid4()
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF")
    pipe.submit(fid, "u1", path)
    assert updates and updates[0][1] == "uploaded"
    pipe.shutdown(wait=False)


def test_claim_sql_uses_skip_locked():
    import inspect

    from ai_agent.kb.store import KbStore

    source = inspect.getsource(KbStore.claim_next_uploaded_kb_file)
    assert "FOR UPDATE SKIP LOCKED" in source
    assert "uploaded" in source


def test_handle_claimed_marks_failed_without_source(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "kb_upload_dir", str(tmp_path))
    fid = uuid.uuid4()
    marked: list[tuple] = []

    def fake_update(file_id, status, **kwargs):
        marked.append((str(file_id), status, kwargs.get("error", "")))

    monkeypatch.setattr(iw.store, "update_kb_status", fake_update)
    iw._handle_claimed(
        {"file_id": str(fid), "user_id": "u1", "filename": "missing.pdf"}
    )
    assert marked and marked[0][1] == "failed"
    assert "源文件" in marked[0][2]


def test_handle_claimed_calls_process(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "kb_upload_dir", str(tmp_path))
    fid = uuid.uuid4()
    folder = tmp_path / str(fid)
    folder.mkdir()
    (folder / "x.pdf").write_bytes(b"%PDF")
    called: dict = {}

    def fake_process(file_id, user_id, path):
        called.update({"file_id": file_id, "user_id": user_id, "path": Path(path)})

    monkeypatch.setattr(iw.kb_pipeline, "process_claimed", fake_process)
    iw._handle_claimed(
        {"file_id": str(fid), "user_id": "u1", "filename": "x.pdf"}
    )
    assert called["file_id"] == fid
    assert called["user_id"] == "u1"
    assert called["path"].name == "x.pdf"
