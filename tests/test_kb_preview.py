"""知识库页检索演示：没勾选文件时查全库，不走对话那条文件名收窄。"""

import uuid

import pytest
from fastapi import HTTPException

from ai_agent.api.schemas import KbSearchRequest
from ai_agent.kb.retrieval import Hit
from ai_agent.memory.db import UserRecord


_USER = UserRecord(user_id="u1", username="u1", display_name="u1")


def _hit() -> Hit:
    return Hit(
        file_id=str(uuid.uuid4()),
        filename="三季报.pdf",
        seq=2,
        page=5,
        kind="text",
        text="普通股股东总数为52,513名。",
        distance=0.2,
        section="股东情况",
    )


def test_preview_without_files_searches_whole_library(monkeypatch):
    from ai_agent.api.routers import kb as kb_mod

    seen: dict = {}

    def fake(query, *, file_ids, top_k):
        seen["query"] = query
        seen["file_ids"] = file_ids
        seen["top_k"] = top_k
        return [_hit()]

    monkeypatch.setattr(kb_mod, "preview_search", fake)
    payload = kb_mod.search_preview(KbSearchRequest(query="  股东人数  "), _USER)
    assert seen == {"query": "股东人数", "file_ids": None, "top_k": 5}
    assert payload["scoped"] is False
    assert payload["hits"][0]["where"] == "第5页"
    assert payload["hits"][0]["section"] == "股东情况"


def test_preview_with_files_passes_those_ids(monkeypatch):
    from ai_agent.api.routers import kb as kb_mod

    fid = uuid.uuid4()
    seen: dict = {}
    monkeypatch.setattr(
        kb_mod,
        "preview_search",
        lambda query, *, file_ids, top_k: seen.update(file_ids=file_ids) or [],
    )
    kb_mod.search_preview(
        KbSearchRequest(query="营收", file_ids=[str(fid)]),
        _USER,
    )
    assert seen["file_ids"] == [fid]


def test_preview_rejects_bad_file_id():
    from ai_agent.api.routers import kb as kb_mod

    with pytest.raises(HTTPException) as raised:
        kb_mod.search_preview(KbSearchRequest(query="营收", file_ids=["nope"]), _USER)
    assert raised.value.status_code == 400
