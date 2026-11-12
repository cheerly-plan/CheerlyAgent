"""多主体提问：按公司分别检索、分组返回。

混在一次检索里排序时，一家会挤掉另一家的名额——实测问「中微跟兆易比」，
top3 里前两条是兆易，因为 2026Q3 的数字和「营业收入」更贴近。
"""

import uuid
from unittest.mock import patch

from ai_agent.kb.retrieval.retrieval import Hit
from ai_agent.tools.kb import kb_search as ks

A_ID = "aaaaaaaa-1111-2222-3333-444455556666"
B_ID = "bbbbbbbb-1111-2222-3333-444455556666"


def _hit(file_id: str, name: str, text: str, period: str = "2026Q3") -> Hit:
    return Hit(
        file_id=file_id,
        filename=name,
        seq=1,
        page=1,
        kind="table",
        text=text,
        distance=0.2,
        section="一、主要财务数据",
        period=period,
    )


def _files() -> list[dict]:
    return [
        {"file_id": A_ID, "filename": "兆易创新2026年第三季度报告",
         "subject": "兆易创新科技集团股份有限公司", "period": "2026Q3", "status": "ready"},
        {"file_id": B_ID, "filename": "中微公司2025年第三季度报告",
         "subject": "中微半导体设备（上海）股份有限公司", "period": "2025Q3", "status": "ready"},
    ]


def _fake_retrieve(query, files=None, top_k=3, kinds=None, sections=None):
    """按 files 参数返回对应公司的命中，模拟逐家检索。"""
    subject = (files or [None])[0]
    if subject and "中微" in subject:
        return [uuid.UUID(B_ID)], [_hit(B_ID, "中微公司2025年第三季度报告", "中微 营业收入 80.63 亿", "2025Q3")]
    return [uuid.UUID(A_ID)], [_hit(A_ID, "兆易创新2026年第三季度报告", "兆易 营业收入 202.97 亿")]


def test_comparison_question_splits_by_subject(monkeypatch):
    monkeypatch.setattr(ks.store, "list_kb_files", lambda: _files())
    monkeypatch.setattr(ks.store, "has_ready_kb_file", lambda: True)
    monkeypatch.setattr(
        "ai_agent.kb.retrieval.router.mentioned_file_ids",
        lambda text: [uuid.UUID(A_ID), uuid.UUID(B_ID)],
    )
    monkeypatch.setattr(ks, "retrieve_for_question", _fake_retrieve)
    monkeypatch.setattr(ks.settings, "kb_search_list_only", False)

    out = ks._kb_search("中微公司 营业收入")
    assert "按主体分别检索" in out
    assert "【兆易创新科技集团股份有限公司】" in out
    assert "【中微半导体设备（上海）股份有限公司】" in out
    # 每组只含本家片段
    first, second = out.split("【中微半导体设备")[0], out.split("【中微半导体设备")[1]
    assert "兆易" in first and "中微 营业收入" not in first


def test_single_subject_keeps_single_search(monkeypatch):
    monkeypatch.setattr(ks.store, "list_kb_files", lambda: _files())
    monkeypatch.setattr(
        "ai_agent.kb.retrieval.router.mentioned_file_ids",
        lambda text: [uuid.UUID(A_ID)],
    )
    assert ks._search_by_subject("兆易创新 营业收入", top_k=3, kinds=None, sections=None) is None


def test_missing_subject_reports_gap(monkeypatch):
    monkeypatch.setattr(ks.store, "list_kb_files", lambda: _files())
    monkeypatch.setattr(
        "ai_agent.kb.retrieval.router.mentioned_file_ids",
        lambda text: [uuid.UUID(A_ID), uuid.UUID(B_ID)],
    )

    def only_first(query, files=None, top_k=3, kinds=None, sections=None):
        if files and "中微" in files[0]:
            return [], []
        return [uuid.UUID(A_ID)], [_hit(A_ID, "兆易创新2026年第三季度报告", "兆易 营业收入")]

    monkeypatch.setattr(ks, "retrieve_for_question", only_first)
    out = ks._search_by_subject("对比", top_k=3, kinds=None, sections=None)
    assert "【中微半导体设备（上海）股份有限公司】库里没有检索到相关内容。" in out


def test_lost_name_fallback_only_when_all_subjects_dropped(monkeypatch):
    """模型保留了主体就别换回整句；全收掉了才换。"""
    import ai_agent.kb.retrieval.router as router

    monkeypatch.setattr(router, "mentioned_file_ids", lambda text: [uuid.UUID(A_ID)] if "兆易" in text else [])
    assert ks._question_names_files_query_lost("营业收入", "兆易创新的营收") is True

    monkeypatch.setattr(
        router,
        "mentioned_file_ids",
        lambda text: [uuid.UUID(A_ID), uuid.UUID(B_ID)] if "中微" in text and "兆易" in text else [uuid.UUID(B_ID)],
    )
    assert ks._question_names_files_query_lost("中微公司 营业收入", "中微跟兆易比谁多") is False
