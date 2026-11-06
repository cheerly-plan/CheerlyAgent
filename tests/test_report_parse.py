"""研报需求只认知识库里的公司和写明的报告期，不补一家没出现的公司。"""

import uuid

from ai_agent.report.parse import parse_request

_CEEC = {
    "file_id": uuid.uuid4(),
    "filename": "中国能源建设股份有限公司2025年第三季度报告.pdf",
    "status": "ready",
    "subject": "中国能源建设股份有限公司",
    "subject_aliases": ["中国能建"],
    "period": "2025Q3",
    "tags": [],
    "description": "",
}
_CEEC_YOY = {
    "file_id": uuid.uuid4(),
    "filename": "中国能源建设股份有限公司2024年第三季度报告.pdf",
    "status": "ready",
    "subject": "中国能源建设股份有限公司",
    "subject_aliases": ["中国能建"],
    "period": "2024Q3",
    "tags": [],
    "description": "",
}
_GOLD = {
    "file_id": uuid.uuid4(),
    "filename": "中国黄金2025年第三季度报告.pdf",
    "status": "ready",
    "subject": "中国黄金集团黄金珠宝股份有限公司",
    "subject_aliases": ["中国黄金"],
    "period": "2025Q3",
    "tags": ["半导体"],
    "description": "",
}


def test_company_request_names_period_files_and_outline(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_CEEC, _CEEC_YOY, _GOLD],
    )
    plan = parse_request("生成中国能源建设 2025Q3财报深度分析PDF研报")
    assert plan.kind == "company"
    assert plan.deliverable == "pdf"
    assert plan.period == "2025Q3"
    assert plan.subjects == ["中国能源建设股份有限公司"]
    assert plan.indicators == ["营业收入", "扣非净利润", "毛利率"]
    assert plan.dimensions == ["同比", "环比"]
    roles = {(item.period, item.role) for item in plan.files}
    assert ("2025Q3", "本期") in roles
    assert ("2024Q3", "同比基期") in roles
    assert plan.files[0].file_id == str(_CEEC["file_id"])
    assert "中国黄金" not in plan.title
    assert any(item.title == "同比与环比" for item in plan.outline)
    # A 股没有独立二季报：环比基期不再是找不到的「2025Q2」，而是
    # 推导单季所需的 Q1 / H1 累计报告。缺口要指名缺哪份累计报告，
    # 不能把「本期季报表格里已披露的同比」也笼统说成无法调取。
    assert any("A 股无独立二季报" in gap and "2025Q1" in gap for gap in plan.gaps)
    assert any("A 股无独立二季报" in gap and "2025H1" in gap for gap in plan.gaps)
    assert not any("2025Q2" in gap for gap in plan.gaps)
    assert ("2025Q1", "累计基期") not in roles
    assert ("2025H1", "累计基期") not in roles


def test_industry_request_does_not_invent_members(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_CEEC, _CEEC_YOY],
    )
    plan = parse_request("半导体行业 2025Q3深度分析报告")
    assert plan.kind == "industry"
    assert plan.industry == "半导体"
    assert plan.deliverable == "markdown"
    assert plan.subjects == []
    assert plan.files == []
    assert plan.dimensions == ["同比", "环比", "公司间对比"]
    assert any("没有标成半导体行业" in gap for gap in plan.gaps)


def test_industry_members_come_from_file_tags(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_CEEC, _GOLD],
    )
    plan = parse_request("半导体行业 2025Q3深度分析报告")
    assert plan.subjects == ["中国黄金集团黄金珠宝股份有限公司"]
    assert [item.role for item in plan.files] == ["本期"]
    assert "中国能源建设" not in plan.subjects


def test_named_indicator_replaces_the_default_set(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_CEEC],
    )
    plan = parse_request("中国能源建设 2025Q3 营收分析报告")
    assert plan.indicators == ["营业收入"]


def test_missing_period_is_not_guessed(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_CEEC],
    )
    plan = parse_request("写一份中国能源建设财报深度分析")
    assert plan.period == ""
    assert plan.files == []
    assert "没有写明报告期" in plan.gaps
