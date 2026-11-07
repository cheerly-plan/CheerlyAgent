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
    assert "营业收入" in plan.indicators
    assert "毛利率" in plan.indicators
    assert "净息差" not in plan.indicators
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


def test_industry_request_uses_skill_companies_when_kb_untagged(monkeypatch):
    """库里没有标成该行业的文件时，仍用 skill 宇宙名单，不编造库外公司。"""
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_CEEC, _CEEC_YOY],
    )
    plan = parse_request("半导体行业 2025Q3深度分析报告")
    assert plan.kind == "industry"
    assert plan.industry == "半导体"
    assert plan.deliverable == "pdf"
    assert plan.skill_id == "semiconductor_industry_report"
    assert "中芯国际" in plan.subjects
    assert "中国能源建设" not in plan.subjects
    assert plan.dimensions == ["同比", "环比", "公司间对比"]
    assert not any("无法确定要调取哪些季报" in gap for gap in plan.gaps)


def test_markdown_only_request_skips_pdf(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_CEEC, _CEEC_YOY],
    )
    plan = parse_request("生成中国能源建设2025Q3财报深度分析，只要 Markdown")
    assert plan.deliverable == "markdown"


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


_SKYVERSE = {
    "file_id": uuid.uuid4(),
    "filename": "深圳中科飞测科技股份有限公司2026年第三季度报告.pdf",
    "status": "ready",
    "subject": "深圳中科飞测科技股份有限公司",
    "subject_aliases": [],
    "period": "2026Q3",
    "tags": [],
    "description": "",
}
_DIRTY_SUBJECT = {
    "file_id": uuid.uuid4(),
    "filename": "中微公司2025年第三季度报告.pdf",
    "status": "ready",
    "subject": "股份有限公司",
    "subject_aliases": [],
    "period": "2025Q3",
    "tags": [],
    "description": "",
}


def test_short_company_name_hits_unique_subject_fragment(monkeypatch):
    """「中科飞测」不在别名表也能点到全称——与 kb 检索点名口径一致。"""
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_SKYVERSE, _CEEC],
    )
    for query in (
        "生成中科飞测2026Q3财报分析",
        "生成深圳中科飞测2026Q3财报分析",
    ):
        plan = parse_request(query)
        assert plan.kind == "company", query
        assert plan.subjects == ["深圳中科飞测科技股份有限公司"], query
        assert plan.period == "2026Q3"
        assert not any("没有点到知识库里的公司" in gap for gap in plan.gaps)


def test_implausible_subject_not_pulled_by_legal_suffix(monkeypatch):
    """需求写了真全称时，脏 subject「股份有限公司」不能跟着命中。"""
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_SKYVERSE, _DIRTY_SUBJECT],
    )
    plan = parse_request("生成深圳中科飞测科技股份有限公司2026Q3研报")
    assert plan.subjects == ["深圳中科飞测科技股份有限公司"]
    assert "股份有限公司" not in plan.subjects
