"""skill 层契约：报告类型注册表路由、显式覆盖、积木工具安全形态。

分层原则（用户评审定稿）：skills 管「做什么报告」（大纲/指标集/维度路由），
tools 管「数字怎么来、怎么验」（抽数/计算/校验，无路径参数不写盘）。
本组用例钉住这条边界，尤其是不许通过 skill 改动防幻觉门禁。
"""

from __future__ import annotations

import uuid
from pathlib import Path

from ai_agent.report import skills as skills_module
from ai_agent.report.industry import BANK
from ai_agent.report.parse import parse_request
from ai_agent.report.skills import (
    SKILLS,
    get_skill,
    resolve_skill,
    skill_ids,
)
from ai_agent.tools.registry import TOOLS_MAP, execute_tool

_BANK_FILE = {
    "file_id": uuid.uuid4(),
    "filename": "中国银行股份有限公司2026年第三季度报告.pdf",
    "status": "ready",
    "subject": "中国银行股份有限公司",
    "subject_aliases": ["中国银行"],
    "period": "2026Q3",
    "tags": [],
    "description": "",
}
_CEEC_FILE = {
    "file_id": uuid.uuid4(),
    "filename": "中国能源建设股份有限公司2026年第三季度报告.pdf",
    "status": "ready",
    "subject": "中国能源建设股份有限公司",
    "subject_aliases": ["中国能建"],
    "period": "2026Q3",
    "tags": ["基建"],
    "description": "",
}


# ——— 路由 ———


def test_resolve_skill_prefers_industry_template_then_generic():
    assert resolve_skill("company", BANK).skill_id == "bank_financial_report"
    assert resolve_skill("company", "").skill_id == "company_financial_report"
    assert resolve_skill("industry", "").skill_id == "industry_report"
    assert (
        resolve_skill("industry", "半导体").skill_id
        == "semiconductor_industry_report"
    )
    # kind 不认识时按通用公司模板兜底，不抛错
    assert resolve_skill("unknown", "").kind == "company"


def test_semiconductor_skill_has_universe_and_outline():
    skill = get_skill("semiconductor_industry_report")
    assert skill is not None
    assert skill.kind == "industry"
    assert "半导体" in skill.industries
    codes = {code for _name, code, _seg, _aka in skill.companies}
    assert {"688981", "002371", "603501", "688256"} <= codes
    assert skill.segment_of("中芯国际") == "制造"
    assert skill.segment_of("寒武纪") == "设计"
    assert skill.segment_of("豪威集团") == "设计"
    notes = skill.rename_notes()
    assert len(notes) == 1
    assert "韦尔股份现名豪威集团（603501）" in notes[0]
    titles = [title for title, _ in skill.sections]
    assert titles[0] == "阅读说明"
    assert "赛道分组" in titles
    assert "组内对比" in titles
    assert "经营解读" in titles
    assert "缺失与口径" in titles
    assert "下期预告" in titles
    assert "景气周期" not in titles
    assert "估值" not in "".join(titles)
    assert "研发费用率" in skill.indicators


def test_semiconductor_roster_renders_from_frontmatter():
    """花名册单一数据源：{companies} 由 frontmatter 展开，不能漏花括号。"""
    skill = get_skill("semiconductor_industry_report")
    assert skill is not None
    outline = skill.build_outline(list(skill.indicators[:3]))
    briefs = " ".join(brief for _title, brief, _sid in outline)
    assert "{companies}" not in briefs
    assert "中芯国际" in briefs
    assert "寒武纪" in briefs
    assert skill.roster_text() in briefs
    assert "制造中芯国际" in skill.roster_text()
    assert "设备北方华创" in skill.roster_text()


def test_semiconductor_skill_md_has_no_duplicate_roster():
    """源文件里每个公司名只出现一次（只在 frontmatter companies 字段）。"""
    skill = get_skill("semiconductor_industry_report")
    assert skill is not None
    path = (
        Path(skills_module.__file__).resolve().parent
        / "semiconductor_industry_report"
        / "SKILL.md"
    )
    text = path.read_text(encoding="utf-8")
    for name, _code, _segment, aka in skill.companies:
        assert text.count(name) == 1, f"{name} 在 SKILL.md 出现 {text.count(name)} 次"
        if aka:
            assert text.count(aka) == 1, f"{aka} 在 SKILL.md 出现 {text.count(aka)} 次"


def test_semiconductor_empty_kb_uses_skill_companies(monkeypatch):
    """空库点「半导体行业」仍能定代表公司名单，不报无法确定标的。"""
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [],
    )
    plan = parse_request("生成半导体行业2025Q3深度分析研报")
    assert plan.kind == "industry"
    assert plan.industry == "半导体"
    assert plan.skill_id == "semiconductor_industry_report"
    assert "中芯国际" in plan.subjects
    assert "寒武纪" in plan.subjects
    assert not any("无法确定要调取哪些季报" in gap for gap in plan.gaps)
    assert any(item.title == "组内对比" for item in plan.outline)
    assert any(item.title == "经营解读" for item in plan.outline)


def test_parse_request_records_routed_skill(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_BANK_FILE],
    )
    plan = parse_request("生成中国银行2026Q3财报深度分析PDF研报")
    assert plan.skill_id == "bank_financial_report"
    assert "净息差" in plan.indicators
    assert "毛利率" not in plan.indicators


def test_generic_company_routes_to_generic_skill(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_CEEC_FILE],
    )
    plan = parse_request("生成中国能源建设2026Q3财报深度分析PDF研报")
    assert plan.skill_id == "company_financial_report"
    assert "营业收入" in plan.indicators
    assert "毛利率" in plan.indicators
    assert "净息差" not in plan.indicators


def test_outline_comes_from_skill_registry(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_BANK_FILE],
    )
    plan = parse_request("生成中国银行2026Q3财报深度分析PDF研报")
    titles = [item.title for item in plan.outline]
    skill = get_skill("bank_financial_report")
    assert titles == [title for title, _brief in skill.sections]
    # {indicators} 占位必须被实际指标集填充，不能漏出花括号
    briefs = " ".join(item.brief for item in plan.outline)
    assert "{indicators}" not in briefs
    assert "净息差" in briefs


# ——— 显式指定报告类型 ———


def test_explicit_skill_overrides_routing(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_BANK_FILE],
    )
    plan = parse_request(
        "生成中国银行2026Q3财报深度分析PDF研报", skill_id="industry_report"
    )
    assert plan.skill_id == "industry_report"
    assert plan.kind == "industry"
    assert any(item.title == "公司间对比" for item in plan.outline)
    assert "公司间对比" in plan.dimensions


def test_unknown_skill_id_falls_back_with_gap(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [_BANK_FILE],
    )
    plan = parse_request(
        "生成中国银行2026Q3财报深度分析PDF研报", skill_id="macro_report"
    )
    # 没注册的类型：照常路由 + 一条缺口说明可选值，不静默忽略
    assert plan.skill_id == "bank_financial_report"
    assert any("未知报告类型 macro_report" in gap for gap in plan.gaps)
    assert any("industry_report" in gap for gap in plan.gaps)


def test_skill_ids_match_registry():
    assert set(skill_ids()) == {skill.skill_id for skill in SKILLS}
    assert len(SKILLS) >= 3


def test_skills_load_from_skill_md_files():
    """每种报告类型对应 skills/<id>/SKILL.md，不再写死在 Python 注册表里。"""
    from pathlib import Path

    from ai_agent.report import skills as skills_mod

    root = Path(skills_mod.__file__).resolve().parent
    for skill in SKILLS:
        skill_md = root / skill.skill_id / "SKILL.md"
        assert skill_md.is_file(), f"缺少 {skill_md}"
        text = skill_md.read_text(encoding="utf-8")
        assert f"name: {skill.skill_id}" in text
        assert "## 大纲" in text
    assert (root / "bank_financial_report" / "SKILL.md").is_file()


# ——— 积木工具：无路径参数、不写盘 ———


def test_extract_tool_returns_facts_with_basis_and_missing(monkeypatch):
    from ai_agent.report.extract import Evidence, ExtractResult, IndicatorFact

    def fake_extract(subjects, period, indicators):
        return ExtractResult(
            facts=[
                IndicatorFact(
                    subject=subjects[0],
                    period=period,
                    name="净息差",
                    value="1.27",
                    unit="%",
                    column="",
                    scope="",
                    basis="point",
                    evidence=Evidence(
                        file_id="7b75379b-69a7-41ad-a670-053372460d3f",
                        filename="中国银行股份有限公司2026年第三季度报告.pdf",
                        page=7,
                        page_end=None,
                        section="3.1 经营情况简要分析",
                        quote="净息差1.27%",
                    ),
                )
            ],
            missing=["不良贷款率：表格和正文里都没有对上的一行"],
        )

    monkeypatch.setattr(
        "ai_agent.tools.report.report_extract.extract_indicators", fake_extract
    )
    out = execute_tool(
        "report_extract_indicators",
        {"subjects": ["中国银行股份有限公司"], "period": "2026Q3"},
    )
    assert "1.27%" in out
    assert "[7b75379b]" in out
    assert "第7页" in out
    assert "禁止换算" in out
    assert "不良贷款率" in out  # 缺口如实回传，不静默丢


def test_extract_tool_default_indicators_use_skill_template(monkeypatch):
    from ai_agent.report.extract import ExtractResult

    captured: dict = {}

    def fake_extract(subjects, period, indicators):
        captured["indicators"] = list(indicators)
        return ExtractResult(missing=[])

    monkeypatch.setattr(
        "ai_agent.tools.report.report_extract.extract_indicators", fake_extract
    )
    execute_tool(
        "report_extract_indicators",
        {"subjects": ["中国银行股份有限公司"], "period": "2026Q3"},
    )
    # indicators 留空 → 银行默认 21 项（skill 层的「写什么」）
    assert "净息差" in captured["indicators"]
    assert "资本充足率" in captured["indicators"]


def test_extract_tool_reports_empty_result_without_inventing(monkeypatch):
    from ai_agent.report.extract import ExtractResult

    monkeypatch.setattr(
        "ai_agent.tools.report.report_extract.extract_indicators",
        lambda *a, **k: ExtractResult(missing=["没有找到 2025Q4 的对应季报"]),
    )
    out = execute_tool(
        "report_extract_indicators",
        {"subjects": ["中国银行股份有限公司"], "period": "2025Q4"},
    )
    assert "没有抽到任何指标" in out
    assert "不要用通用知识补数" in out


def test_compare_tool_passes_basis_through(monkeypatch):
    """口径字段（cumulative/basis/coverage）必须能从积木传进计算层，
    否则 P0 的可比性门禁在工具链路上失效。"""
    out = execute_tool(
        "report_compare",
        {
            "points": [
                {
                    "subject": "中国银行股份有限公司",
                    "name": "营业收入",
                    "period": "2026H1",
                    "value": "356,903",
                    "unit": "百万元",
                    "cumulative": True,
                    "basis": "cumulative",
                    "coverage_months": 6,
                },
                {
                    "subject": "中国银行股份有限公司",
                    "name": "营业收入",
                    "period": "2026Q3",
                    "value": "526,365",
                    "unit": "百万元",
                    "cumulative": True,
                    "basis": "cumulative",
                    "coverage_months": 9,
                },
            ]
        },
    )
    # 6 个月 vs 9 个月累计：不可比，不产出增减——口径门禁在工具链路仍生效
    assert "不做跨期增减计算" in out
    assert "5.46" not in out and "-47.50%" not in out


def test_generate_report_tool_accepts_report_type(monkeypatch):
    """generate_report 降级为通用入口：report_type 透传给 run_report。"""
    import ai_agent.tools.report.report_generate as rg

    captured: dict = {}

    class FakeResult:
        status = "ok"
        document = type("D", (), {"title": "x"})()
        deliverable = "markdown"
        pdf_status = "not_requested"
        pdf_error = ""
        report_id = "abc1234567"
        review_status = "finalized"
        validation = type(
            "V", (), {"summary": "校验通过", "findings": [], "review_queue": []}
        )()
        gaps = []
        artifacts = type(
            "A", (), {"pdf": "", "markdown": "a.md", "provenance": "p.md", "review": "r.json"}
        )()

    def fake_run(request, *, skill_id="", **kwargs):
        captured["skill_id"] = skill_id
        return FakeResult()

    monkeypatch.setattr(rg, "run_report", fake_run)
    out = rg.generate_report_tool.run(
        request="生成半导体行业 2026Q3 深度分析",
        report_type="industry_report",
        write_pdf=False,
    )
    assert captured["skill_id"] == "industry_report"
    assert "/v1/report/files/a.md" in out


def test_skills_endpoint_lists_registry():
    from fastapi.testclient import TestClient

    from ai_agent.api.app import create_app
    from ai_agent.api.dependencies import get_current_user

    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: None
    client = TestClient(app)
    resp = client.get("/v1/report/skills")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == len(SKILLS)
    ids = {item["skill_id"] for item in data["skills"]}
    assert {
        "bank_financial_report",
        "company_financial_report",
        "industry_report",
        "semiconductor_industry_report",
    } <= ids
    bank = next(item for item in data["skills"] if item["skill_id"] == "bank_financial_report")
    assert "净息差" in bank["default_indicators"]
    assert "本期结论" in bank["sections"]


def test_run_report_skill_id_reaches_parse(monkeypatch):
    """run_report 显式 skill_id → parse_request(skill_id=...) → plan.skill_id。"""
    import tempfile
    from pathlib import Path

    from ai_agent.report import pipeline as pl
    from ai_agent.report.extract import Evidence, ExtractResult, IndicatorFact
    from ai_agent.report.parse import OutlineItem, ReportPlan, SourceFile

    captured: dict = {}
    subject = "中国银行股份有限公司"
    filename = "中国银行股份有限公司2026年第三季度报告.pdf"

    def fake_plan(text, skill_id=""):
        captured["skill_id"] = skill_id
        return ReportPlan(
            title=f"{subject} 2026Q3 行业视角",
            kind="industry",
            deliverable="markdown",
            subjects=[subject],
            period="2026Q3",
            dimensions=["同比"],
            indicators=["营业收入"],
            outline=[OutlineItem("覆盖范围", "")],
            files=[SourceFile("file-1", filename, subject, "2026Q3", "本期")],
            skill_id=skill_id or "industry_report",
        )

    def fake_extract(subjects, period, indicators):
        return ExtractResult(
            facts=[
                IndicatorFact(
                    subject=subjects[0],
                    period=period,
                    name="营业收入",
                    value="526,365",
                    unit="百万元",
                    column="年初至报告期末",
                    scope="",
                    evidence=Evidence(
                        file_id="file-1",
                        filename=filename,
                        page=2,
                        page_end=2,
                        section="一、主要财务数据",
                        quote="营业收入 526,365",
                    ),
                )
            ],
            missing=[],
        )

    monkeypatch.setattr(pl, "parse_request", fake_plan)
    monkeypatch.setattr(pl, "extract_indicators", fake_extract)
    monkeypatch.setattr(
        pl, "extract_supplemental", lambda *a, **k: pl.SupplementalResult()
    )
    with tempfile.TemporaryDirectory() as directory:
        result = pl.run_report(
            "生成中国银行2026Q3财报深度分析研报",
            skill_id="industry_report",
            write_pdf=False,
            output_dir=Path(directory),
        )
    assert captured["skill_id"] == "industry_report"
    assert result.plan.skill_id == "industry_report"
    assert result.validation.passed is True
    assert "526,365" in result.document.markdown


def test_registry_contains_granular_tools_without_path_params():
    for name in ("report_extract_indicators", "report_compare", "report_validate"):
        props = TOOLS_MAP[name].schema()["function"]["parameters"]["properties"]
        assert "output_path" not in props
