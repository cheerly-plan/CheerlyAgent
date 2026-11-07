"""公司类型 → 指标模板路由：防跨行业污染与假缺失。"""

from __future__ import annotations

from ai_agent.report.industry import (
    BANK,
    EQUIPMENT,
    FABLESS,
    FOUNDRY,
    GENERIC,
    _INDUSTRIES_YAML,
    _load_profiles,
    _normalize_key,
    classify_subject,
    company_type_of,
    default_indicators,
    excluded_indicators,
    is_excluded,
    profile_of,
)
from ai_agent.report.parse import parse_request


def test_industries_yaml_loads():
    profiles = _load_profiles(_INDUSTRIES_YAML)
    assert GENERIC in profiles
    equipment = profiles[EQUIPMENT]
    assert "合同负债" in equipment.indicators
    assert "存货" in equipment.indicators
    assert "在手订单" in equipment.indicators
    assert profile_of("unknown_type_xyz").indicators == profiles[GENERIC].indicators


def test_unknown_type_falls_back_to_generic():
    assert _normalize_key("not_a_real_industry") == GENERIC
    assert default_indicators("not_a_real_industry") == default_indicators(GENERIC)


def test_classify_semiconductor_roster():
    assert classify_subject("兆易创新科技集团股份有限公司") == FABLESS
    assert classify_subject("中芯国际") == FOUNDRY
    assert classify_subject("深圳中科飞测科技股份有限公司") == EQUIPMENT
    assert classify_subject("中国银行股份有限公司") == BANK


def test_equipment_indicators_prefer_order_side():
    indicators = default_indicators(EQUIPMENT)
    banned = excluded_indicators(EQUIPMENT)
    assert "合同负债" in indicators
    assert "存货" in indicators
    assert "在手订单" in indicators
    assert "存货周转率" not in indicators
    assert "资产负债率" not in indicators
    assert "存货周转率" in banned
    assert "资产负债率" in banned
    assert is_excluded(EQUIPMENT, "产能利用率")


def test_fabless_indicators_exclude_bank_metrics():
    indicators = default_indicators(FABLESS)
    banned = excluded_indicators(FABLESS)
    assert "毛利率" in indicators
    assert "研发费用率" in indicators
    assert "经营活动产生的现金流量净额" in indicators
    assert "存货周转率" in indicators
    assert "净息差" in banned
    assert "不良贷款拨备覆盖率" in banned
    assert is_excluded(FABLESS, "PPOP")
    assert is_excluded(FABLESS, "产能利用率")
    assert not is_excluded(FABLESS, "毛利率")


def test_bank_indicators_exclude_gross_margin():
    indicators = default_indicators(BANK)
    assert "净息差" in indicators
    assert "毛利率" not in indicators
    assert is_excluded(BANK, "毛利率")


def test_parse_gye_routes_to_fabless(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [
            {
                "file_id": "f-gye",
                "filename": "兆易创新2026年第三季度报告",
                "subject": "兆易创新科技集团股份有限公司",
                "subject_aliases": ["兆易创新"],
                "period": "2026Q3",
                "status": "ready",
                "tags": [],
                "description": "",
            }
        ],
    )
    plan = parse_request("生成兆易创新2026Q3财报深度分析PDF研报")
    assert plan.company_type == FABLESS
    assert plan.skill_id == "company_financial_report"
    assert "毛利率" in plan.indicators
    assert "研发费用率" in plan.indicators
    assert "经营活动产生的现金流量净额" in plan.indicators
    assert "净息差" not in plan.indicators
    assert "不良贷款拨备覆盖率" not in plan.indicators
    assert "产能利用率" not in plan.indicators


def test_multi_type_subjects_fall_back_to_generic():
    assert company_type_of(["兆易创新", "中国银行"]) == "generic"


def test_parse_feitce_routes_to_equipment(monkeypatch):
    monkeypatch.setattr(
        "ai_agent.report.parse.store.list_kb_files",
        lambda **k: [
            {
                "file_id": "f-ft",
                "filename": "深圳中科飞测科技股份有限公司2026年第三季度报告",
                "subject": "深圳中科飞测科技股份有限公司",
                "subject_aliases": ["中科飞测"],
                "period": "2026Q3",
                "status": "ready",
                "tags": [],
                "description": "",
            }
        ],
    )
    plan = parse_request("生成中科飞测2026Q3财报深度分析PDF研报")
    assert plan.company_type == EQUIPMENT
    assert "合同负债" in plan.indicators
    assert "存货" in plan.indicators
    assert "在手订单" in plan.indicators
    assert "存货周转率" not in plan.indicators
    assert "资产负债率" not in plan.indicators
