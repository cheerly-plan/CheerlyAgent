"""经营解读：受约束 LLM 后处理与流水线接入回归。"""

from __future__ import annotations

import json

from ai_agent.report.analyze import (
    EvidenceItem,
    _FALLBACK,
    analyze_operations,
)
from ai_agent.report.compose import compose_report
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan
from ai_agent.report.skills import get_skill
from ai_agent.report.validate import validate_report

_SUBJECT = "中国银行股份有限公司"
_FILE_ID = "7b75379b-69a7-41ad-a670-053372460d3f"
_FILENAME = "中国银行股份有限公司2026年第三季度报告.pdf"


def _fact(
    *,
    name: str = "净息差",
    value: str = "1.27",
    unit: str = "%",
    page: int = 5,
) -> IndicatorFact:
    return IndicatorFact(
        subject=_SUBJECT,
        period="2026Q3",
        name=name,
        value=value,
        unit=unit,
        column="年初至报告期末",
        scope="",
        evidence=Evidence(
            file_id=_FILE_ID,
            filename=_FILENAME,
            page=page,
            page_end=page,
            section="主要财务数据",
            quote=f"{name} {value}{unit}",
        ),
    )


def _plan(*titles: str) -> ReportPlan:
    return ReportPlan(
        title=f"{_SUBJECT} 2026Q3 财报深度分析",
        kind="company",
        deliverable="markdown",
        subjects=[_SUBJECT],
        period="2026Q3",
        dimensions=["同比"],
        indicators=["净息差", "营业收入"],
        outline=[OutlineItem(title, "") for title in titles],
        gaps=[],
        skill_id="bank_financial_report",
    )


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            evidence_id="e0",
            file_id=_FILE_ID,
            filename=_FILENAME,
            page=8,
            section="经营情况简要分析",
            quote="净息差 1.27%，较上年同期下降，主要由于资产端收益率下行。",
        )
    ]


def test_bank_and_company_outline_include_analysis_section():
    for skill_id in ("bank_financial_report", "company_financial_report"):
        titles = [title for title, _ in get_skill(skill_id).sections]
        assert "经营解读" in titles
        assert titles.index("经营解读") == titles.index("数据观察") + 1


def test_unauthorized_number_paragraph_dropped():
    payload = {
        "paragraphs": [
            {
                "topic": "nim",
                "text": "净息差 1.27%，同业均值约 1.55%，处于偏低区间",
                "evidence_ids": ["e0"],
                "claim_kind": "comparison",
            }
        ],
        "gaps": [],
    }

    def complete(_system: str, _user: str) -> str:
        return json.dumps(payload, ensure_ascii=False)

    section = analyze_operations(
        _plan("经营解读"),
        [_fact()],
        enabled=True,
        complete=complete,
        evidence=_evidence(),
    )
    assert section.used_llm is True
    assert section.lines == [_FALLBACK]
    assert any("未授权数字" in gap for gap in section.gaps)


def test_causal_without_evidence_dropped():
    payload = {
        "paragraphs": [
            {
                "topic": "nim",
                "text": "净息差 1.27% 因为资产端收益率下行而收窄",
                "evidence_ids": [],
                "claim_kind": "comparison",
            }
        ],
        "gaps": [],
    }

    def complete(_system: str, _user: str) -> str:
        return json.dumps(payload, ensure_ascii=False)

    section = analyze_operations(
        _plan("经营解读"),
        [_fact()],
        enabled=True,
        complete=complete,
        evidence=_evidence(),
    )
    assert section.lines == [_FALLBACK]
    assert any("缺少有效证据" in gap for gap in section.gaps)


def test_valid_analysis_renders_with_anchor_and_passes_validate():
    payload = {
        "paragraphs": [
            {
                "topic": "nim",
                "text": "净息差为 1.27%，季报称主要由于资产端收益率下行",
                "evidence_ids": ["e0"],
                "claim_kind": "restatement",
            }
        ],
        "gaps": ["季报未披露同业对比口径"],
    }

    def complete(_system: str, _user: str) -> str:
        return json.dumps(payload, ensure_ascii=False)

    facts = [_fact()]
    plan = _plan("本期结论", "主要指标与出处", "经营解读", "缺失与口径")
    section = analyze_operations(
        plan,
        facts,
        enabled=True,
        complete=complete,
        evidence=_evidence(),
    )
    assert section.used_llm is True
    assert any("1.27%" in line for line in section.lines)
    assert any("[7b75379b]" in line for line in section.lines)
    assert any("第8页" in line for line in section.lines)

    document = compose_report(plan, facts, analysis=section.lines)
    assert "## 经营解读" in document.markdown
    analysis_body = document.markdown.split("## 经营解读", 1)[1].split("\n## ", 1)[0]
    assert "1.27%" in analysis_body
    assert "[7b75379b]" in analysis_body
    assert _FALLBACK not in analysis_body

    result = validate_report(document, facts, plan=plan)
    assert result.passed
    assert not any(item.kind == "attribution" for item in result.findings)


def test_llm_error_falls_back_without_blocking():
    def complete(_system: str, _user: str) -> str:
        raise RuntimeError("boom")

    facts = [_fact()]
    plan = _plan("本期结论", "主要指标与出处", "经营解读")
    section = analyze_operations(
        plan,
        facts,
        enabled=True,
        complete=complete,
        evidence=_evidence(),
    )
    assert section.used_llm is False
    assert section.lines == [_FALLBACK]

    document = compose_report(plan, facts, analysis=section.lines)
    result = validate_report(document, facts, plan=plan)
    assert result.passed
    assert "## 经营解读" in document.markdown
    assert _FALLBACK in document.markdown


def test_analysis_disabled_writes_fallback():
    section = analyze_operations(
        _plan("经营解读"),
        [_fact()],
        enabled=False,
        complete=lambda *_a: (_ for _ in ()).throw(AssertionError("不应调用")),
    )
    assert section.lines == [_FALLBACK]


def test_attribution_without_anchor_goes_to_review_queue():
    facts = [_fact()]
    plan = _plan("本期结论", "主要指标与出处", "经营解读")
    document = compose_report(
        plan,
        facts,
        analysis=["净息差收窄导致利润承压，资产端收益率下行。"],
    )
    result = validate_report(document, facts, plan=plan)
    # 归因缺锚点只进审核队列，不直接卡死整份
    assert result.passed
    assert any("归因句缺少出处锚点" in item.reason for item in result.review_queue)
