"""经营解读：受约束 LLM 后处理与流水线接入回归。"""

from __future__ import annotations

import json

from ai_agent.report.analyze import (
    EvidenceItem,
    _FALLBACK,
    _deterministic_lines,
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
    assert section.lines != [_FALLBACK]
    assert any("1.27%" in line for line in section.lines)
    assert any("不构成无出处的因果归因" in line for line in section.lines)

    document = compose_report(plan, facts, analysis=section.lines)
    result = validate_report(document, facts, plan=plan)
    assert result.passed
    assert "## 经营解读" in document.markdown
    assert _FALLBACK not in document.markdown


def test_analysis_disabled_writes_deterministic_facts():
    section = analyze_operations(
        _plan("经营解读"),
        [_fact()],
        enabled=False,
        complete=lambda *_a: (_ for _ in ()).throw(AssertionError("不应调用")),
        evidence=_evidence(),
    )
    assert any("1.27%" in line for line in section.lines)
    assert _FALLBACK not in section.lines


def test_operating_quote_strips_shareholder_splice():
    """长电类：股东转融通 + 提示语 + 经营段粘连时，只留经营成句。"""
    from ai_agent.report.analyze import _extract_operating_narrative

    raw = (
        "前 名股东及前 名无限售流通股股东因转融通出借/归还原因导致较上期发生变化 "
        "需提醒投资者关注的关于公司报告期经营情况的其他重要信息、"
        "公司前三季度围绕重点应用板块，持续优化技术、产能与客户布局，"
        "先进封装产能利用率保持高位。"
    )
    cleaned = _extract_operating_narrative(raw)
    assert "转融通" not in cleaned
    assert "需提醒投资者" not in cleaned
    assert "前 名股东" not in cleaned
    assert "围绕重点应用板块" in cleaned
    assert "产能" in cleaned

    pack = [
        EvidenceItem(
            evidence_id="e0",
            file_id=_FILE_ID,
            filename="长电科技2025Q3.pdf",
            page=3,
            section="经营情况讨论与分析",
            quote=raw,
        )
    ]
    plan = ReportPlan(
        title="长电 2025Q3",
        kind="company",
        deliverable="markdown",
        subjects=["江苏长电科技股份有限公司"],
        period="2025Q3",
        dimensions=["同比"],
        indicators=["营业收入"],
        outline=[OutlineItem("经营解读", "")],
        gaps=[],
        skill_id="company_financial_report",
    )
    body = "\n".join(_deterministic_lines(plan, [], pack))
    assert "转融通" not in body
    assert "前 名股东" not in body
    assert "围绕重点应用板块" in body


def test_operating_quote_excises_hollow_tail_keeps_narrative():
    """空缺尾巴／科目罗列要切掉，但有效经营句必须留下（不能整段扔）。"""
    from ai_agent.report.analyze import (
        _extract_operating_narrative,
        _is_usable_operating_quote,
    )

    catalog = (
        "前三季度(月)年前三季度(月)其中:营业成本 利息支出 手续费及佣金支出 "
        "退保金 赔付支出净额 提取保险责任准备金净额 保单红利支出 分保费用 "
        "税金及附加 销售费用 管理费用 研发费用"
    )
    assert _extract_operating_narrative(catalog) == ""
    assert not _is_usable_operating_quote(catalog)

    hollow_only = "扣除非经常性损益后归属于上市公司股东的净利润为人民币 亿元，同比增长 。"
    assert not _is_usable_operating_quote(hollow_only)
    assert _extract_operating_narrative(hollow_only) == ""

    # 拓斯达类：strip 数字后留下「实现营业收入 万元」不得当经营引述
    hollow_revenue = "报告期内，注塑装备业务实现营业收入 万元。"
    assert not _is_usable_operating_quote(hollow_revenue)

    # 经营句 + 空缺尾巴粘在同一段：切尾巴、留经营
    glued = (
        "公司前三季度围绕重点应用板块，持续优化技术、产能与客户布局，"
        "加速推进先进封装产能落地与配套能力升级，"
        "扣除非经常性损益后归属于上市公司股东的净利润为人民币 亿元，同比增长 。"
    )
    cleaned = _extract_operating_narrative(glued)
    assert "围绕重点应用板块" in cleaned
    assert "先进封装产能落地" in cleaned
    assert "人民币 亿元" not in cleaned
    assert "同比增长" not in cleaned

    pack = [
        EvidenceItem(
            evidence_id="e0",
            file_id=_FILE_ID,
            filename="长电科技2025Q3.pdf",
            page=3,
            section="经营情况讨论与分析",
            quote=glued,
        ),
        EvidenceItem(
            evidence_id="e1",
            file_id=_FILE_ID,
            filename="长电科技2025Q3.pdf",
            page=9,
            section="合并利润表",
            quote=catalog,
        ),
    ]
    plan = ReportPlan(
        title="长电 2025Q3",
        kind="company",
        deliverable="markdown",
        subjects=["江苏长电科技股份有限公司"],
        period="2025Q3",
        dimensions=["同比"],
        indicators=["营业收入"],
        outline=[OutlineItem("经营解读", "")],
        gaps=[],
        skill_id="company_financial_report",
    )
    body = "\n".join(_deterministic_lines(plan, [], pack))
    assert "围绕重点应用板块" in body
    assert "先进封装" in body
    assert "退保金" not in body
    assert "营业成本 利息支出" not in body
    assert "人民币 亿元" not in body
    assert "毛利率" not in body or "原文披露" in body
    assert body.count("原文披露：") >= 1


def test_volume_price_quote_not_dropped_by_change_table_headers():
    """变动原因表头不能误伤「量价齐升」单元格原文；两条关键原文都要保留。"""
    pack = [
        EvidenceItem(
            evidence_id="e0",
            file_id=_FILE_ID,
            filename="兆易创新2026年第三季度报告.pdf",
            page=2,
            section="主要会计数据、财务指标发生变动的情况、原因",
            quote=(
                "主要会计数据、财务指标发生变动的情况、原因 项目名称 变动比例(%) 主要原因 "
                "营业收入_本报告期 225.65 报告期内，存储芯片行业供给紧张，公司存储芯片产品"
                "量价齐升，存储业务持续向好；MCU等多个业务产品线持续发力，产品出货规模亦"
                "实现较好增长，整体经营业绩实现大幅增长 营业收入_年初至报告期末 197.11"
            ),
        ),
        EvidenceItem(
            evidence_id="e1",
            file_id=_FILE_ID,
            filename="兆易创新2026年第三季度报告.pdf",
            page=3,
            section="主要会计数据、财务指标发生变动的情况、原因",
            quote="利润总额_本报告期 798.62 主要是本报告期营业收入增长,毛利润增加所致",
        ),
    ]
    facts = [
        IndicatorFact(
            subject="兆易创新科技集团股份有限公司",
            period="2026Q3",
            name="毛利率",
            value="66.22",
            unit="%",
            column="年初至报告期末",
            scope="consolidated",
            evidence=Evidence(
                file_id="188a6f60-0000-0000-0000-000000000001",
                filename="akshare.stock_financial_abstract",
                page=None,
                page_end=None,
                section="财务摘要",
                quote="毛利率 采用=66.22%",
            ),
        )
    ]
    plan = ReportPlan(
        title="兆易创新 2026Q3",
        kind="company",
        deliverable="markdown",
        subjects=["兆易创新科技集团股份有限公司"],
        period="2026Q3",
        dimensions=["同比"],
        indicators=["营业收入", "毛利率"],
        outline=[OutlineItem("经营解读", "")],
        gaps=[],
        skill_id="company_financial_report",
    )
    lines = _deterministic_lines(plan, facts, pack)
    body = "\n".join(lines)
    assert body.count("原文披露：") == 2
    assert "量价齐升" in body
    assert "毛利润增加" in body
    assert "与原文「量价齐升／毛利润增加」并列" in body


def test_structure_line_omits_volume_price_if_not_cited():
    pack = [
        EvidenceItem(
            evidence_id="e1",
            file_id=_FILE_ID,
            filename="t.pdf",
            page=3,
            section="变动原因",
            quote="主要是本报告期营业收入增长,毛利润增加所致",
        )
    ]
    facts = [
        IndicatorFact(
            subject="测试公司",
            period="2026Q3",
            name="毛利率",
            value="40",
            unit="%",
            column="年初至报告期末",
            scope="consolidated",
            evidence=Evidence(
                file_id="a" * 36,
                filename="akshare.stock_financial_abstract",
                page=None,
                page_end=None,
                section="",
                quote="毛利率 40%",
            ),
        )
    ]
    plan = ReportPlan(
        title="t",
        kind="company",
        deliverable="markdown",
        subjects=["测试公司"],
        period="2026Q3",
        dimensions=[],
        indicators=["毛利率"],
        outline=[OutlineItem("经营解读", "")],
        gaps=[],
    )
    body = "\n".join(_deterministic_lines(plan, facts, pack))
    assert "毛利润增加" in body
    assert "量价齐升" not in body


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
