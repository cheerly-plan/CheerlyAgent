"""指标抽取工具：只读知识库、返回结构化事实，无路径参数。

这是「确定性积木」中最常被单独需要的一块：用户在聊天里问
「中国银行 2026Q3 净息差是多少」时，模型不必生成整份研报，
调这个工具拿到带口径和出处的数字即可。

安全边界与 generate_report 一致：不给模型 output_path、不写盘、
返回的数字只来自知识库表格/正文切片，抽不到就如实写 missing。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai_agent.prompts.tools import REPORT_EXTRACT_DESC
from ai_agent.report.basis import label as basis_label
from ai_agent.report.extract import ExtractResult, extract_indicators
from ai_agent.report.industry import company_type_of, default_indicators
from ai_agent.tools.base import Tool

# execute_tool 会把返回截到 3000 字符：事实超量时截断并如实计数，
# 不静默丢条目。
_MAX_FACTS = 18
_MAX_MISSING = 8


class ReportExtractArgs(BaseModel):
    subjects: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="公司名列表（正式名称或库内简称），例如 [\"中国银行股份有限公司\"]",
    )
    period: str = Field(
        ...,
        description="报告期，例如 2026Q3、2026H1。只认需求里写明的那一期",
    )
    indicators: list[str] = Field(
        default_factory=list,
        max_length=15,
        description=(
            "要抽的指标名（报表原文术语，如 利息净收入、不良贷款率）。"
            "留空时按公司细类型路由默认指标集（银行／Fabless／制造等）"
        ),
    )
    basis: str = Field(
        default="both",
        description=(
            "口径列：both=单季与累计都给（默认，推荐）；cumulative=只要累计；"
            "single=只要单季。季报的「本报告期」和「年初至报告期末」两列"
            "覆盖的科目并不完全相同（总资产、现金余额只出现在期末列），"
            "只给一种口径时正文里的另一组数字会缺出处"
        ),
    )
    compare_to: str = Field(
        default="",
        description=(
            "可选：要一起抽的对比期（如 2026Q2）。填了就同时返回两期的原值，"
            "模型可直接传给 report_compare 算环比；不填只抽 period 一期"
        ),
    )


def _report_extract(
    subjects: list[str],
    period: str,
    indicators: list[str],
    compare_to: str = "",
    basis: str = "cumulative",
) -> str:
    wanted = [item for item in (indicators or []) if item and item.strip()]
    if not wanted:
        # 指标留空 → 按公司细类型模板（bank/fabless/foundry…），禁止跨行业通吃。
        wanted = default_indicators(company_type_of(subjects))
    requested = (basis or "").strip().lower()
    chosen = requested if requested in {"single", "cumulative", "both"} else "both"
    result = _extract_by_basis(subjects, period, wanted, chosen)
    base = _render(result)
    other = (compare_to or "").strip()
    if not other or other == (period or "").strip():
        return base
    # 再抽一期：环比需要上一期的原值，工具层补齐比让模型多调一次更省轮次。
    previous = _extract_by_basis(subjects, other, wanted, chosen)
    prev_text = _render(previous)
    from ai_agent.report.compare import compare_measures, format_compare

    # 用 to_measure() 而不是手工构造：口径字段（basis / cumulative /
    # coverage_months / scope）由 extract 统一给出，手抄容易漂移。
    points = [fact.to_measure() for fact in [*previous.facts, *result.facts]]
    compare_text = format_compare(compare_measures(points)) if points else ""
    parts = [base, f"对比期 {other}：", prev_text]
    if compare_text:
        parts.extend(["两次抽取的增减（可直接转述，不要口算）：", compare_text])
    return "\n".join(parts)


def _extract_by_basis(
    subjects: list[str], period: str, wanted: list[str], basis: str
) -> "ExtractResult":
    """按口径抽取。``both`` = 单季与累计各抽一遍并合并。

    两列覆盖的科目并不相同：营业收入、净利润两列都有（累计优先本来会盖掉
    单季），而总资产、存货这类只有期末列。正文通常两种口径都会写，只给
    一种，另一种就全成了「未核对数字」。
    """
    if basis != "both":
        return extract_indicators(subjects, period, wanted, basis=basis)
    single = extract_indicators(subjects, period, wanted, basis="single")
    cumulative = extract_indicators(subjects, period, wanted, basis="cumulative")
    merged = ExtractResult(facts=list(single.facts), missing=list(single.missing))
    seen = {(f.subject, f.name, f.period, f.value) for f in merged.facts}
    for fact in cumulative.facts:
        key = (fact.subject, fact.name, fact.period, fact.value)
        if key in seen:
            continue
        seen.add(key)
        merged.facts.append(fact)
    return merged


def _render(result) -> str:
    lines: list[str] = []
    if not result.facts:
        lines.append("没有抽到任何指标。以下缺口如实转述，不要用通用知识补数：")
        lines.extend(f"- {item}" for item in result.missing[:_MAX_MISSING])
        return "\n".join(lines)

    lines.append(
        f"抽取到 {len(result.facts)} 条（数字与口径原样使用，禁止换算或改写；"
        "要算同比/环比请把原值传给 report_compare，不要口算）："
    )
    for fact in result.facts[:_MAX_FACTS]:
        basis = basis_label(fact.period, fact.basis, fact.coverage_months) if fact.basis else ""
        evidence = fact.evidence
        # 出处用「文件名 · 第N页」，跟 kb_search/kb_read 的抬头同一格式：
        # 模型能直接拿去 kb_read 读回，也不会把内部 file_id 抄进回答
        where = []
        if evidence.page is not None and evidence.filename:
            where.append(f"{evidence.filename} · 第{evidence.page}页")
        elif evidence.filename:
            where.append(evidence.filename)
        elif evidence.page is not None:
            where.append(f"第{evidence.page}页")
        if evidence.section:
            where.append(evidence.section)
        prefix = f"{basis}" if basis else fact.period
        lines.append(
            f"- {fact.subject} {prefix} {fact.name} = {fact.value}{fact.unit}"
            + (f"（取数列 {fact.column}）" if fact.column else "")
            + (f"（出处：{' '.join(where)}）" if where else "")
        )
    if len(result.facts) > _MAX_FACTS:
        lines.append(f"（另有 {len(result.facts) - _MAX_FACTS} 条未列出）")
    if result.missing:
        lines.append("未抽到（不要编数）：")
        lines.extend(f"- {item}" for item in result.missing[:_MAX_MISSING])
    return "\n".join(lines)


report_extract_tool = Tool(
    name="report_extract_indicators",
    description=REPORT_EXTRACT_DESC,
    func=_report_extract,
    args_model=ReportExtractArgs,
)
