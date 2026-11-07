"""校验研报正文。数字和引用都不交给模型判断。

报告里出现的金额、增减、幅度，必须能在抽取结果或对比计算结果里找到。
每条本期结论还要能回到原文片段。对不上的进人工审核队列，不直接当定稿。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai_agent.report import registry
from ai_agent.report.compare import Change, Measure, SeriesCompare, compare_measures
from ai_agent.report.compose import ReportDocument
from ai_agent.report.extract import IndicatorFact, citation_anchor
from ai_agent.report.parse import ReportPlan
from ai_agent.report.supplemental import SupplementalResult

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
# 与 extract._NUMBER 对齐：半年报摘要用空格做千分位（「356 903」），
# 正则必须把整组当一个数，否则会把 356 和 903 拆成两个未核对数字。
_NUMBER = re.compile(r"-?\d[\d,，]*(?:[ \t\u00a0]\d{3})*(?:\.\d+)?%?")
_PAGE = re.compile(r"第\s*(\d+)(?:\s*[–-]\s*(\d+))?\s*页")
_PERIOD_YEAR = re.compile(r"((?:19|20)\d{2})")
# 短锚点形如 [7b75379b]；经营解读也可能引用非本期指标表里的同文件片段。
_SHORT_ANCHOR = re.compile(r"\[[0-9a-fA-F]{6,12}\]")
# 归因词：有则同句须见页码或短锚点，否则进审核队列（warning，不直接 error）。
_ATTRIBUTION = re.compile(
    r"(因为|由于|导致|致使|拉动|拖累|归因于|受[^。；;]{0,12}影响|贡献于)"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；\n])")


@dataclass
class Finding:
    kind: str
    severity: str
    message: str
    value: str = ""
    subject: str = ""
    name: str = ""
    period: str = ""
    file_id: str = ""


@dataclass
class ReviewItem:
    reason: str
    value: str = ""
    subject: str = ""
    name: str = ""
    period: str = ""
    file_id: str = ""
    quote: str = ""


@dataclass
class ProvenanceAttachment:
    markdown: str
    items: list[dict] = field(default_factory=list)


@dataclass
class ValidationResult:
    passed: bool
    findings: list[Finding] = field(default_factory=list)
    review_queue: list[ReviewItem] = field(default_factory=list)
    provenance: ProvenanceAttachment = field(
        default_factory=lambda: ProvenanceAttachment(markdown="")
    )

    @property
    def summary(self) -> str:
        if self.passed and not self.review_queue:
            return "校验通过，可以定稿。"
        if self.passed:
            return f"校验通过，但仍有 {len(self.review_queue)} 条需人工审核。"
        errors = sum(1 for item in self.findings if item.severity == "error")
        return f"校验未通过：{errors} 条错误，{len(self.review_queue)} 条待人工审核。"


def validate_report(
    document: ReportDocument,
    facts: list[IndicatorFact],
    plan: ReportPlan | None = None,
    series: list[SeriesCompare] | None = None,
    views: SupplementalResult | None = None,
) -> ValidationResult:
    """核对正文数字和引用。异常写入 review_queue，不改正文。"""
    supplemental = views or SupplementalResult()
    body = _plain(document.markdown) or _plain(document.html)
    computed = series if series is not None else compare_measures([_measure(fact) for fact in facts])
    result = ValidationResult(passed=True, provenance=_provenance(facts))
    allowed = _allowed(facts, computed, plan, supplemental)
    _check_views(result, supplemental)

    # 推导值的硬门禁：链不完整就不许放行，正文里出现它的数字会被判 error。
    # 这是双保险——compare 本来只在链齐全时产出 derived Change，但若有别的
    # 调用方手工塞进来（或将来加了润色），也不能让无出处的数字溜进正文。
    for item in computed:
        for period, change in item.qoq.items():
            if change.source == "derived" and not _derivation_trusted(change, facts):
                _error(
                    result,
                    "number",
                    f"{item.subject}{item.name} {period} 的推导环比缺完整出处链，不得写入正文",
                    value=change.rate,
                )

    if not body:
        _error(result, "number", "正文为空，无法校验数字")
    else:
        _check_facts(result, facts, plan, body)
        _check_money_units(result, facts, plan)
        _check_body_numbers(result, _body_for_numbers(body, facts, supplemental), allowed)
        _check_citations_in_body(result, facts, plan, body)
        _check_attribution_anchors(result, body)
        _check_lead_yoy_consistency(result, document.markdown or body, facts, computed, plan)

    result.passed = not any(item.severity == "error" for item in result.findings)
    return result


def format_validation(result: ValidationResult) -> str:
    """给后续步骤看的纯文本。"""
    lines = [result.summary]
    for item in result.findings:
        lines.append(f"[{item.severity}/{item.kind}] {item.message}")
    if result.review_queue:
        lines.append("人工审核队列")
        for item in result.review_queue:
            where = " / ".join(part for part in (item.subject, item.name, item.period) if part)
            lines.append(f"- {item.reason}" + (f"（{where}）" if where else ""))
    if result.provenance.markdown:
        lines.append("")
        lines.append(result.provenance.markdown.rstrip())
    return "\n".join(lines).strip() + "\n"


def _measure(fact: IndicatorFact) -> Measure:
    return Measure(
        subject=fact.subject,
        name=fact.name,
        period=fact.period,
        value=fact.value,
        unit=fact.unit,
        scope=fact.scope,
        reported_yoy=fact.reported_yoy,
        cumulative=fact.cumulative,
        source_key=(fact.evidence.file_id or fact.evidence.filename or "").strip(),
    )


def _plain(text: str) -> str:
    if not text:
        return ""
    cleaned = _TAG.sub(" ", text)
    return _SPACE.sub(" ", cleaned).strip()


def _compact(value: str) -> str:
    return _SPACE.sub("", (value or "").replace("，", ","))


def _number_key(value: str) -> str:
    text = _compact(value)
    found = _NUMBER.search(text)
    return _compact(found.group(0)) if found else text


def _derivation_trusted(change: Change, facts: list[IndicatorFact]) -> bool:
    """推导环比必须凑齐整条出处链，且链上每个来源都能在抽取结果里查到。"""
    if change.source != "derived" or len(change.evidence) < 3:
        return False
    known = {
        key.strip()
        for fact in facts
        for key in (fact.evidence.file_id, fact.evidence.filename)
        if key and key.strip()
    }
    return all(key and key in known for key in change.evidence)


def _allowed(
    facts: list[IndicatorFact],
    series: list[SeriesCompare],
    plan: ReportPlan | None,
    views: SupplementalResult,
) -> set[str]:
    allowed: set[str] = set()
    for view in views.quarters:
        for raw in (
            view.quarter_value,
            view.quarter_yoy,
            view.cumulative_value,
            view.cumulative_yoy,
        ):
            key = _number_key(raw)
            if key:
                allowed.add(key)
    for view in views.year_ends:
        for raw in (view.value, view.change, view.prior_value):
            key = _number_key(raw)
            if key:
                allowed.add(key)
    from ai_agent.report.supplemental import format_money_delta, signed_delta

    for view in views.balance_moves:
        for raw in (view.end_value, view.prior_value):
            key = _number_key(raw)
            if key:
                allowed.add(key)
        # 增减额由两列相减，必须进白名单，否则构成表被校验拦下。
        delta = signed_delta(view.end_value, view.prior_value)
        if delta is not None:
            key = _number_key(format_money_delta(delta))
            if key:
                allowed.add(key)
            for other in views.year_ends:
                if "所有者权益" not in other.name and "净资产" not in other.name:
                    continue
                whole = signed_delta(other.value, other.prior_value)
                if whole is None or whole == 0:
                    continue
                # 正文用 f"{share:.0f}%"，与 round 对齐进白名单
                allowed.add(str(int(round(float(delta / whole * 100)))))
    for view in views.year_end_reasons:
        quote = view.reason or ""
        for match in _NUMBER.finditer(quote):
            key = _number_key(match.group(0))
            if key:
                allowed.add(key)
        if view.evidence and view.evidence.page is not None:
            allowed.add(str(view.evidence.page))
    # 归母权益增量（主表两列相减）也要进白名单
    for other in views.year_ends:
        if not other.prior_value:
            continue
        whole = signed_delta(other.value, other.prior_value)
        if whole is not None:
            key = _number_key(format_money_delta(whole))
            if key:
                allowed.add(key)
    for fact in facts:
        key = _number_key(fact.value)
        if key:
            allowed.add(key)
        if fact.evidence.page is not None:
            allowed.add(str(fact.evidence.page))
        if fact.evidence.page_end is not None:
            allowed.add(str(fact.evidence.page_end))
        for year in _PERIOD_YEAR.findall(fact.period or ""):
            allowed.add(year)
    for item in series:
        # 缺口文案里会出现期间号（如「与 2025Q3 期间口径不同」），和 plan.gaps
        # 一样把年份加进 allowed，免得正文的说明文字被当成未核对数字。
        for gap in item.gaps:
            for match in _NUMBER.finditer(gap):
                allowed.add(_compact(match.group(0)))
        for _period, value in item.points:
            key = _number_key(value)
            if key:
                allowed.add(key)
        for change in (*item.yoy.values(), *item.qoq.values()):
            if change.source == "derived" and not _derivation_trusted(change, facts):
                # 链不完整的推导数字不进 allowed：正文里出现就判「未核对数字」
                continue
            for raw in (change.delta, change.rate, change.base_value):
                key = _number_key(raw)
                if key:
                    allowed.add(key)
            for year in _PERIOD_YEAR.findall(change.base_period or ""):
                allowed.add(year)
        for step in item.trend:
            for raw in (step.delta, step.rate):
                key = _number_key(raw)
                if key:
                    allowed.add(key)
            for period in (step.earlier, step.later):
                for year in _PERIOD_YEAR.findall(period or ""):
                    allowed.add(year)
    if plan is not None:
        for year in _PERIOD_YEAR.findall(plan.period or ""):
            allowed.add(year)
        for gap in plan.gaps:
            for match in _NUMBER.finditer(gap):
                allowed.add(_compact(match.group(0)))
        # 技能名单里的证券代码可出现在阅读说明／脚注，不算未核对数字。
        if plan.skill_id:
            from ai_agent.report.skills import get_skill

            skill = get_skill(plan.skill_id)
            if skill is not None:
                for _name, code, _segment in skill.companies:
                    key = _compact(code or "")
                    if key:
                        allowed.add(key)
    return allowed


def _check_views(result: ValidationResult, views: SupplementalResult) -> None:
    """单季/较上年末视图也过一遍原文核对，不能因为换了章节就免检。"""
    for view in views.quarters:
        _check_view_evidence(
            result,
            view.name,
            view.evidence,
            (view.quarter_value, view.quarter_yoy, view.cumulative_value, view.cumulative_yoy),
        )
    for view in views.year_ends:
        _check_view_evidence(result, view.name, view.evidence, (view.value,))
        quote = view.evidence.quote if view.evidence else ""
        change = (view.change or "").strip()
        number = _view_number(change)
        if number and not _value_in_text(number, quote):
            _error(
                result,
                "citation",
                f"{view.name} 的较上年末变动 {change} 未出现在原文片段里",
                value=change,
            )
        # 方向是从原文的「下降/增长」读出来的，必须能在片段里找到对应词
        words = ("下降", "减少") if change.startswith("-") else ("增长", "上升", "增加")
        if change and not any(word in quote for word in words):
            _error(
                result,
                "citation",
                f"{view.name} 的变动方向 {change} 与原文片段不符",
                value=change,
            )
    for view in views.balance_moves:
        _check_view_evidence(
            result, view.name, view.evidence, (view.end_value, view.prior_value)
        )
    for view in views.year_end_reasons:
        if not view.evidence or not (view.reason or "").strip():
            _error(result, "citation", f"{view.name} 变动原因缺少原文片段", value="")
            continue
        if view.reason not in (view.evidence.quote or ""):
            # quote 形如「名称：原因」，允许包含关系
            if view.reason not in (view.evidence.quote or "") and not (
                view.evidence.quote or ""
            ).endswith(view.reason):
                _error(
                    result,
                    "citation",
                    f"{view.name} 变动原因未出现在原文片段里",
                    value=view.reason[:80],
                )


def _check_view_evidence(result, name: str, evidence, values: tuple[str, ...]) -> None:
    if evidence is None or not (evidence.file_id or evidence.filename):
        _error(result, "citation", f"{name} 的视图没有文档出处")
        return
    quote = evidence.quote or ""
    if not quote.strip():
        _error(result, "citation", f"{name} 的视图没有原文片段")
        return
    for raw in values:
        number = _view_number(raw)
        if number and not _value_in_text(number, quote):
            _error(
                result,
                "citation",
                f"{name} 的视图数值 {raw} 未出现在原文片段里",
                value=raw,
            )


def _view_number(value: str) -> str:
    return _number_key(value).lstrip("+-").rstrip("%")


_MONEY_SCALE_UNITS = frozenset(registry.ordering_list("money_scale_units"))
_MONEY_INDICATORS = registry.names_with("money_validation")
_QUOTE_SCALE_UNIT = re.compile(
    r"单位[:：]\s*(?:人民币)?(千元|万元|百万元|亿元)"
)


def _check_money_units(
    result: ValidationResult,
    facts: list[IndicatorFact],
    plan: ReportPlan | None,
) -> None:
    """对比表金额必须统一为元：源单位千元／万元等须已折算，否则 error。"""
    focus = plan.period if plan is not None else ""
    for fact in facts:
        if focus and fact.period != focus:
            continue
        # 已登记指标按 flags.money 判定；未登记的回退原有白名单。
        if not registry.flag(
            fact.name, "money", default=fact.name in _MONEY_INDICATORS
        ):
            continue
        unit = (fact.unit or "").strip()
        if unit in _MONEY_SCALE_UNITS:
            _error(
                result,
                "unit",
                f"{fact.subject}{fact.period}{fact.name} 金额单位仍为{unit}，"
                "须折算为元后再写入同表对比",
                fact=fact,
            )
            continue
        quote = fact.evidence.quote if fact.evidence else ""
        matched = _QUOTE_SCALE_UNIT.search(quote or "")
        if (
            matched
            and unit in {"", "元", "人民币元"}
            and not getattr(fact, "unit_converted", False)
        ):
            _error(
                result,
                "unit",
                f"{fact.subject}{fact.period}{fact.name} 原文为{matched.group(1)}，"
                "但未折算为元（疑似单位丢失）",
                fact=fact,
            )


def _check_facts(
    result: ValidationResult,
    facts: list[IndicatorFact],
    plan: ReportPlan | None,
    body: str,
) -> None:
    focus = plan.period if plan is not None else ""
    for fact in facts:
        evidence = fact.evidence
        if not fact.value.strip():
            _error(
                result,
                "number",
                f"{fact.subject}{fact.period}{fact.name} 没有数值",
                fact=fact,
            )
            continue
        if not evidence.quote.strip():
            _error(
                result,
                "citation",
                f"{fact.subject}{fact.period}{fact.name} 没有原文片段",
                fact=fact,
            )
        elif not _value_in_text(fact.value, evidence.quote):
            _error(
                result,
                "citation",
                f"{fact.subject}{fact.period}{fact.name} 的数值未出现在原文片段里",
                fact=fact,
            )
        if not evidence.file_id.strip() and not evidence.filename.strip():
            _error(
                result,
                "citation",
                f"{fact.subject}{fact.period}{fact.name} 没有文档出处",
                fact=fact,
            )
        if evidence.page is None and not _structured_source(evidence.filename):
            _review(
                result,
                f"{fact.subject}{fact.period}{fact.name} 没有页码，需人工核对出处",
                fact=fact,
            )
        if focus and fact.period == focus and body and not _value_in_text(fact.value, body):
            _review(
                result,
                f"本期指标 {fact.name}={fact.value} 未写入正文",
                fact=fact,
            )


def _structured_source(filename: str) -> bool:
    """akshare / derived 等结构化源无原文页码，不进人工页码队列。"""
    name = (filename or "").strip().lower()
    return name.startswith(("akshare.", "derived."))


_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_YOY_FRAME = registry.ordering_list("yoy_frame")


def _section_text(markdown: str, title: str) -> str:
    if not markdown:
        return ""
    matches = list(_SECTION.finditer(markdown))
    for index, match in enumerate(matches):
        if match.group(1).strip() != title:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        return markdown[start:end]
    return ""


def _check_lead_yoy_consistency(
    result: ValidationResult,
    markdown: str,
    facts: list[IndicatorFact],
    series: list[SeriesCompare],
    plan: ReportPlan | None,
) -> None:
    """同一指标在「本期结论」与「同比与环比」的方向表述必须一致。"""
    from ai_agent.report.rules import render_yoy_rate_bit

    lead = _section_text(markdown, "本期结论")
    yoy_sec = _section_text(markdown, "同比与环比")
    if not lead or not yoy_sec:
        return
    period = plan.period if plan is not None else ""
    by_key = {(fact.subject, fact.name, fact.period): fact for fact in facts}
    for item in series:
        change = item.yoy.get(period) if period else None
        if change is None or not change.rate:
            continue
        fact = by_key.get((item.subject, item.name, period))
        current = fact.value if fact is not None else change.delta
        expected = render_yoy_rate_bit(current or "", change)
        if not expected:
            continue
        # 结论未点名该指标则不查；点名了就必须与正文同一套方向标签。
        if item.name not in lead:
            continue
        if expected in lead:
            continue
        framing = next((token for token in _YOY_FRAME if token in expected), "")
        if framing and framing in yoy_sec and f"同比 {change.rate}" in lead:
            _error(
                result,
                "consistency",
                f"本期结论与同比节表述不一致：{item.name} 正文为「{framing}」，"
                f"结论仍写「同比 {change.rate}」",
                fact=fact,
                value=change.rate,
            )
            continue
        if framing and framing in yoy_sec and framing not in lead:
            _error(
                result,
                "consistency",
                f"本期结论缺少与同比节一致的方向词：{item.name} 应为「{expected}」",
                fact=fact,
                value=change.rate,
            )


def _body_for_numbers(
    body: str, facts: list[IndicatorFact], views: SupplementalResult
) -> str:
    """去掉出处标记再扫数字，避免把 file_id / 短锚点 / 页码里的数字当成金额。"""
    text = body
    markers: list[str] = []
    for fact in facts:
        markers.extend(
            (
                fact.evidence.file_id,
                fact.evidence.filename,
                citation_anchor(fact.evidence.file_id),
            )
        )
    for view in (*views.quarters, *views.year_ends):
        if view.evidence:
            markers.extend(
                (
                    view.evidence.file_id,
                    view.evidence.filename,
                    citation_anchor(view.evidence.file_id),
                )
            )
    for marker in sorted({item for item in markers if item and item.strip()}, key=len, reverse=True):
        text = text.replace(marker, " ")
    text = _PAGE.sub(" ", text)
    # 经营解读等章节可能挂上指标表未列到的同文件锚点，一律剔除短锚点数字。
    text = _SHORT_ANCHOR.sub(" ", text)
    return text


def _check_attribution_anchors(result: ValidationResult, body: str) -> None:
    """归因句若同句看不到页码/短锚点 → 进审核队列，不直接当 error。"""
    if not body:
        return
    for sentence in _SENTENCE_SPLIT.split(body):
        text = sentence.strip()
        if not text or not _ATTRIBUTION.search(text):
            continue
        if _PAGE.search(text) or _SHORT_ANCHOR.search(text):
            continue
        # 「不推断 / 未披露」类 gap 句允许无锚点
        if any(token in text for token in ("未披露", "不推断", "无法判断", "从略", "不构成因果")):
            continue
        result.findings.append(
            Finding(
                kind="attribution",
                severity="warning",
                message=f"归因句缺少出处锚点：{text[:60]}",
                value=text[:80],
            )
        )
        _review(result, f"归因句缺少出处锚点：{text[:60]}", value=text[:80])


def _check_body_numbers(result: ValidationResult, body: str, allowed: set[str]) -> None:
    if not body.strip():
        # 原文可能只剩出处被剥掉后的空串；真正空正文在外层已有事实检查兜底
        return
    for match in _NUMBER.finditer(body):
        raw = match.group(0)
        key = _compact(raw)
        if not key or key in {".", "-", "%"}:
            continue
        if key in allowed:
            continue
        stripped = key.lstrip("+-").rstrip("%")
        if any(allowed_key.lstrip("+-").rstrip("%") == stripped for allowed_key in allowed):
            continue
        # 单独的一两位数多半是章节序号，不进审核
        if re.fullmatch(r"-?\d{1,2}", key):
            continue
        _error(result, "number", f"正文出现未核对数字 {raw}", value=raw)


def _check_citations_in_body(
    result: ValidationResult,
    facts: list[IndicatorFact],
    plan: ReportPlan | None,
    body: str,
) -> None:
    if not body:
        return
    focus = plan.period if plan is not None else ""
    seen: set[tuple[str, str, str, str]] = set()
    for fact in facts:
        if focus and fact.period != focus:
            continue
        # 赛道均值等推算项：数字已在白名单，不要求正文再挂出处锚点。
        if fact.scope == "derived" or (fact.evidence.filename or "").startswith(
            "derived."
        ):
            continue
        key = (fact.subject, fact.name, fact.period, fact.scope)
        if key in seen:
            continue
        seen.add(key)
        if not _value_in_text(fact.value, body):
            continue
        markers = [
            part
            for part in (
                fact.evidence.file_id,
                fact.evidence.filename,
                citation_anchor(fact.evidence.file_id),
            )
            if part and part.strip()
        ]
        if not markers:
            continue
        if any(marker in body for marker in markers):
            continue
        _review(
            result,
            f"正文写了 {fact.name}={fact.value}，但看不到出处标记",
            fact=fact,
        )
        result.findings.append(
            Finding(
                kind="citation",
                severity="error",
                message=f"正文写了 {fact.name}={fact.value}，但看不到出处标记",
                value=fact.value,
                subject=fact.subject,
                name=fact.name,
                period=fact.period,
                file_id=fact.evidence.file_id,
            )
        )


def _provenance(facts: list[IndicatorFact]) -> ProvenanceAttachment:
    items: list[dict] = []
    lines = ["# 溯源附件", ""]
    if not facts:
        lines.append("没有可附的抽取结果。")
        return ProvenanceAttachment(markdown="\n".join(lines) + "\n", items=items)
    lines.extend(
        [
            "| 公司 | 报告期 | 指标 | 数值 | 口径 | 文件 | 页码 | 章节 | 原文 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for fact in facts:
        page = ""
        if fact.evidence.page is not None:
            if fact.evidence.page_end and fact.evidence.page_end != fact.evidence.page:
                page = f"{fact.evidence.page}–{fact.evidence.page_end}"
            else:
                page = str(fact.evidence.page)
        quote = _SPACE.sub(" ", fact.evidence.quote).strip()
        item = {
            "subject": fact.subject,
            "period": fact.period,
            "name": fact.name,
            "value": f"{fact.value}{fact.unit}",
            "scope": fact.scope,
            "file_id": fact.evidence.file_id,
            "filename": fact.evidence.filename,
            "page": page,
            "section": fact.evidence.section,
            "quote": quote,
        }
        items.append(item)
        lines.append(
            "| {subject} | {period} | {name} | {value} | {scope} | {filename}（{file_id}） | "
            "{page} | {section} | {quote} |".format(**{key: _cell(value) for key, value in item.items()})
        )
    return ProvenanceAttachment(markdown="\n".join(lines) + "\n", items=items)


def _cell(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _value_in_text(value: str, text: str) -> bool:
    compact = _compact(value)
    if not compact:
        return False
    return compact in _compact(text)


def _error(
    result: ValidationResult,
    kind: str,
    message: str,
    *,
    fact: IndicatorFact | None = None,
    value: str = "",
) -> None:
    finding = Finding(
        kind=kind,
        severity="error",
        message=message,
        value=value or (fact.value if fact else ""),
        subject=fact.subject if fact else "",
        name=fact.name if fact else "",
        period=fact.period if fact else "",
        file_id=fact.evidence.file_id if fact else "",
    )
    result.findings.append(finding)
    _review(result, message, fact=fact, value=finding.value)


def _review(
    result: ValidationResult,
    reason: str,
    *,
    fact: IndicatorFact | None = None,
    value: str = "",
) -> None:
    item = ReviewItem(
        reason=reason,
        value=value or (fact.value if fact else ""),
        subject=fact.subject if fact else "",
        name=fact.name if fact else "",
        period=fact.period if fact else "",
        file_id=fact.evidence.file_id if fact else "",
        quote=fact.evidence.quote if fact else "",
    )
    key = (item.reason, item.value, item.subject, item.name, item.period, item.file_id)
    existing = {
        (row.reason, row.value, row.subject, row.name, row.period, row.file_id)
        for row in result.review_queue
    }
    if key not in existing:
        result.review_queue.append(item)
