"""把深度分析结构化结论渲染成 PDF 附录 HTML（推断层，显式标注）。

与 ``schema.to_markdown()`` 的区别：这份 HTML 是要**印进对外 PDF** 的，
所以多三层防误读措施：

1. 附录强制另起一页，首页有醒目警示框，声明「不属于已核验财报数据」；
2. 每条结论带【已知】/【推断】/【不确定】标签 + Tier + 证据 id + 置信度，
   推断与不确定行单独底色；
3. 页眉右上角挂 running element「推断层 · 未经事实层校验」——它只在附录
   出现，正文页眉不受影响（``position: running`` 是本项目 Logo 已在用的机制，
   不依赖 WeasyPrint 的 named page 特性，避免版本兼容风险）。

安全：所有来自 LLM / 外部数据源的文本一律 ``html.escape``，禁止模型输出
把 ``<script>``、``<iframe>`` 或伪造表格注入进 PDF。本模块只依赖标准库与
同包 schema，不引用 ``report/`` 的业务对象，也不发任何 LLM 调用。

数字口径：附录里的数字只来自 ``DeepAnalysisResult``（已被 analyzer 的台账
硬门禁过滤）。情景概率、置信度是**模型主观赋值**，在警示框里明确说明，
不是财报披露值，也不进事实层 ``validate.py`` 的允许数字集合。
"""

from __future__ import annotations

import html
from typing import Any

from ai_agent.report.deep.schema import QUALITY_LABELS, DeepAnalysisResult

# 合并进 PDF 后必须进人工审核队列：这条 reason 由 pipeline 复用。
APPENDIX_REVIEW_REASON = (
    "深度分析附录已合并进 PDF：含大模型推断结论，"
    "需人工确认推断标注、证据边界与免责声明完整性"
)
# 双 PDF 档的审核项文案定义在 review.py（门禁侧），避免两处文案漂移。
APPENDIX_FLAG = "推断层 · 未经事实层校验"

_CLAIM_CLASS = {
    "known": "deep-claim-known",
    "inferred": "deep-claim-inferred",
    "uncertain": "deep-claim-uncertain",
}


def _esc(value: Any) -> str:
    """文本一律转义；None 当空串，避免把 'None' 印进 PDF。"""
    return html.escape(str(value if value is not None else ""), quote=True)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{_esc(cell)}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _claim_table(claims) -> str:
    """结论表：推断/不确定行加底色，避免和【已知】混成一片。"""
    if not claims:
        return ""
    head = (
        "<tr><th>类型</th><th>结论</th><th>Tier</th>"
        "<th>证据 id</th><th>置信度</th></tr>"
    )
    rows: list[str] = []
    for claim in claims:
        css = _CLAIM_CLASS.get(claim.claim_type, "deep-claim-uncertain")
        evidence = "、".join(claim.evidence_ids) or "无直接证据"
        rows.append(
            f'<tr class="{css}"><td>【{_esc(claim.label)}】</td>'
            f"<td>{_esc(claim.text)}</td><td>{_esc(claim.tier)}</td>"
            f"<td>{_esc(evidence)}</td><td>{claim.confidence:.2f}</td></tr>"
        )
    return (
        '<table class="deep-claims"><thead>'
        + head
        + "</thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _list(items: list[str], *, empty: str = "无记录。") -> str:
    if not items:
        return f"<p>{_esc(empty)}</p>"
    return "<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in items) + "</ul>"


def render_deep_appendix_html(
    result: DeepAnalysisResult, *, standalone: bool = False
) -> str:
    """渲染推断层 section。返回值插到打印模板的 ``</body>`` 前，或作为独立 PDF 全文。

    ``standalone=True`` 时这份 section 是**独立深度 PDF 的全部内容**：去掉
    「附录」措辞与首页强制分页（否则首页会空一页），警示文案改为指向配套的
    事实层底稿 PDF。两种形态共用同一套三分标签、Tier、证据 id 与转义逻辑。
    """
    execution = result.execution
    if standalone:
        section_class = "deep-appendix deep-standalone"
        notice = [
            '<p class="deep-notice-title">【警示】独立深度投资分析报告（推断层）</p>',
            "<p>本报告由大模型基于事实台账生成，含【推断】与【不确定】结论，"
            "<strong>不属于已核验财报数据</strong>。全部量化事实以配套生成的"
            "事实层底稿 PDF 及其溯源附件为准；本报告不构成估值或买卖建议。</p>",
            "<p>与底稿的分界：底稿数字来自财报原文并逐条通过校验门禁；本报告数字"
            "仅来自深度分析台账，情景概率与置信度是模型主观赋值，不是财报披露值。"
            "本 PDF 有独立的人工审核记录，未经批准不得作为定稿件外发。</p>",
        ]
        heading = (
            f"<h1>{_esc(result.target)} {_esc(result.period)} 深度投资分析（推断层）</h1>"
        )
    else:
        section_class = "deep-appendix"
        notice = [
            '<p class="deep-notice-title">【警示】附录 · 深度投资分析（推断层）</p>',
            "<p>本附录由大模型基于事实台账生成，含【推断】与【不确定】结论，"
            "<strong>不属于已核验财报数据</strong>。全部量化事实以本报告正文及"
            "其溯源附件为准；本附录不构成估值或买卖建议。</p>",
            "<p>与正文的分界：正文数字来自财报原文并逐条通过校验门禁；本附录数字"
            "仅来自深度分析台账，情景概率与置信度是模型主观赋值，不是财报披露值。"
            "附录已随本 PDF 一并进入人工审核队列，未经批准不得作为定稿件外发。</p>",
        ]
        heading = f"<h1>附录：{_esc(result.target)} {_esc(result.period)} 深度投资分析</h1>"
    parts: list[str] = [
        f'<section class="{section_class}" id="deep-analysis-appendix">',
        f'<div class="deep-flag">{_esc(APPENDIX_FLAG)}</div>',
        '<div class="deep-notice">',
        *notice,
        "</div>",
        heading,
        "<h2>附录 A · 执行报告</h2>",
        _table(
            ["项", "值"],
            [
                ["分析框架", execution.framework],
                ["分析模式", "公司" if result.mode == "company" else "行业"],
                ["启用章节", ", ".join(str(n) for n in execution.enabled_chapters)],
                ["跳过章节", ", ".join(str(n) for n in execution.skipped_chapters) or "无"],
                ["评审轮次", str(execution.review_rounds)],
                ["LLM 调用次数", str(execution.llm_calls)],
                ["章节/评审模型", execution.chapter_model or "未启用"],
                ["第 9 章收敛模型", execution.convergence_model or "未启用"],
                ["模型路由", execution.model_route or "未启用 LLM"],
                [
                    "收敛质量门禁",
                    QUALITY_LABELS.get(execution.convergence_quality, "未执行收敛"),
                ],
                [
                    "估算 token（输入/输出）",
                    f"{execution.estimated_input_tokens} / {execution.estimated_output_tokens}",
                ],
            ],
        ),
        _list(list(execution.notes), empty="无执行说明。"),
    ]

    parts.extend(["<h2>附录 B · 候选主线（假设，非结论）</h2>"])
    if result.company_type:
        parts.append(f"<p>公司类型：{_esc(result.company_type)}</p>")
    if result.mainlines:
        parts.append(
            "<ul>"
            + "".join(
                f"<li><strong>{_esc(item.business)}</strong>（{_esc(item.rule)}）"
                f" → 初始链条：{_esc(item.chain)}；{_esc(item.tier)}；"
                f"{_esc(item.status)}</li>"
                for item in result.mainlines
            )
            + "</ul>"
        )
    else:
        parts.append("<p>已抽取指标不足以生成候选主线，记为数据缺口。</p>")

    parts.append("<h2>附录 C · 分章结论（假设—验证—结论闭环）</h2>")
    if not result.chapters:
        parts.append("<p>本次未产出章节结论（零 token 骨架或章节被跳过）。</p>")
    for chapter in result.chapters:
        parts.extend(
            [
                f"<h3>第 {chapter.number} 章 {_esc(chapter.title)}"
                '<span class="deep-badge">推断层</span></h3>',
                f"<p><strong>核心问题</strong>：{_esc(chapter.question)}</p>",
                f"<p><strong>方法</strong>：{_esc(chapter.method)}</p>",
                "<p class=\"deep-verdict\"><strong>章节结论</strong>："
                f"{_esc(chapter.verdict or '未形成结论（数据不足）')}</p>",
            ]
        )
        if chapter.claims:
            parts.extend(["<h4>结论与证据</h4>", _claim_table(chapter.claims)])
        if chapter.sources:
            parts.extend(
                [
                    "<h4>数据来源</h4>",
                    _table(
                        ["数据项", "源头文件", "页码/位置", "Tier", "获取方式"],
                        [
                            [r.item, r.source_file, r.location, r.tier, r.acquisition]
                            for r in chapter.sources
                        ],
                    ),
                ]
            )
        if chapter.gaps:
            parts.extend(
                [
                    "<h4>数据缺口</h4>",
                    _table(
                        ["缺口数据", "涉及章节", "影响", "处置"],
                        [
                            [
                                g.item,
                                ",".join(str(n) for n in g.chapters),
                                g.impact,
                                g.handling,
                            ]
                            for g in chapter.gaps
                        ],
                    ),
                ]
            )
        if chapter.review:
            parts.extend(
                [
                    "<h4>专家评审记录</h4>",
                    _table(
                        ["轮次", "专家", "意见", "处理", "修订/理由"],
                        [
                            [
                                f"第{o.round_no}轮",
                                o.expert,
                                o.issue,
                                o.disposition_label,
                                o.revision or o.reason,
                            ]
                            for o in chapter.review
                        ],
                    ),
                ]
            )
        parts.append(f'<p class="deep-status">章节状态：{_esc(chapter.status)}</p>')
        if chapter.fallback_reason:
            parts.append(
                '<p class="deep-status">降级原因：'
                f'{_esc(chapter.fallback_reason)}</p>'
            )

    if result.convergence is not None:
        convergence = result.convergence
        parts.extend(
            [
                "<h2>附录 D · 核心链条收敛与护城河</h2>",
                '<p class="deep-verdict"><strong>收敛质量门禁（代码级检查）</strong>：'
                f"{_esc(QUALITY_LABELS.get(execution.convergence_quality, '未执行收敛'))}</p>",
                _list(
                    list(execution.convergence_checks),
                    empty="本次未执行收敛质量检查。",
                ),
                "<p><strong>核心链条终稿</strong>："
                f"{_esc(convergence.final_chain or '未收敛')}</p>",
                "<p><strong>与候选主线的差异</strong>："
                f"{_esc(convergence.delta or '未记录')}</p>",
                _list(
                    [
                        f"{d.get('mainline', '')} → {d.get('decision', '')}："
                        f"{d.get('reason', '')}"
                        for d in convergence.decisions
                    ],
                    empty="无主线处置记录。",
                ),
                "<p><strong>护城河综合评级（推断）</strong>："
                f"{_esc(convergence.moat_rating or '未评级')}</p>",
                _list(
                    [f"{name}：{rating}" for name, rating in convergence.moat_sources.items()],
                    empty="无护城河分项证据。",
                ),
            ]
        )

    if result.checklist:
        parts.extend(
            [
                "<h2>附录 E · 个人 checklist</h2>",
                _table(
                    ["检查项", "结果", "说明"],
                    [[c.item, c.result, c.note] for c in result.checklist],
                ),
            ]
        )

    parts.append("<h2>附录 F · 跟踪、证伪与情景</h2>")
    parts.append(
        _table(
            ["关键判断", "先行指标", "验证/证伪方向", "数据来源", "频率"],
            [
                [t.judgment, t.indicator, t.direction, t.source, t.frequency]
                for t in result.tracking
            ],
        )
        or "<p>判断不足以支撑跟踪指标，风险更高。</p>"
    )
    parts.append(
        _table(
            ["判断", "证伪信号", "严重程度"],
            [[f.judgment, f.signal, f.severity] for f in result.falsifiers],
        )
    )
    parts.append(
        _table(
            ["情景", "概率（模型主观赋值）", "关键假设", "触发条件", "对判断的影响"],
            [
                [
                    s.name,
                    f"{s.probability:.0%}",
                    s.assumption,
                    s.trigger,
                    s.impact,
                ]
                for s in result.scenarios
            ],
        )
    )

    parts.extend(
        [
            "<h2>附录 G · 数据缺口总表</h2>",
            _table(
                ["缺口数据", "涉及章节", "影响", "处置"],
                [
                    [
                        g.item,
                        ",".join(str(n) for n in g.chapters),
                        g.impact,
                        g.handling,
                    ]
                    for g in result.all_gaps
                ],
            )
            or "<p>无显式缺口（不代表数据完备，仅代表本次未被识别）。</p>",
            "<h2>附录 H · 跨章一致性审查</h2>",
            _list(list(result.cross_checks), empty="本次未记录跨章一致性问题。"),
            '<div class="deep-footer">',
            "<p>帮助验证，不替你做决定。证伪信号触发时应回查对应章节修订评级，"
            "而不是删除本报告。本附录不构成估值或买卖建议。</p>",
            "</div>",
            "</section>",
        ]
    )
    text = "\n".join(parts)
    if standalone:
        # 独立 PDF 不是任何报告的附录：小节保留 A-H 编号便于与底稿对照，
        # 但去掉「附录」字样，并把正文里所有「本附录」统一换成「本报告」。
        text = text.replace("<h2>附录 ", "<h2>").replace("本附录", "本报告")
    return text


def deep_standalone_document(result: DeepAnalysisResult) -> str:
    """把推断层 section 包成完整 HTML 文档，交给 ``render_html_pdf`` 排版。

    ``render_report_pdf`` 吃的是 HTML ``ReportDocument``，直接把 ``.deep.md``
    的 Markdown 喂进去会印成一坨没样式的纯文本；这里复用已经带三分标签、
    Tier、证据 id、警示框与转义的 HTML，避免另写一套渲染逻辑。
    """
    section = render_deep_appendix_html(result, standalone=True)
    title = f"{result.target} {result.period} 深度投资分析（推断层）"
    return (
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        f"<title>{_esc(title)}</title></head><body>{section}</body></html>"
    )
