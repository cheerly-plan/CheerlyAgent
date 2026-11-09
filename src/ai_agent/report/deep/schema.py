"""深度分析层的结构化结论对象。

与 ``report/`` 事实层的边界：事实层输出 Markdown 字符串（给人看），本层输出
可序列化、可查询、可跟踪、可证伪的结论对象（给系统和后续复盘看）。

三条硬约束，对应 investment-deep-analysis v2.0 的输出规范：

1. 每条结论都要三分：``known``（有 L1 原文）/ ``inferred``（推断）/ ``uncertain``；
2. 每个量化数据都要标 Tier、回指证据 id，数据缺口显式列出；
3. 报告是「可证伪的假设集合」——关键判断必须配跟踪指标与证伪信号。

本模块只定义数据形状，不含 LLM 调用、不引用 report/ 的业务对象。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

CLAIM_TYPES = ("known", "inferred", "uncertain")
TIERS = ("Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5")
_CLAIM_LABEL = {"known": "已知", "inferred": "推断", "uncertain": "不确定"}


@dataclass
class Claim:
    """一条结论。``inferred`` 必须带证据或低置信度，不允许伪装成事实。"""

    text: str
    claim_type: str = "known"
    tier: str = "Tier 1"
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.8

    @property
    def label(self) -> str:
        return _CLAIM_LABEL.get(self.claim_type, "不确定")


@dataclass
class DataSourceRow:
    """数据来源表一行：回答「这个数字最初出现在哪个文件」。"""

    item: str
    source_file: str
    location: str
    tier: str
    acquisition: str


@dataclass
class DataGap:
    """数据缺口：禁止臆测填补，且缺口若在结论中成为关键变量须标注低置信度。"""

    item: str
    chapters: list[int] = field(default_factory=list)
    impact: str = ""
    handling: str = "不推断，列入待补数据源"


@dataclass
class ReviewOpinion:
    """一条评审意见。每条必须被采纳或书面驳回，不允许忽略。"""

    round_no: int
    expert: str
    issue: str
    disposition: str = "adopted"  # adopted / rejected
    reason: str = ""
    revision: str = ""

    @property
    def disposition_label(self) -> str:
        return "采纳修订" if self.disposition == "adopted" else "驳回"


@dataclass
class ChapterResult:
    """一章的定稿结果。章节是「假设-验证-结论」闭环的验证单元。"""

    number: int
    title: str
    question: str
    method: str
    verdict: str = ""
    claims: list[Claim] = field(default_factory=list)
    sources: list[DataSourceRow] = field(default_factory=list)
    gaps: list[DataGap] = field(default_factory=list)
    review: list[ReviewOpinion] = field(default_factory=list)
    status: str = "draft"  # draft / reviewed / skipped / fallback
    # 降级原因（空响应 / reasoning-only / JSON 解析失败 / 调用异常）；成功章节为空串。
    # 与 status 一起进 .deep.json / .deep.md / 深度 PDF，便于事后定位而不必重跑。
    fallback_reason: str = ""

    @property
    def reviewed(self) -> bool:
        return self.status == "reviewed"


@dataclass
class Mainline:
    """第 1 章候选主线：注意它是假设，不是结论。"""

    business: str
    rule: str
    chain: str
    tier: str = "Tier 1"
    status: str = "待验证"


@dataclass
class Convergence:
    """第 9 章收敛点：对候选主线做确认／修正／推翻，并显式记录差异。"""

    final_chain: str = ""
    decisions: list[dict] = field(default_factory=list)
    delta: str = ""
    moat_rating: str = ""
    moat_sources: dict[str, str] = field(default_factory=dict)
    falsifiers: list[str] = field(default_factory=list)


@dataclass
class TrackingItem:
    judgment: str
    indicator: str
    direction: str
    source: str = ""
    frequency: str = "季度"


@dataclass
class Falsifier:
    judgment: str
    signal: str
    severity: str = "中"


@dataclass
class Scenario:
    name: str  # 基准 / 乐观 / 悲观
    probability: float
    assumption: str
    trigger: str
    impact: str


@dataclass
class ChecklistItem:
    item: str
    result: str = "⚠️"  # ✅ / ⚠️ / ❌
    note: str = ""


@dataclass
class ExecutionReport:
    """Gate 4 执行报告：框架使用情况、缺口汇总、评审汇总、成本与调用数。"""

    target: str = ""
    period: str = ""
    mode: str = "company"
    framework: str = "investment-deep-analysis v2.0 / 10章闭环"
    enabled_chapters: list[int] = field(default_factory=list)
    skipped_chapters: list[int] = field(default_factory=list)
    llm_calls: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    review_rounds: int = 0
    notes: list[str] = field(default_factory=list)
    # 模型路由（决策 ④）：章节/评审走 SMALL_MODEL，第 9 章收敛与终审走主模型。
    # 记录的是**实际解析出的模型名**，不是配置占位符，供 A/B 与成本归因核查。
    chapter_model: str = ""
    convergence_model: str = ""
    model_route: str = ""  # none / injected / small+primary / all-primary / all-small / override
    # 第 9 章收敛质量门禁：代码级检查，不依赖模型自觉遵守 system prompt。
    convergence_quality: str = ""  # solid / shallow / incomplete / ""（未执行收敛）
    convergence_checks: list[str] = field(default_factory=list)
    # 模型通道事件：空 content 自动升级主模型等，按发生顺序记录，供成本归因与排障。
    model_events: list[str] = field(default_factory=list)


# 质量判定的人话标签，Markdown 与 PDF 附录共用，避免两处文案漂移。
QUALITY_LABELS = {
    "solid": "solid · 检查全部通过",
    "shallow": "shallow · 停在断言层，需人工复核",
    "incomplete": "incomplete · 存在完整性缺口，需人工复核",
    "": "未执行收敛",
}


@dataclass
class DeepAnalysisResult:
    """一次深度分析的完整产物。可 JSON 落盘、可 A/B 比对、可被后续证伪回查。"""

    target: str
    period: str
    mode: str
    mainlines: list[Mainline] = field(default_factory=list)
    company_type: str = ""
    chapters: list[ChapterResult] = field(default_factory=list)
    convergence: Convergence | None = None
    checklist: list[ChecklistItem] = field(default_factory=list)
    tracking: list[TrackingItem] = field(default_factory=list)
    falsifiers: list[Falsifier] = field(default_factory=list)
    scenarios: list[Scenario] = field(default_factory=list)
    cross_checks: list[str] = field(default_factory=list)
    execution: ExecutionReport = field(default_factory=ExecutionReport)
    # 台账级缺口（数据源未接入、二手数据未回溯等）：不属于任何单章，
    # 但必须显式声明。零 token 骨架模式下章节全是 skipped，只有这里能承载。
    ledger_gaps: list[DataGap] = field(default_factory=list)

    @property
    def all_gaps(self) -> list[DataGap]:
        merged: list[DataGap] = []
        seen: set[str] = set()
        sources: list[DataGap] = [*self.ledger_gaps]
        for chapter in self.chapters:
            sources.extend(chapter.gaps)
        for gap in sources:
            key = f"{gap.item}|{','.join(str(n) for n in gap.chapters)}"
            if key not in seen:
                seen.add(key)
                merged.append(gap)
        return merged

    @property
    def all_sources(self) -> list[DataSourceRow]:
        rows: list[DataSourceRow] = []
        seen: set[str] = set()
        for chapter in self.chapters:
            for row in chapter.sources:
                key = f"{row.item}|{row.source_file}|{row.location}"
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
        return rows

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        """渲染成给人看的报告。事实与推断在版面上物理分隔，避免误读。"""
        lines: list[str] = [
            f"# {self.target} {self.period} 深度投资分析（{'公司' if self.mode == 'company' else '行业'}）",
            "",
            "> **本层是分析推断层，不是财报数据底稿。** 所有量化事实以底层财报底稿及",
            "> 其溯源附件为准；本层结论按【已知】/【推断】/【不确定】标注，",
            "> 推断不构成事实，不构成估值或买卖建议。",
            "",
            "## 执行报告",
            "",
            "| 项 | 值 |",
            "| --- | --- |",
            f"| 分析框架 | {self.execution.framework} |",
            f"| 启用章节 | {', '.join(str(n) for n in self.execution.enabled_chapters)} |",
            f"| 跳过章节 | {', '.join(str(n) for n in self.execution.skipped_chapters) or '无'} |",
            f"| 评审轮次 | {self.execution.review_rounds} |",
            f"| LLM 调用 | {self.execution.llm_calls} |",
            f"| 章节/评审模型 | {self.execution.chapter_model or '未启用'} |",
            f"| 第 9 章收敛模型 | {self.execution.convergence_model or '未启用'} |",
            f"| 模型路由 | {self.execution.model_route or '未启用 LLM'} |",
            f"| 收敛质量门禁 | "
            f"{QUALITY_LABELS.get(self.execution.convergence_quality, '未执行收敛')} |",
            f"| 估算 token（输入/输出） | {self.execution.estimated_input_tokens} / "
            f"{self.execution.estimated_output_tokens} |",
            "",
        ]
        if self.execution.notes:
            lines.extend(f"- {note}" for note in self.execution.notes)
            lines.append("")
        if self.execution.convergence_checks:
            lines.extend(["**第 9 章收敛质量检查（代码级，不依赖提示词自觉）**", ""])
            lines.extend(f"- {item}" for item in self.execution.convergence_checks)
            lines.append("")

        lines.extend(["## 第 1 章 候选主线（假设，待验证）", ""])
        if self.company_type:
            lines.append(f"公司类型：{self.company_type}。")
            lines.append("")
        for item in self.mainlines:
            lines.append(
                f"- **{item.business}**（{item.rule}）→ 初始链条：{item.chain}；"
                f"{item.tier}；{item.status}"
            )
        if not self.mainlines:
            lines.append("- 已抽取指标不足以生成候选主线，记为数据缺口。")
        lines.append("")

        for chapter in self.chapters:
            lines.extend([f"## 第 {chapter.number} 章 {chapter.title}", ""])
            lines.append(f"**核心问题**：{chapter.question}")
            lines.append("")
            lines.append(f"**方法**：{chapter.method}")
            lines.append("")
            lines.append(f"**章节结论**：{chapter.verdict or '未形成结论（数据不足）'}")
            lines.append("")
            if chapter.claims:
                lines.append("**结论与证据**")
                lines.append("")
                for claim in chapter.claims:
                    cite = "、".join(claim.evidence_ids) or "无直接证据"
                    lines.append(
                        f"- 【{claim.label}】{claim.text}（{claim.tier}，证据 {cite}，"
                        f"置信度 {claim.confidence:.2f}）"
                    )
                lines.append("")
            if chapter.sources:
                lines.extend(_table(
                    ["数据项", "源头文件", "页码/位置", "Tier", "获取方式"],
                    [[r.item, r.source_file, r.location, r.tier, r.acquisition]
                     for r in chapter.sources],
                ))
            if chapter.gaps:
                lines.extend(["**数据缺口**", ""])
                lines.extend(_table(
                    ["缺口数据", "涉及章节", "影响", "处置"],
                    [[g.item, ",".join(str(n) for n in g.chapters), g.impact, g.handling]
                     for g in chapter.gaps],
                ))
            if chapter.review:
                lines.extend(["**评审记录**", ""])
                lines.extend(_table(
                    ["轮次", "专家", "意见", "处理", "修订/理由"],
                    [[f"第{o.round_no}轮", o.expert, o.issue, o.disposition_label,
                      o.revision or o.reason] for o in chapter.review],
                ))
            lines.append(f"章节状态：`{chapter.status}`")
            if chapter.fallback_reason:
                lines.append(f"> 降级原因：{chapter.fallback_reason}")
            lines.append("")

        if self.convergence is not None:
            lines.extend(["## 第 9 章 核心链条收敛与护城河", ""])
            lines.append(f"**核心链条终稿**：{self.convergence.final_chain or '未收敛'}")
            lines.append("")
            lines.append(f"**与候选主线的差异**：{self.convergence.delta or '未记录'}")
            lines.append("")
            for decision in self.convergence.decisions:
                lines.append(
                    f"- {decision.get('mainline', '')} → {decision.get('decision', '')}："
                    f"{decision.get('reason', '')}"
                )
            lines.append("")
            lines.append(f"**护城河综合评级**：{self.convergence.moat_rating or '未评级'}")
            lines.append("")
            for name, rating in self.convergence.moat_sources.items():
                lines.append(f"- {name}：{rating}")
            lines.append("")

        if self.checklist:
            lines.extend(["## 第 10 章 个人 checklist", ""])
            lines.extend(_table(
                ["检查项", "结果", "说明"],
                [[c.item, c.result, c.note] for c in self.checklist],
            ))

        lines.extend(["## 跟踪与证伪", ""])
        if self.tracking:
            lines.extend(_table(
                ["关键判断", "先行指标", "验证/证伪方向", "数据来源", "频率"],
                [[t.judgment, t.indicator, t.direction, t.source, t.frequency]
                 for t in self.tracking],
            ))
        else:
            lines.append("> 判断不足以支撑跟踪指标，风险更高。")
            lines.append("")
        if self.falsifiers:
            lines.extend(_table(
                ["判断", "证伪信号", "严重程度"],
                [[f.judgment, f.signal, f.severity] for f in self.falsifiers],
            ))
        if self.scenarios:
            lines.extend(_table(
                ["情景", "概率", "关键假设", "触发条件", "对判断的影响"],
                [[s.name, f"{s.probability:.0%}", s.assumption, s.trigger, s.impact]
                 for s in self.scenarios],
            ))

        lines.extend(["## 跨章一致性审查（Gate 5）", ""])
        lines.extend(f"- {item}" for item in self.cross_checks)
        lines.append("")
        lines.extend(["## 数据缺口总表", ""])
        if self.all_gaps:
            lines.extend(_table(
                ["缺口数据", "涉及章节", "影响", "处置"],
                [[g.item, ",".join(str(n) for n in g.chapters), g.impact, g.handling]
                 for g in self.all_gaps],
            ))
        else:
            lines.append("无显式缺口（不代表数据完备，仅代表本次未被识别）。")
            lines.append("")
        lines.append("> 帮助验证，不替你做决定。证伪信号触发时应回查对应章节修订评级，")
        lines.append("> 而不是删除本报告。")
        return "\n".join(lines)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """极简 Markdown 表格；单元格里的竖线会破坏表格，统一转义。"""
    def clean(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").strip()

    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    out.extend("| " + " | ".join(clean(cell) for cell in row) + " |" for row in rows)
    out.append("")
    return out
