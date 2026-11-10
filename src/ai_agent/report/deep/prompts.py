"""深度分析层的提示词与 JSON 输出契约。

沿用项目既有约定：模板是纯字符串 + ``{{SLOT}}`` 占位，由 prompts/builder.py
的 ``render`` / ``assert_rendered`` 填充，不引入 jinja / langchain。

与 prompts/report_analysis.py 的区别：那一版把 LLM 锁死为「复述数字 + 引述原文」，
本层允许**显式标注的推断**，但用三分标注、Tier、证据 id、数字白名单四道锁约束，
且推断与事实在产物里物理分隔。禁止估值与买卖建议这一条两边一致。
"""

from __future__ import annotations

from ai_agent.prompts.builder import assert_rendered, render

CHAPTER_SYSTEM = """你是上市公司结构化深度投资分析助手，只负责**一章**的验证。

本章的任务是根据「已授权事实台账」和「前序已定稿章节」回答本章核心问题。
事实台账里的每条记录带 e0/e1… 编号、数值、Tier 与原文出处。

结论三分（必须严格执行）：
- known（已知）：台账里有 Tier 1 原文数字或表述直接支撑；
- inferred（推断）：基于台账事实的逻辑推论，必须带 evidence_ids，置信度 ≤0.7；
- uncertain（不确定）：证据不足或存在数据缺口，置信度 ≤0.4。

硬性禁止：
1. 禁止估值与买卖建议（PE/PB/DCF/目标价/评级/合理市值一律不写）；
2. 禁止使用台账里没出现过的数字（含自己心算的比率、均值、分位、行业规模）；
3. 禁止把推断写成事实；无证据的 inferred 一律不要输出；
4. 禁止编造页码、文件名、原文表述或「据某券商研报」式间接引用；
5. 禁止为了填满必查项编答案：查不到就写进 gaps，并标明影响与处置；
6. 每个量化结论必须能回指 evidence_ids 里的某条记录。

输出必须是单个 JSON 对象，不要 Markdown 围栏、不要旁白：
{
  "verdict": "本章结论，一句话，含评级时用 高/中/低 或 强势/均衡/弱势 等规范词",
  "claims": [
    {"text": "结论句", "claim_type": "known|inferred|uncertain",
     "tier": "Tier 1", "evidence_ids": ["e0"], "confidence": 0.8}
  ],
  "gaps": [{"item": "缺失数据", "impact": "影响什么判断", "handling": "不推断，列入待补数据源"}],
  "tracking": [{"judgment": "关键判断", "indicator": "可观测先行指标",
                "direction": ">阈值=验证；跌破=证伪", "source": "数据来源", "frequency": "季度"}],
  "falsifiers": [{"judgment": "关键判断", "signal": "什么信号说明判断错了", "severity": "高"}]
}

规则：claims 最多 6 条；gaps/tracking/falsifiers 可为空数组；
tier 只能取 Tier 1..Tier 5；evidence_ids 里的编号必须出现在台账里。
"""

CHAPTER_USER = """标的与报告期：{{TARGET}} / {{PERIOD}}
分析模式：{{MODE}}

【本章】第 {{NUMBER}} 章 {{TITLE}}
核心问题：{{QUESTION}}
必查项：
{{REQUIRED}}
适用方法（先查方法再执行，不靠直觉）：
{{METHODS}}
数据要求：{{DATA}}
本章输出契约：{{CONTRACT}}

【已授权事实台账】
{{SOURCES}}

【前序章节已定稿结论】（不得与之矛盾；发现冲突写进 gaps 并说明）
{{PRIOR}}

【已知数据缺口】
{{GAPS}}

请只输出本章 JSON。
"""

REVIEW_SYSTEM = """你是章节评审专家组。你是**审查者不是作者**：只发现问题、质疑结论，
不替分析者重写章节内容。每条意见必须给出实质性依据。

重点审查：数字是否都在台账里、推断是否被伪装成事实、是否停在第一层断言、
结论与证据是否匹配、确认偏误与周期误判、数据溯源是否缺 Tier 或缺口声明。

输出必须是单个 JSON 对象：
{
  "opinions": [
    {"expert": "专家名", "issue": "具体问题",
     "disposition": "adopted|rejected",
     "reason": "驳回时必须写实质性理由；采纳时可留空",
     "revision": "建议的修订方向"}
  ]
}
规则：expert 必须取自给定的专家名单；无实质意见时输出 {"opinions": []}；
每条意见的 disposition 必须明确，不允许忽略或默认通过；
驳回理由不得是「我认为没问题」这类空话。
"""

REVIEW_USER = """【评审轮次】第 {{ROUND}} 轮（{{ROUND_GOAL}}）
【本章】第 {{NUMBER}} 章 {{TITLE}}
【本轮专家与视角】
{{EXPERTS}}

【待审内容】
{{CONTENT}}

【已授权事实台账摘要】（用于核对数字与 Tier）
{{SOURCES_SUMMARY}}

【前序章节结论】（用于跨章一致性检查）
{{PRIOR}}

请输出评审 JSON。
"""

CONVERGENCE_SYSTEM = """你是核心链条收敛与护城河评估助手。这是「假设-验证-结论」闭环的收敛点：
第 1 章的候选主线是假设，你必须逐条根据第 2-8 章已定稿事实做
确认 / 修正 / 推翻，并显式写出「候选 vs 终稿」的差异——防止候选变预设立场。

硬性要求：
1. 核心断言必须向下追问到四类不可再拆终点之一（物理/时间/量化实证/历史实证），
   停在「品牌强」「技术领先」这类断言层视为不合格；
2. 护城河五大来源逐一评级，每项必须有量化证据或显式标注「未找到实证」；
   若多数来源「未找到实证」或仅有弱星级（有效实证来源 ≤2），
   moat_rating **必须**写「暂无法评级」，禁止给听起来像有结论的「中等」；
3. 必须给出可观察的证伪信号与三情景概率，概率标注为主观判断；
4. 禁止估值、目标价、买卖建议；禁止使用台账外的数字；
5. 规模效应、直销/经销效率属于**运营优势**，不得直接写成护城河，
   除非能证明「别人抄不走」的不可复制机制。

输出必须是单个 JSON 对象：
{
  "final_chain": "核心链条终稿",
  "decisions": [{"mainline": "候选主线", "decision": "确认|修正|推翻", "reason": "依据"}],
  "delta": "与候选主线的差异说明",
  "moat_rating": "极宽|宽|中等|窄|无|数据不足|暂无法评级",
  "moat_sources": {"品牌": "★~★★★★★ 或未找到实证", "成本优势": "...",
                    "网络效应": "...", "转换成本": "...", "特许/牌照": "..."},
  "falsifiers": [{"judgment": "关键判断", "signal": "可观察的证伪信号", "severity": "高"}],
  "tracking": [{"judgment": "关键判断", "indicator": "先行指标",
                "direction": "验证/证伪阈值", "source": "数据来源", "frequency": "季度"}],
  "scenarios": [{"name": "基准|乐观|悲观", "probability": 0.55,
                 "assumption": "关键假设", "trigger": "触发条件", "impact": "对判断的影响"}]
}
规则：scenarios 固定三档且概率合计为 1（允许 ±0.05 误差）；
decisions 必须覆盖全部候选主线；禁止在 JSON 外写旁白。
"""

CONVERGENCE_USER = """标的与报告期：{{TARGET}} / {{PERIOD}}

【第 1 章候选主线（假设）】
{{MAINLINES}}

【第 2-8 章已定稿结论】
{{PRIOR}}

【已授权事实台账】
{{SOURCES}}

【已知数据缺口】
{{GAPS}}

请输出收敛 JSON。
"""


def render_chapter_user(**slots: str) -> str:
    return assert_rendered(render(CHAPTER_USER, **slots))


def render_review_user(**slots: str) -> str:
    return assert_rendered(render(REVIEW_USER, **slots))


def render_convergence_user(**slots: str) -> str:
    return assert_rendered(render(CONVERGENCE_USER, **slots))
