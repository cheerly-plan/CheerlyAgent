"""财报深度分析层：事实层之上的可插拔「假设验证」层。

定位：在 ``report/`` 确定性事实层之上，补上 investment-deep-analysis v2.0 的
「10 章假设验证闭环 + 数据溯源分级 + 专家评审 + 跟踪证伪」，产出**结构化结论对象**。

边界（刻意守住）：

* 事实层的 Markdown 正文仍是数字事实的唯一权威来源，本层结论按
  已知/推断/不确定三分，推断不得伪装成事实；
* 深度产物单独落 ``.deep.json`` / ``.deep.md`` / ``.deep.pdf``，**不进
  Markdown 正文、不进 ``ReportArtifacts``**；
* PDF 默认 ``pdf_mode="separate"`` 生成**两份 PDF**：事实层底稿沿用原门禁、
  可自动定稿外发；推断层独立成文件，由 ``<report_id>-deep`` 这条独立审核
  记录强制人工批准。``"merge"`` 是兼容档，把带警示标注的推断层附录追加进
  同一份 PDF 并让整份报告进入人工审核。两种模式都不改 Markdown 正文、
  溯源附件与数字校验门禁；
* ``run_report(deep=None)`` 时本包完全不被 import，零开销、零行为变化；
* 不做估值、不给买卖建议，只输出投资逻辑与需跟踪验证的关键经营变量。

三档用法：

1. **零 token 骨架**（先验收数据覆盖度，推荐第一步）::

       from ai_agent.report.pipeline import run_report
       from ai_agent.report.deep import DeepOptions

       result = run_report("生成兆易创新 2026Q3 财报深度分析",
                           deep=DeepOptions(enabled=False))
       print(result.deep_analysis.all_gaps)     # 缺哪几类源文件，具体到章节

2. **完整 10 章**（约 31 次调用 / 1.5-4 分钟）::

       result = run_report(..., deep=DeepOptions(enabled=True))
       result.deep_analysis.to_json()           # 结构化结论
       result.artifacts.pdf                     # 事实层底稿 PDF（可自动定稿）
       result.deep_artifacts["pdf"]             # 独立深度 PDF（单独人工审）
       result.deep_pdf_review_id                # "<report_id>-deep"

   回退档：``pdf_mode="merge"`` 合成一份含附录的 PDF（整份强制人工审）；
   ``write_deep_pdf=False`` 则只落 ``.deep.json`` / ``.deep.md``，不渲染深度 PDF。
   事实层未生成 PDF 时，两种模式都只落结构化产物。

3. **手动调用**（不经过主流水线，便于 A/B 与 bad case 回归）::

       from ai_agent.report.deep import DeepConfig, run_deep_analysis

       deep = run_deep_analysis("兆易创新", "2026Q3", facts,
                                config=DeepConfig(enabled=True))

外部数据源（Phase 3）：``sources.collect_sources`` 盘点知识库近 3 年源文件、
读取本地结构化事实清单、并为行业统计留注入点；L4/L5 二手数据必须回溯到
L1-L3 才准进 ``SourceLedger``。本层不解析 PDF 附注、不内置任何第三方 API。

**模型策略（决策 ④ / ④-B）**：库级默认章节分析与专家评审走 ``SMALL_MODEL`` +
本地网关，第 9 章收敛**及其终审**走 ``MODEL``（主模型 + 主网关）；
**对外入口（HTTP / 聊天工具）默认 ``chapter_channel="primary"``**——章节与评审
直接走主模型，不再先跑小模型（决策 ④-B：本地思考型小模型空响应实测 100%，
先跑一轮只会白付延迟）。实际模型名与路由写进 ``ExecutionReport``，在推断层
PDF 的 A 节可见，供 A/B 与成本归因核查。
想回小模型优先：``DeepConfig(chapter_channel="small")`` 或环境变量
``REPORT_DEEP_CHAPTER_CHANNEL=small``（对外入口生效）；
``SMALL_MODEL`` 未配置时会显式告警，不会静默变成全流程主模型。
想做单章 A/B：给 ``run_deep_analysis`` 注入 ``convergence_complete``。

第 9 章另有代码级收敛质量门禁（断言层 / 护城河分项覆盖 / 主线处置 /
三情景概率 / 证伪与跟踪），判定为 solid / shallow / incomplete，只标注不删改结论，
在 Markdown 执行报告、checklist 与 PDF 附录 D 中均可见。

注意：LLM 调用走 ``llm/client.complete_chat``（OpenAI 兼容层）；升级 openai SDK
或换网关后请回归 ``tests/test_report_deep_*.py``，确认 JSON mode 降级路径仍可用。
"""

from ai_agent.report.deep.bridge import DeepOptions, attach_deep
from ai_agent.report.deep.chapters import CHAPTERS, chapters_for
from ai_agent.report.deep.ledger import SourceLedger
from ai_agent.report.deep.pdf_appendix import (
    APPENDIX_REVIEW_REASON,
    deep_standalone_document,
    render_deep_appendix_html,
)
from ai_agent.report.deep.analyzer import assess_convergence, normalize_moat_rating
from ai_agent.report.deep.pipeline import (
    ENTRY_CHAPTER_CHANNEL,
    ENTRY_CHAPTER_CONCURRENCY,
    ENTRY_REVIEW_ROUND_CONCURRENCY,
    DeepConfig,
    ModelPlan,
    pick_industry_sample,
    resolve_model_plan,
    run_deep_analysis,
)
from ai_agent.report.deep.schema import (
    QUALITY_LABELS,
    ChapterResult,
    DeepAnalysisResult,
)
from ai_agent.report.deep.sources import (
    ExternalFact,
    FactsLintReport,
    SourceBundle,
    SourceProvider,
    collect_sources,
    default_facts_dir,
    inspect_fact_row,
    lint_facts_dir,
)

__all__ = [
    "APPENDIX_REVIEW_REASON",
    "CHAPTERS",
    "ENTRY_CHAPTER_CHANNEL",
    "ENTRY_CHAPTER_CONCURRENCY",
    "ENTRY_REVIEW_ROUND_CONCURRENCY",
    "ChapterResult",
    "DeepAnalysisResult",
    "DeepConfig",
    "DeepOptions",
    "ExternalFact",
    "FactsLintReport",
    "ModelPlan",
    "QUALITY_LABELS",
    "SourceBundle",
    "SourceLedger",
    "SourceProvider",
    "assess_convergence",
    "attach_deep",
    "chapters_for",
    "collect_sources",
    "deep_standalone_document",
    "default_facts_dir",
    "inspect_fact_row",
    "lint_facts_dir",
    "normalize_moat_rating",
    "pick_industry_sample",
    "render_deep_appendix_html",
    "resolve_model_plan",
    "run_deep_analysis",
]
