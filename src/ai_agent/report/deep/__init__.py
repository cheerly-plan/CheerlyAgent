"""财报深度分析层：事实层之上的可插拔「假设验证」层。

定位：在 ``report/`` 确定性事实层之上，补上 investment-deep-analysis v2.0 的
「10 章假设验证闭环 + 数据溯源分级 + 专家评审 + 跟踪证伪」，产出**结构化结论对象**。

边界（刻意守住）：

* 事实层的 Markdown 正文仍是数字事实的唯一权威来源，本层结论按
  已知/推断/不确定三分，推断不得伪装成事实；
* 深度产物单独落 ``.deep.json`` / ``.deep.md``，**不进 Markdown 正文、不进
  ``ReportArtifacts``**（PDF 附录是唯一例外，见下一条）；
* PDF 默认 ``pdf_mode="merge"`` 会在事实层末尾追加**带警示标注的推断层附录**，
  并强制进入人工审核队列（禁止自动定稿）；``"separate"`` 则完全不碰 PDF。
  两种模式都不改 Markdown 正文、溯源附件与数字校验门禁；
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
       result.deep_artifacts                    # 落盘文件名
       result.deep_pdf_merged                   # 附录是否已并入同一份 PDF

   想让 PDF 保持纯事实层：``DeepOptions(enabled=True, pdf_mode="separate")``。
   合并档会多一次 PDF 渲染（约数秒），且必须经人工审核才能下载 PDF。

3. **手动调用**（不经过主流水线，便于 A/B 与 bad case 回归）::

       from ai_agent.report.deep import DeepConfig, run_deep_analysis

       deep = run_deep_analysis("兆易创新", "2026Q3", facts,
                                config=DeepConfig(enabled=True))

外部数据源（Phase 3）：``sources.collect_sources`` 盘点知识库近 5 年源文件、
读取本地结构化事实清单、并为行业统计留注入点；L4/L5 二手数据必须回溯到
L1-L3 才准进 ``SourceLedger``。本层不解析 PDF 附注、不内置任何第三方 API。

**模型策略（决策 ④）**：章节分析与专家评审走 ``SMALL_MODEL`` + 本地网关，
第 9 章收敛**及其终审**走 ``MODEL``（主模型 + 主网关）。实际模型名与路由
写进 ``ExecutionReport``，合并进 PDF 后可在附录 A 看到，供 A/B 与成本归因核查；
``SMALL_MODEL`` 未配置时会显式告警，不会静默变成全流程主模型。
想退回全流程小模型：``DeepConfig(convergence_model="<小模型名>")``；
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
    render_deep_appendix_html,
)
from ai_agent.report.deep.analyzer import assess_convergence
from ai_agent.report.deep.pipeline import (
    DeepConfig,
    ModelPlan,
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
    SourceBundle,
    SourceProvider,
    collect_sources,
)

__all__ = [
    "APPENDIX_REVIEW_REASON",
    "CHAPTERS",
    "ChapterResult",
    "DeepAnalysisResult",
    "DeepConfig",
    "DeepOptions",
    "ExternalFact",
    "ModelPlan",
    "QUALITY_LABELS",
    "SourceBundle",
    "SourceLedger",
    "SourceProvider",
    "assess_convergence",
    "attach_deep",
    "chapters_for",
    "collect_sources",
    "render_deep_appendix_html",
    "resolve_model_plan",
    "run_deep_analysis",
]
