"""研报进度文案与阶段百分比：SSE ``tool_progress`` 与 ``/v1/report/jobs`` 同源。

边界：

* **流水线**只通过本模块的 ``msg_*`` 构造进度句，再 ``emit_progress``；
* **任务表**只通过 ``stage_percent`` 把进度句映成百分比，不猜散落文案；
* 匹配针（``STAGE_NEEDLES``）必须与 ``msg_*`` 产出的子串一致；更具体的针排在前面，
  避免「合并深度分析」被笼统的「深度分析」抢先命中。
"""

from __future__ import annotations

# 运行中进度上限；任务成功时由 jobs 置 100。
MAX_RUNNING_PROGRESS = 96

# (子串, 百分比)。顺序即优先级。
STAGE_NEEDLES: tuple[tuple[str, int], ...] = (
    ("理解研报需求", 5),
    ("准备财报原文", 12),
    ("核对／补齐财报原文", 12),
    ("核对／补齐近", 18),
    ("抽取财务指标", 30),
    ("计算同比", 45),
    ("校验数字", 58),
    ("渲染 PDF", 70),
    ("深度分析旁路", 71),
    ("从年报抽取主指标", 72),
    ("深度事实清单已写入", 74),
    ("开始撰写深度章节", 74),
    ("正在撰写深度分析第 2–8 章", 76),
    ("正在撰写深度分析第", 80),
    ("深度分析章节进度", 86),
    ("正在收敛深度分析第 9 章", 92),
    ("合并深度", 93),
    ("独立深度分析 PDF", 94),
    ("深度分析", 86),
    ("落盘产物", 95),
    ("研报已生成", 96),
)

DEEP_ETA_HINT = "深度层约 10–15 分钟，主模型多章评审，进度会持续更新"


def stage_percent(message: str) -> int:
    """把进度文案映成 0–``MAX_RUNNING_PROGRESS``；未命中返回 0。"""
    text = (message or "").strip()
    if not text:
        return 0
    return next((percent for key, percent in STAGE_NEEDLES if key in text), 0)


def advance_progress(current: int, message: str) -> int:
    """进度只增不减，并夹在运行中上限内。"""
    return max(int(current or 0), min(MAX_RUNNING_PROGRESS, stage_percent(message)))


# ——— 文案构造：与 STAGE_NEEDLES 子串对齐 ———


def msg_entry_with_deep() -> str:
    return f"正在生成底稿与深度解析（{DEEP_ETA_HINT}）…"


def msg_understanding() -> str:
    return "正在理解研报需求…"


def msg_ensure_filings(*, subject_count: int = 1) -> str:
    if subject_count > 1:
        return f"正在核对／补齐财报原文（{subject_count} 家公司）…"
    return "正在核对／补齐财报原文…"


def msg_extract_indicators() -> str:
    return "正在抽取财务指标…"


def msg_compare_and_compose() -> str:
    return "正在计算同比／环比与撰写正文…"


def msg_validate() -> str:
    return "正在校验数字与出处…"


def msg_render_pdf() -> str:
    return "正在渲染 PDF…"


def msg_deep_bypass() -> str:
    return "正在生成深度分析旁路产物…"


def msg_merge_deep() -> str:
    return "正在合并深度分析附录…"


def msg_separate_deep_pdf() -> str:
    return "正在生成独立深度分析 PDF…"


def msg_write_artifacts() -> str:
    return "正在落盘产物…"


def msg_report_done(*, gaps: int = 0) -> str:
    if gaps:
        return f"研报已生成（另有 {gaps} 处数据缺口待说明）"
    return "研报已生成"


def msg_ensure_deep_annuals(*, years: int) -> str:
    return f"正在核对／补齐近 {years} 年年报（深度台账）…"


def msg_extract_deep_facts() -> str:
    return "正在从年报抽取主指标写入深度事实清单…"


def msg_deep_facts_ready(*, fact_count: int) -> str:
    return f"深度事实清单已写入（{fact_count} 条），开始撰写深度章节…"


def msg_deep_chapters_batch(*, workers: int, parallel: bool) -> str:
    suffix = f"（{workers} 路并行）…" if parallel else "…"
    return f"正在撰写深度分析第 2–8 章{suffix}"


def msg_deep_chapter_serial(*, number: int, title: str, index: int, total: int) -> str:
    return f"正在撰写深度分析第{number}章 {title}（{index}/{total}）…"


def msg_deep_chapter_progress(*, done: int, total: int, number: int, title: str) -> str:
    return f"深度分析章节进度 {done}/{total}（第{number}章 {title}）…"


def msg_deep_converge_ch9() -> str:
    return "正在收敛深度分析第 9 章（主线裁决／证伪／情景）…"
