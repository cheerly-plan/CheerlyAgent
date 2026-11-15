"""业绩分析 skill：把「问某公司业绩」从模型自由决策变成脚本化预取。

背景：实测同一个问题「中科飞测的业绩情况」，模型调 7–13 次工具，次数
每次不同——抽取分几次、重复读原文、同指标反复算。把调用顺序写进提示词
能把概率压下来，但保证不了。

这里走另一条路：命中时由代码按固定顺序预取数据（抽取 → 对比），把结果
作为一段上下文塞给模型，模型基于已有数据作答。脚本只负责「取数」，
取完模型仍可自由调用其它工具补充——脚本不该成为单点故障。

与 skills 目录的区别：``report/skills`` 是**产物模板**（研报长什么样、
该有哪些指标、公司名单），这里编的是**行为顺序**（先调谁再调谁）。
"""
from __future__ import annotations

from dataclasses import dataclass

from ai_agent.core.logging import get_logger

logger = get_logger(__name__)

# 命中判定用的指标词（与 followup._METRIC_WORDS 同源语义：问具体数字）
_PERFORMANCE_WORDS = (
    "业绩", "营收", "营业收入", "收入", "利润", "净利", "归母", "毛利", "毛利率",
    "净利率", "净息差", "同比", "环比", "增长", "下滑", "财务",
)
# 明确要研报/导出时不该进这个 skill
_EXCLUDE_WORDS = ("研报", "报告", "导出", "PDF", "下载")

# 预取默认取的指标（抽取工具留空会按行业模板给，这里显式列常用几项，
# 保证输出规模可控；模型需要更多可以自己再调工具）
_DEFAULT_INDICATORS = [
    "营业收入",
    "归属于上市公司股东的净利润",
    "扣除非经常性损益的净利润",
    "基本每股收益",
    "研发投入合计",
]
_MARKER = "【业绩数据（系统预取）】"


@dataclass
class Prefetch:
    """预取结果：一段可直接塞进上下文的文本，加实际调用过的工具名。"""

    text: str
    tools: tuple[str, ...]


def wants_performance(text: str) -> bool:
    """问句是否在问业绩数字。只看字面，不做语义判断。"""
    body = (text or "").strip()
    if not body:
        return False
    if any(word in body for word in _EXCLUDE_WORDS):
        return False
    return any(word in body for word in _PERFORMANCE_WORDS)


def prefetch(
    user_text: str,
    *,
    file_ids: list[str] | None = None,
) -> Prefetch | None:
    """按固定顺序预取业绩数据；不命中或全部失败时返回 None（调用方降级）。

    只做两件事：抽本期指标（含同比所需的对比期）、算增减。校验留给模型
    写完正文之后自己调 ``validate_figures`` —— 那时才有正文可校验。
    """
    from ai_agent.tools.report.report_extract import _extract_by_basis
    from ai_agent.report.industry import company_type_of, default_indicators

    try:
        from ai_agent.kb.retrieval import mentioned_subjects

        subjects = sorted(mentioned_subjects(user_text))
    except Exception:
        logger.warning("业绩预取：主体识别失败，降级给模型", exc_info=True)
        return None
    if not subjects:
        return None

    # 报告期从问句取；问句没写就去库内文件清单里取最新一期。
    # extract_indicators 不认空报告期（直接返回「需要公司、报告期和指标」），
    # 所以这一步不能省。
    period = _period_of(user_text) or _latest_period(subjects)
    if not period:
        logger.info("业绩预取：问句与库内都没有报告期，降级给模型")
        return None
    # 行业模板给的是全量指标，取前面几项控制输出规模
    indicators = list(default_indicators(company_type_of(subjects)))[
        : len(_DEFAULT_INDICATORS)
    ] or list(_DEFAULT_INDICATORS)

    tools: list[str] = []
    try:
        result = _extract_by_basis(list(subjects), period, indicators, "both")
        tools.append("report_extract_indicators")
    except Exception:
        logger.warning("业绩预取：指标抽取失败，降级给模型", exc_info=True)
        return None
    if not result.facts:
        # 抽不到数据比不预取更糟（模型会以为「库里没有」），直接降级
        logger.info("业绩预取：没有抽到任何指标，降级给模型")
        return None

    lines = [
        _MARKER,
        f"系统已按固定顺序预取以下数据（公司：{'、'.join(subjects)}，报告期：{period or '最新一期'}）。",
        "直接用这些数字作答即可，**不要为了复核再读原文**；需要解释变动原因时才检索原文。",
        "回答里带数字或表格时，写完先调 validate_figures 校验一次，表格出处写在表下即可。",
        "",
    ]
    lines.extend(_render_facts(result))
    logger.info(
        "业绩预取完成 subjects=%s facts=%d", "、".join(subjects), len(result.facts)
    )
    return Prefetch(text="\n".join(lines), tools=tuple(tools))


def _latest_period(subjects: list[str]) -> str:
    """库内这些公司的最新报告期（如 2026Q3）。"""
    from ai_agent.storage.store import store

    try:
        rows = store.list_kb_files()
    except Exception:
        logger.warning("业绩预取：读取文件清单失败", exc_info=True)
        return ""
    wanted = [s for s in subjects if s]
    best = ""
    for row in rows:
        subject = str(row.get("subject") or "")
        period = str(row.get("period") or "")
        if not period:
            continue
        if wanted and not any(w in subject or subject in w for w in wanted):
            continue
        best = max(best, period, key=_period_rank)
    return best


def _period_rank(period: str) -> tuple[int, int]:
    """报告期排序键：年份优先，同一年里年报 > 三季 > 半年 > 一季。"""
    import re

    match = re.match(r"((?:19|20)\d{2})\s*(Q[1-4]|H1)?", period or "")
    if not match:
        return (0, 0)
    year = int(match.group(1))
    suffix = {None: 5, "": 5, "Q4": 4.5, "Q3": 3, "H1": 2, "Q2": 2, "Q1": 1}.get(
        match.group(2), 0
    )
    return (year, int(suffix * 10))


def _period_of(text: str) -> str:
    """从问句里取报告期（2026Q3 / 2026H1 / 2026）；取不到返回空串。"""
    import re

    match = re.search(r"((?:19|20)\d{2})\s*年?\s*(Q[1-4]|H1|上半年|三季|半年度|全年)?", text or "")
    if not match:
        return ""
    year, part = match.group(1), (match.group(2) or "").strip()
    mapping = {"上半年": "H1", "三季": "Q3", "半年度": "H1", "全年": ""}
    return f"{year}{mapping.get(part, part)}"


def _render_facts(result) -> list[str]:
    lines: list[str] = []
    for fact in result.facts:
        period = fact.period or ""
        # 期末余额类（总资产/存货/合同负债）没有单季概念，不标口径
        if fact.basis == "point" or "期末" in (fact.column or ""):
            basis = "期末"
        elif fact.basis == "single" or not fact.cumulative:
            basis = "单季"
        else:
            basis = "累计"
        where = ""
        if fact.evidence.page is not None:
            where = f"（{fact.evidence.filename or ''} 第{fact.evidence.page}页）"
        lines.append(
            f"- {fact.subject} {period}{basis} {fact.name} = {fact.value}{fact.unit or ''}{where}"
        )
    return lines


def inject(messages: list, prefetch_text: str) -> None:
    """把预取结果作为一条 system 消息插到末尾（紧跟在本轮用户消息之后）。"""
    messages.append(
        {"role": "system", "content": prefetch_text, "performance_prefetch": True}
    )


def already_injected(messages: list) -> bool:
    """续跑时上下文里已经有预取结果，不重复预取。

    以消息上的 ``performance_prefetch`` 标记为准，不靠文本前缀——前缀一改
    幂等就悄悄失效，重复预取会让模型同时看到两份数据。
    """
    return any(msg.get("performance_prefetch") for msg in messages)


def prefetch_once(
    messages: list,
    *,
    enabled: bool,
    already_retrieved: bool = False,
    user_text: str = "",
) -> Prefetch | None:
    """给一轮对话做一次业绩预取；不满足条件时什么都不做，返回 None。

    幂等：上下文里已有预取结果（续跑）就跳过。降级：不命中、主体识别不到、
    抽不到数据都返回 None，调用方继续按模型自主决策走。
    """
    if not enabled or already_retrieved or already_injected(messages):
        return None
    text = (user_text or _latest_user_text(messages)).strip()
    if not wants_performance(text):
        return None
    result = prefetch(text)
    if result is None:
        return None
    inject(messages, result.text)
    return result


def _latest_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""
