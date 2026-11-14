"""章节分析器与评审器：LLM 调用 + 后处理硬门禁。

沿用 report/analyze.py 已被 885 个用例验证过的三件套：
JSON mode 失败降级、``load_llm_json`` 稳健解析、代码级后处理门禁（不是只靠提示词）。
差异：本层允许推断，但推断必须带证据 + 三分标注 + 低置信度，且数字仍过白名单。
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from ai_agent.core.exceptions import LLMError
from ai_agent.core.logging import get_logger
from ai_agent.llm.client import complete_chat, lightweight_llm
from ai_agent.llm.json_utils import load_llm_json
from ai_agent.report.deep.chapters import ChapterSpec, EXPERT_VIEWS
from ai_agent.report.deep.ledger import NUMBER_RE, SourceLedger, normalize_number
from ai_agent.report.deep.prompts import (
    CHAPTER_SYSTEM,
    CONVERGENCE_SYSTEM,
    REVIEW_SYSTEM,
    render_chapter_user,
    render_convergence_user,
    render_review_user,
)
from ai_agent.report.deep.schema import (
    CLAIM_TYPES,
    TIERS,
    ChapterResult,
    Claim,
    Convergence,
    DataGap,
    ReviewOpinion,
    TrackingItem,
)

logger = get_logger(__name__)

Complete = Callable[[str, str], str]

# 事实层已有的禁令在本层继续生效：不做估值、不给买卖建议。
_FORBIDDEN = (
    "目标价", "买入评级", "卖出评级", "增持评级", "估值建议", "合理市值",
    "PE估值", "PB估值", "DCF", "市盈率", "市净率", "估值分位", "建议买入", "建议卖出",
)
_EMPTY_REASON = ("我认为没问题", "没问题", "无需修改", "ok", "无")
_SCENARIO_NAMES = frozenset({"基准", "乐观", "悲观"})


class BudgetExceeded(RuntimeError):
    """LLM 调用预算耗尽。由编排层捕获，剩余章节标 skipped 而不是悄悄缩水。"""


def default_complete(
    max_tokens: int,
    temperature: float,
    timeout: float,
    model: str,
    *,
    escalation_model: str = "",
    events: list[str] | None = None,
    circuit_break_after: int = 2,
) -> Complete:
    """构造默认补全函数：JSON mode 不被网关支持时去掉该参数重试一次。

    思考型小模型会把 ``max_tokens`` 全部消耗在 reasoning 上并返回空 content
    （本地 Ollama qwen3 实测：completion_tokens=1600 / finish_reason=length /
    content=""）。检测到空响应时，用 ``escalation_model``（通常是主模型）重试一次，
    并把事件写进 ``events`` 供 ExecutionReport 与 PDF 附录 A 审计。小模型优先的
    路由意图不变，成本只在失败时上升。

    连续 ``circuit_break_after`` 次空响应即熔断：后续调用直接走升级模型，
    避免「每次都先空跑一轮必空的小模型」把整份报告的耗时拖到十几分钟。
    ``escalation_model`` 为空表示本条通道本来就在主模型上（决策 ④-B 的
    ``chapter_channel="primary"``），此时不重试——同一模型重试只会多付一次延迟。
    """

    def once(use_model: str, system: str, user: str) -> tuple[str, str, dict]:
        name, client = lightweight_llm(use_model)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {
            "model": name,
            "client": client,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        diag: dict[str, Any] = {}
        try:
            raw = complete_chat(
                messages, response_format={"type": "json_object"},
                diagnostics=diag, **kwargs
            )
        except LLMError as exc:
            # 只有网关明确拒绝 JSON mode（400）才是「不支持」；429/5xx/超时等
            # 是临时故障，去掉 response_format 重试能过，但原因要如实记录，
            # 别让日志误导成「模型不支持 JSON」。
            if exc.status_code == 400:
                logger.warning(
                    "深度分析 JSON mode 不被网关支持（400），去掉 response_format 重试"
                )
            else:
                logger.warning(
                    "深度分析首次调用失败（status=%s）：%s；去掉 response_format 重试",
                    exc.status_code,
                    str(exc)[:200],
                )
            raw = complete_chat(messages, diagnostics=diag, **kwargs)
        return raw or "", name, diag

    # 章节并行时同一个补全函数会被多个线程共用；熔断状态与事件列表必须同步。
    state_lock = threading.RLock()

    def record(event: str) -> None:
        with state_lock:
            if events is not None:
                events.append(event)
        logger.warning("深度分析模型事件：%s", event)

    empty_streak = 0
    fused = False

    def call(system: str, user: str) -> str:
        nonlocal empty_streak, fused
        with state_lock:
            bypass = bool(fused and escalation_model)
        if bypass:
            raw, _, _ = once(escalation_model, system, user)
            return raw
        raw, name, diag = once(model, system, user)
        if raw.strip():
            with state_lock:
                empty_streak = 0
            return raw
        with state_lock:
            empty_streak += 1
            streak = empty_streak
        detail = (
            f"finish_reason={diag.get('finish_reason') or '未知'}、"
            f"completion_tokens={diag.get('completion_tokens') or '未知'}、"
            f"reasoning_chars={diag.get('reasoning_chars', 0)}"
        )
        if not escalation_model or escalation_model == name:
            record(f"[{name}] 返回空 content（{detail}）；无可升级模型，本次调用作废")
            return ""
        upgraded, upgraded_name, _ = once(escalation_model, system, user)
        outcome = (
            f"成功返回 {len(upgraded)} 字符" if upgraded.strip() else "仍为空，章节将降级"
        )
        record(
            f"[{name}] 返回空 content（{detail}，多为思考型模型 reasoning 耗尽 token）；"
            f"已自动升级 [{upgraded_name}] 重试：{outcome}"
        )
        if streak >= max(1, circuit_break_after):
            with state_lock:
                already_fused = fused
                fused = True
            # 并发下多个线程可能同时跨过阈值；熔断事件只记一次，避免
            # 「连续 2 次」「连续 3 次」重复刷屏，也避免误导成本归因。
            if not already_fused:
                record(
                    f"[{name}] 连续 {streak} 次空 content，后续调用直接走 "
                    f"[{upgraded_name}]（熔断，免每章空跑小模型）"
                )
        return upgraded

    return call


def _number_violation(key: str, allowed: set[str]) -> bool:
    """台账外数字判定：章号/小比例/年份豁免规则只在这里维护一份。"""
    if not key:
        return False
    raw = key.lstrip("-")
    if re.fullmatch(r"\d{1,2}", raw) or re.fullmatch(r"(?:19|20)\d{2}", raw):
        return False
    return key not in allowed


def numbers_allowed(text: str, allowed: set[str]) -> bool:
    """正文中出现的数字必须在事实台账里。豁免章号/小比例与年份。"""
    return not any(
        _number_violation(normalize_number(match.group(0)), allowed)
        for match in NUMBER_RE.finditer(text or "")
    )


_UNTRACEABLE = "〔未溯源〕"


def guard_numbers(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    """把台账外数字遮蔽成占位符，返回 (安全文本, 被拦数字原文列表)。

    与 ``numbers_allowed`` 共用判定规则：不是另立一套门禁，而是把「丢弃整条」
    用于 claims、「遮蔽数字保留结论骨架」用于 verdict / final_chain。
    """
    blocked: list[str] = []

    def replace(match: re.Match) -> str:
        token = match.group(0)
        if not _number_violation(normalize_number(token), allowed):
            return token
        blocked.append(token)
        return _UNTRACEABLE

    return NUMBER_RE.sub(replace, text or ""), blocked


def guard_conclusion(
    text: str, ledger: SourceLedger, chapter_no: int
) -> tuple[str, DataGap | None]:
    """章节结论 / 收敛链条统一过数字台账门禁；拦到的数字遮蔽并记缺口。"""
    guarded, blocked = guard_numbers(text, ledger.allowed_numbers)
    if not blocked:
        return text, None
    # 缺口清单本身也要进 PDF，不能把被拦下的未溯源数字原样写进去。
    return guarded, DataGap(
        item=f"第{chapter_no}章结论含台账外数字（{len(blocked)} 处）",
        chapters=[chapter_no],
        impact="推断层禁止台账外数字，已遮蔽为占位符，不得外发未溯源数值",
        handling="需补 L1-L3 原文溯源或从结论中删除该数值；不得由模型心算补齐",
    )


def _clamp_confidence(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _as_list(payload: dict, key: str) -> list:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _text(value: Any, limit: int = 400) -> str:
    return str(value or "").strip()[:limit]


def _claims_from(payload: dict, ledger: SourceLedger, limit: int, chapter_no: int) -> tuple[list[Claim], list[DataGap]]:
    claims: list[Claim] = []
    gaps: list[DataGap] = []
    for raw in _as_list(payload, "claims")[: limit * 2]:
        if not isinstance(raw, dict):
            continue
        text = _text(raw.get("text"))
        if not text or any(word in text for word in _FORBIDDEN):
            continue
        kind = _text(raw.get("claim_type"), 20)
        if kind not in CLAIM_TYPES:
            continue
        tier = _text(raw.get("tier"), 20)
        if tier not in TIERS:
            tier = "Tier 5"
        evidence_ids = [
            _text(item, 20)
            for item in (raw.get("evidence_ids") or [])
            if isinstance(item, (str, int)) and _text(item, 20) in ledger.ids
        ]
        if kind == "inferred" and not evidence_ids:
            continue  # 无证据的推断直接丢弃，不靠提示词自觉
        if not numbers_allowed(text, ledger.allowed_numbers):
            continue  # 台账外数字（含模型心算）一律丢弃
        confidence = _clamp_confidence(raw.get("confidence"), 0.8 if kind == "known" else 0.4)
        if kind == "inferred":
            confidence = min(confidence, 0.7)
        elif kind == "uncertain":
            confidence = min(confidence, 0.4)
        claims.append(
            Claim(text=text, claim_type=kind, tier=tier,
                  evidence_ids=evidence_ids, confidence=confidence)
        )
        if len(claims) >= limit:
            break
    for raw in _as_list(payload, "gaps")[:5]:
        if not isinstance(raw, dict):
            continue
        item = _text(raw.get("item"), 120)
        if item:
            gaps.append(
                DataGap(
                    item=item,
                    chapters=[chapter_no],
                    impact=_text(raw.get("impact"), 200),
                    handling=_text(raw.get("handling"), 200) or "不推断，列入待补数据源",
                )
            )
    return claims, gaps


def analyze_chapter(
    spec: ChapterSpec,
    ledger: SourceLedger,
    *,
    target: str,
    period: str,
    mode: str,
    prior: str,
    complete: Complete,
    max_claims: int = 6,
) -> ChapterResult:
    """跑一章分析。任何异常都降级为 fallback 章节，不中断整份报告。"""
    result = ChapterResult(
        number=spec.number, title=spec.title, question=spec.question,
        method="、".join(spec.methods),
    )
    user = render_chapter_user(
        TARGET=target,
        PERIOD=period,
        MODE="行业横比（对照组思维）" if mode == "industry" else "单公司",
        NUMBER=str(spec.number),
        TITLE=spec.title,
        QUESTION=spec.question,
        REQUIRED="\n".join(f"- {item}" for item in spec.required),
        METHODS="\n".join(f"- {item}" for item in spec.methods),
        DATA="；".join(spec.data_requirements),
        CONTRACT=spec.output_contract,
        SOURCES=ledger.render_for([spec.number]),
        PRIOR=prior or "（无前序章节）",
        GAPS="\n".join(f"- {gap.item}：{gap.impact}" for gap in ledger.gaps) or "（无）",
    )
    try:
        raw = complete(CHAPTER_SYSTEM, user)
    except BudgetExceeded:
        raise  # 预算闸必须冒泡到编排层，不能被降级逻辑吞掉后继续空转
    except Exception as exc:
        logger.warning("第 %s 章 LLM 失败，降级 fallback：%s", spec.number, exc)
        result.status = "fallback"
        result.verdict = "本章模型调用失败，结论从略"
        result.fallback_reason = guard_numbers(
            f"LLM 调用异常：{type(exc).__name__}: {str(exc)[:160]}",
            ledger.allowed_numbers,
        )[0]
        result.gaps.append(DataGap(item=f"第{spec.number}章分析", chapters=[spec.number],
                                   impact="本章未验证", handling="稍后重试，不得补写推断"))
        return result
    try:
        payload = load_llm_json(raw, repair=True)
        parse_error = ""
    except Exception as exc:
        payload = None
        parse_error = f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        result.status = "fallback"
        result.verdict = "本章输出无法解析，结论从略"
        if not (raw or "").strip():
            result.fallback_reason = (
                "模型返回空 content：思考型模型的 reasoning 可能耗尽 max_tokens；"
                "自动升级主模型仍为空（详见执行报告模型事件）"
            )
        else:
            excerpt = guard_numbers(
                (raw or "").strip()[:120], ledger.allowed_numbers
            )[0]
            result.fallback_reason = guard_numbers(
                f"输出未解析成 JSON 对象（得到 {type(payload).__name__}；"
                f"{parse_error or '结构不符'}）；原始 {len(raw or '')} 字符，"
                f"摘要：{excerpt!r}",
                ledger.allowed_numbers,
            )[0]
        return result

    result.verdict, number_gap = guard_conclusion(
        _text(payload.get("verdict"), 300), ledger, spec.number
    )
    result.claims, result.gaps = _claims_from(payload, ledger, max_claims, spec.number)
    # _claims_from 会整体重赋 gaps，门禁缺口必须在它之后追加，否则被覆盖。
    if number_gap:
        result.gaps.append(number_gap)
    used_ids: set[str] = set()
    for claim in result.claims:
        used_ids.update(claim.evidence_ids)
    result.sources = ledger.source_rows(sorted(used_ids)) or ledger.source_rows()[:5]
    for gap in ledger.gaps:
        if gap.item not in {item.item for item in result.gaps}:
            result.gaps.append(gap)
    result.status = "draft"
    return result


def render_chapter_for_review(chapter: ChapterResult) -> str:
    parts = [f"结论：{chapter.verdict or '（无）'}"]
    parts.extend(
        f"- 【{claim.label}】{claim.text}（{claim.tier}，证据 {'、'.join(claim.evidence_ids) or '无'}）"
        for claim in chapter.claims
    )
    parts.extend(f"- 缺口：{gap.item}" for gap in chapter.gaps)
    return "\n".join(parts)


def planned_review_rounds(rounds: int) -> list[int]:
    """评审实际会跑的轮次：``rounds`` 轮 + 第 5 轮终审（与 rounds 重合时去重）。"""
    return list(dict.fromkeys(list(range(1, max(1, int(rounds)) + 1)) + [5]))


def review_round_count(rounds: int) -> int:
    return len(planned_review_rounds(rounds))


def review_chapter(
    chapter: ChapterResult,
    ledger: SourceLedger,
    *,
    experts: tuple[str, ...],
    rounds: int,
    prior: str,
    complete: Complete,
    max_opinions: int = 4,
    round_concurrency: int = 1,
    on_round: Callable[[int, int], None] | None = None,
) -> list[ReviewOpinion]:
    """按轮次执行专家评审。默认 2 轮 + 1 次一致性与终审（合并第 4/5 轮）。

    12 专家 × 5 轮全开会到 600 次调用量级；这里把同一轮的多个视角合并进
    **一次**调用，是成本与质量的工程折中。full_panel 可切回全量视角。

    轮次之间**互不依赖**：``render_review_user`` 只吃章节正文与台账，不带上
    上一轮的意见，所以同一章的 3 次评审可以并发跑。``round_concurrency``：
    ``1`` = 逐轮串行（库级默认，行为逐字不变）；``0``/负数 = 所有轮一起发；
    其它正整数 = 线程池上限。并发时结果仍按轮次顺序合并，输出顺序与串行一致。
    """

    expert_block = "\n".join(
        f"- {name}：{EXPERT_VIEWS.get(name, '独立审查视角')}" for name in experts
    )
    goals = {1: "初审（求广）", 2: "修订审查", 3: "深度审查", 4: "一致性检查", 5: "终审确认"}
    schedule = planned_review_rounds(rounds)
    limit = max_opinions * len(schedule)

    def run_round(round_no: int) -> list[ReviewOpinion]:
        user = render_review_user(
            ROUND=str(round_no),
            ROUND_GOAL=goals.get(round_no, "终审确认"),
            NUMBER=str(chapter.number),
            TITLE=chapter.title,
            EXPERTS=expert_block,
            CONTENT=render_chapter_for_review(chapter),
            SOURCES_SUMMARY=ledger.render_for([chapter.number], limit=8),
            PRIOR=prior or "（无前序章节）",
        )
        try:
            payload = load_llm_json(complete(REVIEW_SYSTEM, user), repair=True)
        except BudgetExceeded:
            raise
        except Exception as exc:
            logger.warning("第 %s 章第 %s 轮评审失败：%s", chapter.number, round_no, exc)
            return []
        if not isinstance(payload, dict):
            return []
        rows: list[ReviewOpinion] = []
        for raw in _as_list(payload, "opinions")[: max_opinions * 2]:
            if not isinstance(raw, dict):
                continue
            expert = _text(raw.get("expert"), 30)
            issue = _text(raw.get("issue"), 300)
            if expert not in experts or not issue:
                continue
            disposition = _text(raw.get("disposition"), 20)
            if disposition not in {"adopted", "rejected"}:
                continue
            reason = _text(raw.get("reason"), 200)
            if disposition == "rejected" and (
                len(reason) < 8 or any(reason.startswith(word) for word in _EMPTY_REASON)
            ):
                continue  # 空驳回不算处理，按协议丢弃
            rows.append(
                ReviewOpinion(
                    round_no=round_no, expert=expert, issue=issue,
                    disposition=disposition, reason=reason,
                    revision=_text(raw.get("revision"), 200),
                )
            )
        return rows

    if round_concurrency != 1 and len(schedule) > 1:
        return _review_rounds_concurrently(
            run_round, schedule, limit=limit,
            workers=len(schedule) if round_concurrency <= 0 else round_concurrency,
            on_round=on_round,
        )

    opinions: list[ReviewOpinion] = []
    for index, round_no in enumerate(schedule, start=1):
        opinions.extend(run_round(round_no))
        if on_round is not None:
            on_round(index, len(schedule))
        if len(opinions) >= limit:
            # 串行时旧实现是在 append 的瞬间就返回（恰好 limit 条）；截断保持同序同量
            return opinions[:limit]
    return opinions


def _review_rounds_concurrently(
    run_round: Callable[[int], list[ReviewOpinion]],
    schedule: list[int],
    *,
    limit: int,
    workers: int,
    on_round: Callable[[int, int], None] | None,
) -> list[ReviewOpinion]:
    """轮次并发执行，按 ``schedule`` 顺序合并，保证输出与串行逐字同序。

    ``on_round`` 在调用方线程里递增回调（不并发），进度文案不会乱序。
    预算闸：任一轮抛 ``BudgetExceeded`` 时，等在途轮次收尾后原样上抛；已产出的
    意见随异常一起丢弃，与串行「超预算即中断评审」的语义一致。
    """
    results: dict[int, list[ReviewOpinion]] = {}
    finished = 0
    budget_error: BudgetExceeded | None = None
    with ThreadPoolExecutor(
        max_workers=max(1, min(workers, len(schedule))),
        thread_name_prefix="deep-review",
    ) as pool:
        futures = {pool.submit(run_round, round_no): round_no for round_no in schedule}
        for future in as_completed(futures):
            round_no = futures[future]
            try:
                results[round_no] = future.result()
            except BudgetExceeded as exc:
                budget_error = budget_error or exc
                continue
            finished += 1
            if on_round is not None:
                on_round(finished, len(schedule))
    if budget_error is not None:
        raise budget_error

    opinions: list[ReviewOpinion] = []
    for round_no in schedule:
        opinions.extend(results.get(round_no, []))
    return opinions[:limit]


def converge(
    *,
    target: str,
    period: str,
    mainlines_text: str,
    prior: str,
    ledger: SourceLedger,
    complete: Complete,
) -> tuple[Convergence | None, list[TrackingItem], list[dict], list[dict]]:
    """第 9 章收敛：候选主线裁决 + 护城河 + 证伪 + 跟踪 + 三情景。"""
    user = render_convergence_user(
        TARGET=target,
        PERIOD=period,
        MAINLINES=mainlines_text or "（无候选主线）",
        PRIOR=prior or "（前序章节无定稿结论）",
        SOURCES=ledger.render_for([9]),
        GAPS="\n".join(f"- {gap.item}：{gap.impact}" for gap in ledger.gaps) or "（无）",
    )
    try:
        payload = load_llm_json(complete(CONVERGENCE_SYSTEM, user), repair=True)
    except BudgetExceeded:
        raise
    except Exception as exc:
        logger.warning("第 9 章收敛失败：%s", exc)
        return None, [], [], []
    if not isinstance(payload, dict):
        return None, [], [], []
    decisions = [
        {
            "mainline": _text(item.get("mainline"), 120),
            "decision": _text(item.get("decision"), 20),
            "reason": _text(item.get("reason"), 200),
        }
        for item in _as_list(payload, "decisions")
        if isinstance(item, dict) and _text(item.get("mainline"))
    ]
    raw_moat = payload.get("moat_sources")
    moat_sources = (
        {_text(key, 30): _text(value, 120) for key, value in raw_moat.items()}
        if isinstance(raw_moat, dict)
        else {}
    )
    falsifier_rows: list[dict] = []
    for item in _as_list(payload, "falsifiers")[:8]:
        if isinstance(item, dict):
            signal = _text(item.get("signal"), 300)
            if not signal:
                continue
            falsifier_rows.append({
                "judgment": _text(item.get("judgment"), 200) or "核心链条",
                "signal": signal,
                "severity": _text(item.get("severity"), 20) or "中",
            })
        elif isinstance(item, str) and item.strip():
            # 兼容模型把证伪信号写成字符串数组的常见退格
            falsifier_rows.append({
                "judgment": "核心链条", "signal": item.strip()[:300], "severity": "高",
            })
    convergence = Convergence(
        final_chain=_text(payload.get("final_chain"), 600),
        decisions=decisions,
        delta=_text(payload.get("delta"), 400),
        moat_rating=normalize_moat_rating(
            _text(payload.get("moat_rating"), 40), moat_sources
        ),
        moat_sources=moat_sources,
        falsifiers=[row["signal"] for row in falsifier_rows],
    )
    tracking = _tracking_from(_as_list(payload, "tracking"))
    scenarios = []
    for raw in _as_list(payload, "scenarios")[:3]:
        if not isinstance(raw, dict):
            continue
        name = _text(raw.get("name"), 20)
        if name not in _SCENARIO_NAMES:
            continue
        scenarios.append({
            "name": name,
            "probability": _clamp_confidence(raw.get("probability"), 0.0),
            "assumption": _text(raw.get("assumption"), 200),
            "trigger": _text(raw.get("trigger"), 200),
            "impact": _text(raw.get("impact"), 200),
        })
    return convergence, tracking, scenarios, falsifier_rows


# 第 9 章收敛质量门禁：CONVERGENCE_SYSTEM 的硬性要求不能只靠模型自觉遵守。
# 命中「断言层」标记且完全没有机制/实证表述时判 shallow；其余缺项判 incomplete。
_SHALLOW_ASSERTIONS = (
    "品牌强", "技术领先", "竞争力强", "护城河深", "优势明显", "壁垒高",
    "行业龙头", "实力雄厚", "口碑好", "管理优秀", "质地优良",
)
_DEPTH_MARKERS = (
    "因为", "由于", "导致", "从而", "机制", "占比", "份额", "成本", "单价",
    "产能", "良率", "周转", "年限", "实证", "证据", "可复制", "不可再拆",
    "←", "→", "=>", "验证", "回落", "扩张",
)
_MOAT_DIMENSIONS = ("品牌", "成本", "网络", "转换", "特许")
_VALID_DECISIONS = frozenset({"确认", "修正", "推翻"})
_MOAT_MISSING = ("未找到实证", "无实证", "数据不足", "无法验证")


def normalize_moat_rating(rating: str, moat_sources: dict[str, str] | None) -> str:
    """证据不足时不允许给出「中等」等听起来有结论的分档。

    有效实证：分项文本含星级且不是「未找到实证」类空话。
    有效来源 ≤2 或缺失项 ≥3 → 强制「暂无法评级」。
    """
    sources = moat_sources or {}
    values = [str(v or "").strip() for v in sources.values()]
    missing = 0
    evidenced = 0
    for value in values:
        if not value or any(token in value for token in _MOAT_MISSING):
            missing += 1
            continue
        if "★" in value or re.search(r"[1-5]\s*分", value):
            evidenced += 1
        else:
            # 有文字但无星级：算弱证据，不计入 evidenced
            missing += 1
    raw = (rating or "").strip() or "数据不足"
    if evidenced <= 2 or missing >= 3:
        return "暂无法评级"
    if raw in {"中等", "中", "一般"} and evidenced <= 3 and missing >= 2:
        return "暂无法评级"
    return raw


def _probability(item: Any) -> float:
    """兼容 Scenario dataclass 与 dict 两种来源，便于纯函数单测。"""
    raw = item.get("probability") if isinstance(item, dict) else getattr(item, "probability", 0)
    try:
        return float(raw or 0)
    except (TypeError, ValueError):
        return 0.0


def assess_convergence(
    convergence: Convergence | None,
    mainlines: list,
    scenarios: list,
    falsifiers: list,
    tracking: list,
) -> tuple[str, list[str]]:
    """检查第 9 章收敛是否真的钻到了不可再拆终点，而非停在「品牌强」。

    **只标注、不删改结论**：不承诺准确率，判定结果写进 ExecutionReport，
    合并进 PDF 后审核人可见；后续用评测集量化 shallow 比例再决定是否调模型。
    """
    if convergence is None:
        return "", ["⚠️ 本次未产出收敛结论（零 token 骨架、预算闸触发或模型调用失败）"]

    checks: list[str] = []
    shallow = False
    incomplete = False

    chain = (convergence.final_chain or "").strip()
    if not chain:
        checks.append("❌ 核心链条终稿为空")
        incomplete = True
    elif any(word in chain for word in _SHALLOW_ASSERTIONS) and not any(
        word in chain for word in _DEPTH_MARKERS
    ):
        checks.append("❌ 核心链条停在断言层，未追问到物理/时间/量化实证/历史实证终点")
        shallow = True
    else:
        checks.append("✅ 核心链条含机制或实证表述，未命中断言层标记")

    moat_text = "".join(convergence.moat_sources or {})
    covered = [dim for dim in _MOAT_DIMENSIONS if dim in moat_text]
    if len(covered) >= 4:
        checks.append(f"✅ 护城河五大来源覆盖 {len(covered)}/5")
    else:
        checks.append(
            f"⚠️ 护城河分项仅覆盖 {len(covered)}/5；未覆盖项须显式标注「未找到实证」"
        )
        incomplete = True
    rating = (convergence.moat_rating or "").strip()
    if rating == "暂无法评级":
        checks.append("✅ 护城河证据不足，已降级为「暂无法评级」")
    elif rating in {"中等", "中", "一般"}:
        normalized = normalize_moat_rating(rating, {
            str(k): str(v) for k, v in (convergence.moat_sources or {}).items()
        })
        if normalized == "暂无法评级":
            checks.append("❌ 护城河证据不足却给出「中等」，应降为「暂无法评级」")
            incomplete = True

    if mainlines:
        valid = [
            item for item in convergence.decisions
            if isinstance(item, dict) and item.get("decision") in _VALID_DECISIONS
        ]
        if len(valid) >= len(mainlines):
            checks.append(f"✅ {len(mainlines)} 条候选主线均被确认/修正/推翻")
        else:
            checks.append(
                f"⚠️ 候选主线 {len(mainlines)} 条，有效处置仅 {len(valid)} 条，"
                "存在把假设直接当结论的风险"
            )
            incomplete = True
    else:
        checks.append("✅ 无候选主线，跳过处置覆盖检查")

    total = sum(_probability(item) for item in scenarios)
    if len(scenarios) == 3 and abs(total - 1.0) <= 0.05:
        checks.append(f"✅ 三情景概率合计 {total:.2f}（模型主观赋值，非财报披露值）")
    else:
        checks.append(
            f"⚠️ 情景 {len(scenarios)} 档、概率合计 {total:.2f}，应为 3 档且合计 1±0.05"
        )
        incomplete = True

    if falsifiers and tracking:
        checks.append(f"✅ 证伪信号 {len(falsifiers)} 条、跟踪指标 {len(tracking)} 条")
    else:
        checks.append("⚠️ 证伪信号或跟踪指标缺失，关键判断不可验证")
        incomplete = True

    quality = "shallow" if shallow else ("incomplete" if incomplete else "solid")
    return quality, checks


def _tracking_from(raw_items: list) -> list[TrackingItem]:
    out: list[TrackingItem] = []
    for raw in raw_items[:8]:
        if not isinstance(raw, dict):
            continue
        judgment = _text(raw.get("judgment"), 200)
        indicator = _text(raw.get("indicator"), 200)
        if not judgment or not indicator:
            continue
        out.append(
            TrackingItem(
                judgment=judgment,
                indicator=indicator,
                direction=_text(raw.get("direction"), 200),
                source=_text(raw.get("source"), 120),
                frequency=_text(raw.get("frequency"), 20) or "季度",
            )
        )
    return out
