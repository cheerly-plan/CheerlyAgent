"""章节分析器与评审器：LLM 调用 + 后处理硬门禁。

沿用 report/analyze.py 已被 885 个用例验证过的三件套：
JSON mode 失败降级、``load_llm_json`` 稳健解析、代码级后处理门禁（不是只靠提示词）。
差异：本层允许推断，但推断必须带证据 + 三分标注 + 低置信度，且数字仍过白名单。
"""

from __future__ import annotations

import re
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


def default_complete(max_tokens: int, temperature: float, timeout: float, model: str) -> Complete:
    """构造默认补全函数：JSON mode 不被网关支持时去掉该参数重试一次。"""
    def call(system: str, user: str) -> str:
        name, client = lightweight_llm(model)
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
        try:
            return complete_chat(
                messages, response_format={"type": "json_object"}, **kwargs
            )
        except LLMError:
            logger.warning("深度分析 JSON mode 不被支持，去掉 response_format 重试")
            return complete_chat(messages, **kwargs)
    return call


def numbers_allowed(text: str, allowed: set[str]) -> bool:
    """正文中出现的数字必须在事实台账里。豁免章号/小比例与年份。"""
    for match in NUMBER_RE.finditer(text or ""):
        key = normalize_number(match.group(0))
        if not key:
            continue
        raw = key.lstrip("-")
        if re.fullmatch(r"\d{1,2}", raw):
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", raw):
            continue
        if key not in allowed:
            return False
    return True


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
        payload = load_llm_json(complete(CHAPTER_SYSTEM, user), repair=True)
    except BudgetExceeded:
        raise  # 预算闸必须冒泡到编排层，不能被降级逻辑吞掉后继续空转
    except Exception as exc:
        logger.warning("第 %s 章 LLM 失败，降级 fallback：%s", spec.number, exc)
        result.status = "fallback"
        result.verdict = "本章模型调用失败，结论从略"
        result.gaps.append(DataGap(item=f"第{spec.number}章分析", chapters=[spec.number],
                                   impact="本章未验证", handling="稍后重试，不得补写推断"))
        return result
    if not isinstance(payload, dict):
        result.status = "fallback"
        result.verdict = "本章输出无法解析，结论从略"
        return result

    result.verdict = _text(payload.get("verdict"), 300)
    result.claims, result.gaps = _claims_from(payload, ledger, max_claims, spec.number)
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


def review_chapter(
    chapter: ChapterResult,
    ledger: SourceLedger,
    *,
    experts: tuple[str, ...],
    rounds: int,
    prior: str,
    complete: Complete,
    max_opinions: int = 4,
) -> list[ReviewOpinion]:
    """按轮次执行专家评审。默认 2 轮 + 1 次一致性与终审（合并第 4/5 轮）。

    12 专家 × 5 轮全开会到 600 次调用量级；这里把同一轮的多个视角合并进
    **一次**调用，是成本与质量的工程折中。full_panel 可切回全量视角。
    """
    expert_block = "\n".join(
        f"- {name}：{EXPERT_VIEWS.get(name, '独立审查视角')}" for name in experts
    )
    goals = {1: "初审（求广）", 2: "修订审查", 3: "深度审查", 4: "一致性检查", 5: "终审确认"}
    opinions: list[ReviewOpinion] = []
    schedule = list(range(1, max(1, rounds) + 1)) + [5]
    seen_rounds: set[int] = set()
    for round_no in schedule:
        if round_no in seen_rounds:
            continue
        seen_rounds.add(round_no)
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
            continue
        if not isinstance(payload, dict):
            continue
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
            opinions.append(
                ReviewOpinion(
                    round_no=round_no, expert=expert, issue=issue,
                    disposition=disposition, reason=reason,
                    revision=_text(raw.get("revision"), 200),
                )
            )
            if len(opinions) >= max_opinions * len(schedule):
                return opinions
    return opinions


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
        moat_rating=_text(payload.get("moat_rating"), 40) or "数据不足",
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
