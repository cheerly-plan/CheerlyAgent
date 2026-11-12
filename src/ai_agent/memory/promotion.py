"""渐进式记忆 · 第 4 步：观察态、命中证据与晋升梯（决策 + 编排层）。

现状的两个缺口：

1. ``EpisodeStore`` 的 episode 是「抽取命中即落库、7 天 TTL 到期物理删除」，
   反复被用到的项目事实不会固化，一次冷门近况和一条长期约定同等待遇；
2. 画像写入是一步到位，没有「先观察、再候选、经审核才进 L1」的中间态。

本模块把「哪些证据算够、什么能升、什么必须拦」做成**确定性纯函数**，
再由薄薄一层编排把它们接到现有的 store 上。核心纪律与研报推断层一致：

- **候选永不直接进 system 人格块**：晋升要过确定性门禁（跨会话复现 +
  出处），晋升后写入画像才可能被注入。
- **数字/事实类必须带出处**：无出处的数字不给晋升，连候选都不轻易进。
- **用户明确否定立即删除**，不走衰减；长期未命中只降权、不物理删除。
- **恒定开启**：没有总开关。安全性来自门禁本身（候选要复现、数字要出处），
  不来自配置。
- **fail-open**：任何 DB / 依赖异常都只记日志，绝不影响主对话。

第 5 步补上两件事：``/v1/memory/candidates`` 的人工审核端点（批准时可
由审核人核对原文后补录 ``source``，让含数字的候选真正能闭环），以及
``memory/archive.py`` 的 L4 蒸馏归档。候选仍是 pending / promoted /
rejected 三态，归档只是 promoted 与稳定 episode 的只读副本。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.memory.facets import infer_facet, normalize_facet
from ai_agent.storage.db import MemoryItem, current_user_id
from ai_agent.storage.store import store

logger = get_logger(__name__)

MemoryStage = Literal["observe", "candidate", "hold", "reject", "promote"]

# 可核验事实：阿拉伯数字、比例、金额、年份、明确的组织/文件后缀。
# 有这些标记的文本一旦被当成「长期事实」写进画像，出错代价远高于一条偏好。
_NUMBERISH = re.compile(r"\d|[%％]|百分之|[零一二三四五六七八九十百千万亿]+\s*[元块]")
# 只收明确的量化指标词：泛词「报告 / 公告」会把「用户在写年度报告」这类
# 普通近况误判成需核验事实，让它永远卡在 hold 无法进候选。
_FACT_MARK = re.compile(r"份额|占比|同比|环比|营收|市值|增长率|毛利率|净利率")

# 出处写法：只要出现一个可回溯标记就算有出处，不做语义判断（零 LLM）。
_SOURCE_MARK = re.compile(
    r"出处|来源|据|文档|报告|年报|公告|链接|页码|第\s*\d+\s*页|章节|台账|原文|"
    r"source|ref|http",
    re.IGNORECASE,
)

_REMEMBER = re.compile(r"记住|记下来|记一下|记着|别忘了|以后都|每次都|请一直")
_REJECT = re.compile(
    r"别再?记|不要记|不用记|删掉(这条|这个|记忆)|取消(这条|这个)|"
    r"我说错了|不是这样|搞错了|记错了|忽略(这条|这个)"
)


@dataclass(frozen=True)
class PromotionDecision:
    """一条记忆的处置结论。只用于编排与日志，不进 prompt、不进响应。"""

    stage: MemoryStage
    reason: str
    text: str = ""
    facet: str = "other"


@dataclass(frozen=True)
class Candidate:
    """L3 候选层的一条记录（读自 agent_memory_candidate）。"""

    id: str
    user_id: str
    text: str
    facet: str
    source: str
    hit_count: int
    session_seen: int
    status: str


def needs_source(text: str) -> bool:
    """含数字 / 比例 / 金额 / 财务事实 → 晋升前必须能回溯到出处。"""
    return bool(_NUMBERISH.search(text) or _FACT_MARK.search(text))


def has_source(source: str) -> bool:
    return bool((source or "").strip())


def is_explicit_remember(user_input: str) -> bool:
    return bool(_REMEMBER.search(user_input or ""))


def is_explicit_reject(user_input: str) -> bool:
    """刻意收窄：普通对话里的「不是」不算否定，只有明确撤回记忆才算。"""
    return bool(_REJECT.search(user_input or ""))


def evaluate_episode(
    *,
    text: str,
    source: str = "",
    hit_count: int = 0,
    session_seen: int = 0,
    user_input: str = "",
    min_sessions: int = 3,
    min_hits: int = 3,
) -> PromotionDecision:
    """观察态 episode 的晋升判定。证据不足就老实留在观察态。

    判定顺序有意为之：否定 > 出处门禁 > 显式记住 > 跨会话复现 > 观察。
    出处门禁排在显式记住之前——用户说「记住 12.5% 这个数」也不能绕过
    「数字必须有出处」，否则一句口误就会变成长期事实。
    """
    if is_explicit_reject(user_input):
        return PromotionDecision("reject", "explicit_reject", text)
    if needs_source(text) and not has_source(source):
        return PromotionDecision("hold", "missing_source", text)
    if is_explicit_remember(user_input):
        return PromotionDecision("candidate", "explicit_remember", text)
    if session_seen >= min_sessions and hit_count >= min_hits:
        return PromotionDecision("candidate", "recurred", text)
    return PromotionDecision("observe", "insufficient_evidence", text)


def evaluate_candidate(
    candidate: Candidate,
    *,
    min_sessions: int = 3,
    duplicate: bool = False,
) -> PromotionDecision:
    """L3 候选能否升 L1 画像——只看证据，不需要人工批准。

    三道门禁都是**确定性**的，不依赖人：
    * 去重（同文本已在画像里）；
    * 出处门禁：含数字/量化词的条目必须能回溯到出处；
    * 跨会话复现：至少 ``min_sessions`` 个会话里出现过。

    晋升后的写入复用 ``apply_profile_action`` 既有的 facet 归一化、
    ``is_redundant_profile`` 确定性去重与 Jev 语义去重，本层不另造一套。
    """
    if duplicate:
        return PromotionDecision("hold", "duplicate", candidate.text, candidate.facet)
    if needs_source(candidate.text) and not has_source(candidate.source):
        return PromotionDecision("hold", "missing_source", candidate.text, candidate.facet)
    if candidate.session_seen < min_sessions:
        return PromotionDecision(
            "hold", "insufficient_evidence", candidate.text, candidate.facet
        )
    return PromotionDecision("promote", "evidence", candidate.text, candidate.facet)


def decay_hit_count(hit_count: int, idle_seconds: float, decay_seconds: int) -> int:
    """长期未命中：热度减一而非把记忆删掉。decay_seconds <= 0 视为不降权。"""
    if decay_seconds <= 0 or idle_seconds < decay_seconds or hit_count <= 0:
        return hit_count
    return max(0, hit_count - 1)


def should_retain(hit_count: int, min_hits: int) -> bool:
    """反复被用到的 episode 即使过了 TTL 也不物理删除，改由降权慢慢退场。"""
    return min_hits > 0 and hit_count >= min_hits


def observe_episode(
    item: MemoryItem,
    *,
    user_id: str | None = None,
    chat_session_id: str = "",
    user_input: str = "",
    source: str = "",
    facet: str = "",
) -> PromotionDecision | None:
    """写入 episode 后登记观察证据；达到门槛才产出 L3 候选。

    任何异常返回 None 并记日志，绝不阻断抽取链路。
    ``source`` 由调用方给出（目前只有结构化来源才传，普通对话留空）。
    ``facet`` 从抽取结果透传；为空/``other`` 时用确定性线索词推断，
    避免所有候选都落 ``other``、晋升后丢掉偏好/约束语义。
    """
    uid = user_id or current_user_id()
    try:
        if chat_session_id:
            hit_count, session_seen = store.touch_episode_session(
                item.id, chat_session_id, user_id=uid
            )
        else:
            hit_count, session_seen = store.episode_evidence(item.id, user_id=uid)
        decision = evaluate_episode(
            text=item.text,
            source=source,
            hit_count=hit_count,
            session_seen=session_seen,
            user_input=user_input,
            min_sessions=settings.memory_promotion_min_sessions,
            min_hits=settings.memory_promotion_min_hits,
        )
        if decision.stage == "candidate":
            candidate_facet = normalize_facet(facet) if facet else "other"
            if candidate_facet == "other":
                candidate_facet = infer_facet(item.text)
            store.upsert_candidate(
                text=item.text,
                facet=candidate_facet,
                source=source,
                hit_count=hit_count,
                session_seen=session_seen,
                user_id=uid,
            )
        elif decision.stage == "reject":
            store.delete_candidates_matching(item.text, user_id=uid)
        logger.info(
            "memory_promotion stage=%s reason=%s hits=%d sessions=%d episode=%s",
            decision.stage,
            decision.reason,
            hit_count,
            session_seen,
            item.id,
        )
        return decision
    except Exception:
        logger.exception("记忆观察态登记失败 episode=%s", item.id)
        return None


def reject_candidates(text: str, *, user_id: str | None = None) -> int:
    """用户明确否定：候选立即删除，不走衰减。"""
    if not (text or "").strip():
        return 0
    uid = user_id or current_user_id()
    try:
        return store.delete_candidates_matching(text, user_id=uid)
    except Exception:
        logger.exception("候选记忆删除失败")
        return 0


def review_pending_candidates(*, user_id: str | None = None, limit: int = 20) -> int:
    """后台审核 L3 候选：证据够就自动晋升，不需要人工逐条批准。

    批准后的写入复用 ``apply_profile_action``：它已经带了 facet 归一化、
    ``is_redundant_profile`` 确定性去重与 Jev 语义去重，本层不另造一套冲突
    检测，避免两套门禁口径漂移。写入被去重拦下（返回 None）时候选保持
    pending 并标注，不误标为已晋升。
    """
    uid = user_id or current_user_id()
    approved_count = 0
    try:
        rows = store.list_candidates(uid, status="pending", limit=limit)
    except Exception:
        logger.exception("候选记忆读取失败")
        return 0
    for row in rows:
        candidate = Candidate(
            id=str(row.get("id") or ""),
            user_id=uid,
            text=str(row.get("text") or ""),
            facet=str(row.get("facet") or "other"),
            source=str(row.get("source") or ""),
            hit_count=int(row.get("hit_count") or 0),
            session_seen=int(row.get("session_seen") or 0),
            status="pending",
        )
        decision = evaluate_candidate(
            candidate,
            min_sessions=settings.memory_promotion_min_sessions,
        )
        if decision.stage != "promote":
            logger.info(
                "memory_candidate hold id=%s reason=%s", candidate.id, decision.reason
            )
            continue
        try:
            # 延迟导入：profile.py 会 import 本模块做否定处理，模块级互导会成环。
            from ai_agent.memory.profile import ProfileAction, apply_profile_action

            written = apply_profile_action(
                ProfileAction(op="add", facet=candidate.facet, content=candidate.text),
                user_id=uid,
            )
            if written is None:
                logger.info(
                    "memory_candidate deduped id=%s text=%s",
                    candidate.id,
                    candidate.text[:60],
                )
                continue
            store.mark_candidate(candidate.id, "promoted", user_id=uid)
            approved_count += 1
        except Exception:
            logger.exception("候选记忆晋升失败 id=%s", candidate.id)
    return approved_count


def approve_candidate(
    candidate_id: str,
    *,
    user_id: str | None = None,
    source: str | None = None,
    reviewed_by: str = "",
    note: str = "",
) -> bool:
    """人工批准单条候选 → 升 L1 画像。

    自动审核（``review_pending_candidates``）已经按证据放行，这个入口是
    **人工兜底**：审核人核对原文后可以补录出处，让原本卡在出处门禁的条目
    通过；也可以提前放行证据还不足的条目。仍然要过 ``evaluate_candidate``
    的出处门禁与 ``apply_profile_action`` 的去重门禁——**人工批准不等于绕过
    事实校验**。

    ``source`` 是第 5 步新增的人工出处补录：含数字/量化词的候选在抽取阶段
    因为「结构化出处通道未打通」而恒为 hold，审核人核对原文后把出处填进
    请求体，才能在批准时放行。字符串来自人，绝不来自 LLM 现场抽数。
    不传 source 时读取路径与第 4 步逐字一致。
    """
    uid = user_id or current_user_id()
    try:
        if source and source.strip():
            row = store.update_candidate_source(candidate_id, source, user_id=uid)
        else:
            row = store.get_candidate(candidate_id, user_id=uid)
        if not row or row.get("status") != "pending":
            return False
        candidate = Candidate(
            id=str(row.get("id") or ""),
            user_id=uid,
            text=str(row.get("text") or ""),
            facet=str(row.get("facet") or "other"),
            source=str(row.get("source") or ""),
            hit_count=int(row.get("hit_count") or 0),
            session_seen=int(row.get("session_seen") or 0),
            status="pending",
        )
        # 人工批准：跳过「跨会话复现」这条证据门槛，其余门禁照旧
        decision = evaluate_candidate(candidate, min_sessions=0)
        if decision.stage != "promote":
            logger.info(
                "memory_candidate approval blocked id=%s reason=%s by=%s",
                candidate_id,
                decision.reason,
                reviewed_by or "-",
            )
            return False
        from ai_agent.memory.profile import ProfileAction, apply_profile_action

        written = apply_profile_action(
            ProfileAction(op="add", facet=candidate.facet, content=candidate.text),
            user_id=uid,
        )
        if written is None:
            return False
        mark_kwargs: dict = {"user_id": uid}
        if reviewed_by or note:
            mark_kwargs.update(reviewed_by=reviewed_by, note=note)
        return store.mark_candidate(candidate_id, "promoted", **mark_kwargs)
    except Exception:
        logger.exception("候选记忆人工批准失败 id=%s", candidate_id)
        return False


def reject_candidate(
    candidate_id: str,
    *,
    user_id: str | None = None,
    reviewed_by: str = "",
    note: str = "",
) -> bool:
    """人工驳回单条候选（记审核人/备注，不删行以便追溯）。"""
    uid = user_id or current_user_id()
    try:
        row = store.get_candidate(candidate_id, user_id=uid)
        if not row or row.get("status") != "pending":
            return False
        return store.mark_candidate(
            candidate_id,
            "rejected",
            user_id=uid,
            reviewed_by=reviewed_by,
            note=note,
        )
    except Exception:
        logger.exception("候选记忆人工驳回失败 id=%s", candidate_id)
        return False


def review_all_pending_candidates(*, limit_per_user: int = 20) -> int:
    """后台线程入口：按用户遍历待审候选。"""
    try:
        users = store.list_candidate_users(status="pending", enabled_only=True)
    except Exception:
        logger.exception("候选记忆用户列表读取失败")
        return 0
    return sum(
        review_pending_candidates(user_id=uid, limit=limit_per_user) for uid in users
    )
