"""渐进式记忆 · 第 1–2 步：索引块 + 注入 trace。

背景：``agents/loop.py::_profile_prompt`` 原本每轮把用户画像**全量**拼进
system 尾部，只受 ``system_dynamic_max_chars`` 这个整体动态块上限兜底。
画像条数一多，要么常驻占满预算，要么被整体裁剪规则切掉中间行——既没有
「先给索引 + 相关条目」的中间档，也没有任何记录能回答「这轮到底注入了什么」。

本步**不提供**「展开被省略条目」的工具入口：search_memory 只查 episode，
查不到 profile。索引文案因此只做诚实标注（未列出 ≠ 不存在），全文展开
留到第 3 步的 expand_memory 再做。

本模块只做两件纯函数式的事：

1. ``build_memory_index``：按当前问句相关性挑 top-k 条画像，拼「索引 +
   相关条目 + 省略提示」，并裁到独立的字符预算内；
2. ``MemoryTrace``：记录注入/丢弃的 id、得分、字符数与裁剪原因，只进日志。

设计约束（与项目既有记忆哲学一致）：
- **纯加法**：本模块只产出记忆块，不写库、不动 episode、不改抽取链路；
  调用方在任何异常下都回退到全量画像 ``format_profile_block``。
- **零 LLM / 零向量调用**：相关性用确定性字符二元组重叠（中文友好），
  不引入 embedding 请求、不增加延迟、不带来新的失败点。
- **只写不改**：不写库、不动 episode、不改抽取与画像写入链路。
- **fail-open**：调用方负责捕获异常并回退全量画像；本模块保持纯函数，
  异常能直接冒出来以便测试和排障。
- **两块预算单位不同**：``system_dynamic_max_chars`` 是字符上限（用于
  动态块形状控制）；上下文窗口改用 ``llm_context_token_budget`` + Qwen
  tokenizer 计数。摘要/裁剪只认 token。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime

from ai_agent.core.logging import get_logger
from ai_agent.core.text import clip_text
from ai_agent.memory.facets import FACETS, FACET_LABELS, format_profile_line
from ai_agent.prompts.memory import PROFILE_BLOCK_HEADER
from ai_agent.storage.db import MemoryItem

logger = get_logger(__name__)

# 独立省略提示。不能写「其余条目用 search_memory 查询」——该工具只查
# episode（store.search_episode），查不到被省略的 profile 条目；按需展开
# 属于第 3 步的 expand_memory，本步只做诚实标注，不承诺不存在的入口。
_INDEX_CLIP_MARK = "\n[记忆索引已省略 {n} 字符，未列出的画像不代表不存在]\n"
_RECALL_CLIP_MARK = "\n[情景记忆摘要已省略 {n} 字符]\n"
_RECALL_ITEM_MAX_CHARS = 120

# 索引说明行的保守长度预留。先按这个额度给条目留位置，最后再用 clip_text
# 兜一次底，保证任何输入下块长都不超过 max_chars。
_NOTE_RESERVE = 160

_FACET_RANK = {facet: i for i, facet in enumerate(FACETS)}


@dataclass(frozen=True)
class MemoryTrace:
    """一轮记忆注入的可观测快照。只用于日志，不进 prompt、不进响应。"""

    mode: str  # "full" | "index" | "fallback"
    total: int
    injected: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    scores: tuple[tuple[str, float], ...] = ()
    chars: int = 0
    budget: int = 0
    clipped: bool = False
    reason: str = ""

    def emit(self, log=None) -> None:
        """输出一行结构化日志。日志本身失败也不该影响对话主流程。"""
        try:
            (log or logger).info(
                "memory_trace mode=%s total=%d injected=%d dropped=%d "
                "chars=%d budget=%d clipped=%s reason=%s ids=%s scores=%s",
                self.mode,
                self.total,
                len(self.injected),
                len(self.dropped),
                self.chars,
                self.budget,
                self.clipped,
                self.reason,
                ",".join(self.injected[:12]),
                ",".join(f"{mid}:{score:.2f}" for mid, score in self.scores[:12]),
            )
        except Exception:
            pass


def full_trace(
    items: list[MemoryItem],
    block: str,
    *,
    reason: str = "progressive_disabled",
    mode: str = "full",
) -> MemoryTrace:
    """默认分支（全量注入）的 trace。让开关关闭时也能观察现状体量。

    ``mode="fallback"`` 用于渐进链路异常、退回全量画像的那次调用，方便在
    日志里把「本来就没开」和「开了但挂了」区分开。
    """
    return MemoryTrace(
        mode=mode,
        total=len(items),
        injected=tuple(item.id for item in items),
        chars=len(block),
        reason=reason,
    )


def _bigrams(text: str) -> set[str]:
    """字符二元组。中文没有词边界，bigram 重叠比分词/空格切词更稳。"""
    compact = re.sub(r"\s+", "", text.lower())
    if len(compact) < 2:
        return set(compact)
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def relevance(query: str, item: MemoryItem) -> float:
    """0~1 的确定性相关性：问句 bigram 命中条目正文/facet 标签的比例。

    不调模型、不调向量库。query 为空时统一返回 0，排序退化为 facet 顺序 +
    id，保证同一份画像在任何会话里都得到同一结果（可复现、好写测试）。
    """
    q = _bigrams(query)
    if not q:
        return 0.0
    label = FACET_LABELS.get(item.facet or "other", "其他")
    t = _bigrams(f"{label}{item.text}")
    if not t:
        return 0.0
    return len(q & t) / len(q)


# 重要性：按 facet 的固定权重 + 文本里的「长期指令」信号 + episode 命中热度。
# 不做 LLM 打分——记忆排序必须可复现，同一份输入两次跑必须给同一个顺序。
_FACET_IMPORTANCE = {
    "constraint": 1.0,
    "preference": 0.8,
    "name": 0.7,
    "language": 0.7,
    "locale": 0.6,
    "occupation": 0.6,
    "other": 0.4,
}
# 「以后都 / 一律 / 必须 / 不要」这类词说明用户把这条当长期约束
_DIRECTIVE = re.compile(r"以后都|今后|永远|一律|必须|不要|禁止|每次|始终|一律用")


def _parse_stamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def recency_score(item: MemoryItem, *, now: float | None = None) -> float:
    """0~1 的新鲜度：半衰期 ``memory_recency_half_life_days``。"""
    from ai_agent.core.config import settings

    stamp = _parse_stamp(item.last_seen) or _parse_stamp(item.created_at)
    if stamp is None:
        return 0.5
    half_life = max(1, settings.memory_recency_half_life_days) * 86400
    age = max(0.0, (now or time.time()) - stamp)
    return 0.5 ** (age / half_life)


def importance_score(item: MemoryItem) -> float:
    """0~1 的重要性：facet 权重 + 长期指令措辞 + 命中热度。"""
    base = _FACET_IMPORTANCE.get(item.facet or "other", 0.4)
    if _DIRECTIVE.search(item.text or ""):
        base += 0.3
    base += min(item.hits / 5, 1.0) * 0.3
    return min(1.0, base)


def rank_score(item: MemoryItem, relevance_value: float) -> float:
    """三者加权：相关性为主，新鲜度与重要性做修正。

    Generative Agents 的 recency/importance/relevance 三因子，权重可配
    （``MEMORY_RANK_WEIGHT_*``）。相关性为 0 但重要性高的条目仍能进前列——
    这正是「永远不要用表格」这类长期约束该有的行为。
    """
    from ai_agent.core.config import settings

    return (
        settings.memory_rank_weight_relevance * relevance_value
        + settings.memory_rank_weight_recency * recency_score(item)
        + settings.memory_rank_weight_importance * importance_score(item)
    )


def rerank_items(
    items: list[MemoryItem], query: str, *, top_k: int | None = None
) -> list[MemoryItem]:
    """按三因子重排（episode 召回用：向量先取宽一点，再按综合分收窄）。"""
    scored = [(item, relevance(query, item)) for item in items]
    ranked = sorted(scored, key=lambda pair: -rank_score(pair[0], pair[1]))
    picked = [item for item, _ in ranked]
    return picked[:top_k] if top_k else picked


def _facet_counts(items: list[MemoryItem]) -> str:
    counts: dict[str, int] = {}
    for item in items:
        key = item.facet if item.facet in _FACET_RANK else "other"
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (_FACET_RANK.get(kv[0], 99), kv[0]))
    return "、".join(
        f"{FACET_LABELS.get(facet, '其他')} {count}" for facet, count in ordered
    )


def build_memory_index(
    items: list[MemoryItem],
    *,
    user_input: str = "",
    top_k: int = 8,
    max_chars: int = 800,
    expand_hint: bool = False,
    total: int | None = None,
) -> tuple[str, MemoryTrace]:
    """产出「索引 + 相关条目」块，以及本轮注入 trace。

    返回空串表示没有画像，调用方应像原实现一样把该块整个跳过。
    ``top_k <= 0`` 视为不限条数，``max_chars <= 0`` 视为不限字符——两者
    都交给外层 ``system_dynamic_max_chars`` 兜底，便于灰度时单独关掉某一维。
    ``expand_hint`` 只在第 3 步的 ``expand_memory`` 工具确实已注册时开启：
    没挂工具就绝不能提示模型去调它。
    """
    budget = max(0, max_chars)
    if not items:
        return "", MemoryTrace(mode="index", total=0, budget=budget, reason="no_profile")

    has_query = bool(user_input.strip())
    scored = [(item, relevance(user_input, item)) for item in items]
    if has_query:
        # 有问句：相关性 + 新鲜度 + 重要性 三因子
        scored = [(item, rank_score(item, rel)) for item, rel in scored]
        ranked = sorted(scored, key=lambda pair: -pair[1])
    else:
        # 没有问句就排不了相关性，退回 facet + id 的稳定顺序（可复现）；
        # 此时 trace.scores 记的是 relevance（全 0），不是三因子分
        ranked = sorted(
            scored,
            key=lambda pair: (
                -pair[1],
                _FACET_RANK.get(pair[0].facet or "other", 99),
                str(pair[0].id),
            ),
        )
    limit = len(ranked) if top_k <= 0 else max(0, top_k)

    lines = [PROFILE_BLOCK_HEADER]
    chosen: list[MemoryItem] = []
    clipped = False
    room = max(0, budget - _NOTE_RESERVE) if budget else 0
    for item, _score in ranked[:limit]:
        line = format_profile_line(item)
        candidate = "\n".join([*lines, line])
        if room and len(candidate) > room:
            clipped = True
            break
        lines.append(line)
        chosen.append(item)

    # list_profile 默认只载入 20 条。载入已满时 len(items) 不是库内总数，
    # 直接说「画像共 N 条」会低报，还会让 expand_memory 的「完整」承诺落空。
    # 只有调用方查到真实总数且确实超限时才换诚实文案；total=None（默认）
    # 时输出与改动前逐字一致。
    truncated = isinstance(total, int) and total > len(items)
    if truncated:
        note = (
            f"【记忆索引】画像共 {total} 条，本次只载入 {len(items)} 条"
            f"（按类别：{_facet_counts(items)}）；"
            f"本轮注入 {len(chosen)} 条。未列出的画像不代表不存在，"
            "不要据此断言用户没有某项偏好或约束。"
        )
    else:
        note = (
            f"【记忆索引】画像共 {len(items)} 条（按类别：{_facet_counts(items)}）；"
            f"本轮按当前问句相关性注入 {len(chosen)} 条。"
            "这是相关条目而非全量清单，未列出的画像不代表不存在，"
            "不要据此断言用户没有某项偏好或约束。"
        )
    # 第 2 步这里一度写「其余用 search_memory 查询」是错的：那个工具只查
    # episode。第 3 步注册了 expand_memory 后才能给出真实可用的展开入口。
    if expand_hint:
        if truncated:
            note += (
                f"需要更多条目时调用 expand_memory(query)，"
                f"单次最多返回 {len(items)} 条。"
            )
        else:
            note += "需要完整条目时调用 expand_memory(query)。"
    lines.append(note)
    block = "\n".join(lines)
    if budget and len(block) > budget:
        trimmed = clip_text(block, budget, marker=_INDEX_CLIP_MARK)
        clipped = clipped or trimmed != block
        block = trimmed

    injected = tuple(item.id for item in chosen)
    dropped = tuple(item.id for item in items if item.id not in set(injected))
    trace = MemoryTrace(
        mode="index",
        total=len(items),
        injected=injected,
        dropped=dropped,
        scores=tuple((item.id, round(score, 3)) for item, score in ranked[:limit]),
        chars=len(block),
        budget=budget,
        clipped=clipped,
        reason="budget_exceeded" if clipped else "ok",
    )
    return block, trace


def format_recall_block(
    hits: list[MemoryItem], *, max_chars: int = 400
) -> str:
    """把自动召回的情景记忆压成「一行一条的摘要块」。

    第 3 步只做**摘要级**自动召回，全文仍按需由 ``expand_memory`` 取：
    这样每轮固定成本是几条短行，而不是把整个 episode 表塞进 system。

    ``hits`` 由调用方用 ``store.search_episode`` 取回（带 score）；本函数是
    纯函数，不碰 DB、不碰网络，异常由调用方 fail-open 处理。
    """
    if not hits:
        return ""
    lines = ["【相关情景记忆（自动召回）】"]
    for item in hits:
        text = " ".join(str(item.text or "").split())
        if len(text) > _RECALL_ITEM_MAX_CHARS:
            text = text[:_RECALL_ITEM_MAX_CHARS] + "…"
        lines.append(f"- {text}")
    lines.append(
        "以上是摘要而非全量记忆；需要完整原文、或核对索引里未列出的画像条目时，"
        "调用 expand_memory(query)。"
    )
    block = "\n".join(lines)
    if max_chars > 0 and len(block) > max_chars:
        block = clip_text(block, max_chars, marker=_RECALL_CLIP_MARK)
    return block
