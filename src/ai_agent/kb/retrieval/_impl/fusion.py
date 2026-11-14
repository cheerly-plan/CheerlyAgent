"""多路融合：RRF 合并、去冗余、按主体与文件配额打散。

由原 retrieval.py 拆分而来，逻辑未改动。
"""

from __future__ import annotations

import re
import uuid

from ai_agent.kb.ingest.tables import SCOPE_PARENT
from ai_agent.kb.ingest.tables import SCOPE_CONSOLIDATED
from ai_agent.storage.store import store
from ai_agent.core.config import settings

from ai_agent.core.logging import get_logger

from ai_agent.kb.retrieval._impl.types import (
    Hit,
)

logger = get_logger(__name__)


# 测试用 monkeypatch.setattr(retrieval, NAME, ...) 打补丁的知识点：
# 直接 NAME(...) 走的是本模块 globals，绕过模块属性，补丁会失效；
# 统一经 _public() 取「对外模块」的当前属性再调用，补丁才生效。
def _public():
    import ai_agent.kb.retrieval.retrieval as public

    return public


def _fuse_with_keywords(
    query: str,
    vector_rows: list[dict],
    file_ids: list[uuid.UUID] | None,
    *,
    filters: dict | None = None,
) -> list[dict]:
    """混合检索：向量召回 + 关键词召回做 RRF 融合。

    两路各按名次贡献 1/(60+名次) 的分数，都命中的片段叠加后天然排前。
    关键词路失败/为空时退回纯向量，绝不把主检索拖挂。
    """
    if not settings.kb_hybrid_enabled:
        return vector_rows
    keywords = _public()._extract_keywords(query)
    if not keywords:
        return vector_rows
    keywords = _public()._apply_short_phrases(query, keywords, file_ids)
    if not keywords:
        return vector_rows
    keywords = _public()._keep_distinctive_keywords(keywords, file_ids)
    if not keywords:
        return vector_rows
    try:
        keyword_rows = store.search_kb_chunks_keyword(
            keywords,
            file_ids=file_ids,
            top_k=settings.kb_hybrid_candidates,
            **(filters or {}),
        )
    except Exception:
        logger.warning("kb 关键词召回失败，退回纯向量", exc_info=True)
        return vector_rows
    if not keyword_rows:
        return vector_rows
    return _public()._rrf_merge(vector_rows, keyword_rows, n_keywords=len(keywords))

def _keep_distinctive_keywords(
    keywords: list[str], file_ids: list[uuid.UUID] | None
) -> list[str]:
    """丢掉全库过半切片都含有的词。

    「营业收入」「同比」这类词命中与否不代表相关性：财务表几乎每一片都有，
    放进关键词路等于把整份报表召回回来，再靠 RRF 的「双路命中」把真正
    回答问题的切片挤下去——实测把含 -4.42 的那片从向量第 2 位挤到第 6 位，
    top_k 一取就没了。只有编号、专名这类稀有词才配得上「命中即相关」。

    词频取不到时保留全部：宁可沿用旧行为，也不让一次统计失败改变召回。
    """
    try:
        total, freq = store.keyword_doc_freq(keywords, file_ids=file_ids)
    except Exception:
        logger.warning("kb 关键词词频统计失败，保留全部关键词", exc_info=True)
        return keywords
    if total <= 0:
        return keywords
    kept = [kw for kw in keywords if freq.get(kw, 0) * 2 <= total]
    if len(kept) < len(keywords):
        logger.info(
            "kb 关键词筛除无区分度词：%s",
            [kw for kw in keywords if kw not in set(kept)],
        )
    return kept

def _rrf_merge(
    vector_rows: list[dict],
    keyword_rows: list[dict],
    n_keywords: int | None = None,
) -> list[dict]:
    """两路召回按名次做 RRF 融合：score = Σ 权重/(60+名次)，双路命中叠加。

    关键词路默认带 `_RRF_KEYWORD_WEIGHT` 的折扣：它只数「命中了几个词」，
    不像向量路衡量语义相关性，同权会盖过真正带数据的切片。但「全命中」
    （一条切片命中了本次检索的全部查询词，即 `keyword_score == n_keywords`）
    是强字面信号，基本就是用户要的字面答案，此时恢复 `_RRF_KEYWORD_WEIGHT_FULL`
    权重，让其压过向量路中低名次、进入重排候选。`n_keywords` 为 None 时
    （旧调用方/离线用例）一律走 0.5 折扣，保持原行为。

    关键词独有片段没有 distance（向量化这条路没走过），保持缺失；
    _to_hit 写成 None，不能填 0。融合结果按分数降序返回。
    """
    scores: dict[tuple[str, int], float] = {}
    merged: dict[tuple[str, int], dict] = {}
    for rank, row in enumerate(vector_rows):
        key = (str(row["file_id"]), int(row["seq"]))
        merged.setdefault(key, dict(row))
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, row in enumerate(keyword_rows):
        key = (str(row["file_id"]), int(row["seq"]))
        if key not in merged:
            merged[key] = dict(row)
        kw_score = row.get("keyword_score")
        weight = (
            _RRF_KEYWORD_WEIGHT_FULL
            if n_keywords and kw_score == n_keywords
            else _RRF_KEYWORD_WEIGHT
        )
        scores[key] = scores.get(key, 0.0) + weight / (_RRF_K + rank + 1)
    ranked = sorted(merged.items(), key=lambda item: scores[item[0]], reverse=True)
    # 分数留在行上：口径偏好要按分数比例加权，只改名次会破坏 RRF 的相对差距
    fused = []
    for key, row in ranked:
        row = dict(row)
        row["_fused_score"] = scores[key]
        fused.append(row)
    logger.info(
        "kb 混合检索 向量 %d 条 + 关键词 %d 条 → 融合 %d 条",
        len(vector_rows), len(keyword_rows), len(fused),
    )
    return fused

def _prefer_scope(query: str) -> str:
    """用户的默认口径：明说母公司才按单体，其余都按集团（合并）理解。"""
    return SCOPE_PARENT if _PARENT_INTENT.search(query) else SCOPE_CONSOLIDATED

def _apply_scope_preference(query: str, rows: list[dict]) -> list[dict]:
    """按查询的默认口径给报表切片做小幅加成，重排候选顺序。

    只在**两种口径同时出现**时生效：那才构成"同一个指标两份数"的歧义。
    只有合并表（很多公司不披露母公司报表）或候选里没有报表时，不存在
    选错的问题，一律不动——否则用户问「有哪些产品」，候选里的合并报表
    会因为"命中口径偏好"白拿 6%，把产品描述挤下去。
    """
    if len(rows) < 2:
        return rows
    present = {row.get("scope") for row in rows}
    if not {SCOPE_PARENT, SCOPE_CONSOLIDATED} <= present:
        return rows
    prefer = _prefer_scope(query)
    scored: list[tuple[float, int, dict]] = []
    for rank, row in enumerate(rows):
        base = row.get("_fused_score")
        if base is None:
            # 纯向量路径（混合检索关闭或无关键词）没走 RRF：按名次折算，
            # 与融合路径同一量级，避免两套分数不可比
            base = 1.0 / (_RRF_K + rank + 1)
        scope = row.get("scope")
        factor = 1.0 if not scope else (
            _SCOPE_MATCH if scope == prefer else _SCOPE_MISMATCH
        )
        scored.append((base * factor, rank, row))
    # 同分时保持原顺序（rank 升序），不让口径规则把并列项搅乱
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored]

def _diversify(
    query: str, rows: list[dict], top_k: int, *, per_file: bool = False
) -> list[dict]:
    """多期问句、或多家公司一起查时按文件留名额，其余问句仍按主体留名额。

    ``per_file`` 用在范围已经收成这几份文件的时候。股东表的切片主体会被改成
    股东名，按主体留名额会把一家公司的股东表排满，另一家进不了 top_k。
    """
    from ai_agent.kb.ingest.meta import query_period_policy

    policy, _token = query_period_policy(query or "")
    if per_file or policy in {"all", "span_q", "span_h"}:
        return _public()._diversify_by_file(query, rows, top_k)
    return _public()._diversify_by_subject(rows, top_k)

def _focused_file_ids(query: str, rows: list[dict]) -> set[str]:
    """结果里被问句点名的文件。用文件名还原主体，不看切片上被改写过的主体。"""
    from ai_agent.kb.ingest.meta import extract_subject, fragment_subjects

    catalog: list[tuple[str, list[str]]] = []
    by_subject: dict[str, list[str]] = {}
    for row in rows:
        subject = extract_subject(str(row.get("filename") or ""))
        fid = str(row.get("file_id") or "")
        if not subject or not fid:
            continue
        by_subject.setdefault(subject, []).append(fid)
        catalog.append((subject, []))
    mentioned = fragment_subjects(query, catalog)
    for subject in by_subject:
        if _public().name_in_text(query, subject):
            mentioned.add(subject)
    found: set[str] = set()
    for subject in mentioned:
        found.update(by_subject.get(subject, []))
    return found

def _diversify_by_file(query: str, rows: list[dict], top_k: int) -> list[dict]:
    """同一主体的两份报告各留一条，避免后一年把前一年挤出 top_k。

    股东表会被归属到股东名上，按切片 subject 分桶分不开 2025 和 2026。
    这里按文件留名额。问句点了哪家，就先留那家各期的头名。
    """
    if top_k <= 0 or len(rows) <= top_k:
        return rows[:top_k]
    head_at: dict[str, int] = {}
    for index, row in enumerate(rows):
        fid = str(row.get("file_id") or "")
        if fid and fid not in head_at:
            head_at[fid] = index
    if len(head_at) <= 1:
        return _public()._diversify_by_subject(rows, top_k)
    focus = _public()._focused_file_ids(query, rows)
    if focus:
        ordered = [i for fid, i in head_at.items() if fid in focus]
        ordered += [i for fid, i in head_at.items() if fid not in focus]
    else:
        ordered = list(head_at.values())
    chosen: list[int] = []
    seen: set[int] = set()
    for index in ordered:
        if len(chosen) >= top_k:
            break
        chosen.append(index)
        seen.add(index)
    for index in range(len(rows)):
        if len(chosen) >= top_k:
            break
        if index not in seen:
            chosen.append(index)
            seen.add(index)
    if not focus:
        return [rows[index] for index in sorted(chosen)]
    # 点名主体的各期放在前面，模型先看到两年，而不是先读完一年的变体
    focus_n = sum(1 for fid in head_at if fid in focus)
    front = chosen[:focus_n]
    rest = sorted(index for index in chosen[focus_n:])
    return [rows[index] for index in front + rest]

def _diversify_by_subject(rows: list[dict], top_k: int) -> list[dict]:
    """重排之后按主体留名额：够近的主体至少一条进 top_k。

    召回阶段的主体配额只保证候选里有第二家，cross-encoder 仍可能把名额
    全给分数更高的那家。这里在最终截断前，把带宽内每个主体的重排头名
    换进结果，其余位置仍按重排顺序填。带宽为 0 时关闭。没有主体、或只有
    一个主体时原样截断。
    """
    if top_k <= 0 or len(rows) <= top_k:
        return rows[:top_k]
    band = settings.kb_subject_quota_band
    if band <= 0:
        return rows[:top_k]

    def distance_of(row: dict) -> float | None:
        raw = row.get("distance")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    known = [dist for dist in (distance_of(row) for row in rows) if dist is not None]
    best = min(known) if known else None

    def within_band(row: dict) -> bool:
        dist = distance_of(row)
        if dist is None or best is None:
            return True
        return dist <= best + band

    head_at: dict[str, int] = {}
    for index, row in enumerate(rows):
        subject = str(row.get("subject") or "").strip()
        if not subject or subject in head_at:
            continue
        if within_band(row):
            head_at[subject] = index
    if len(head_at) <= 1:
        return rows[:top_k]

    chosen = set(sorted(head_at.values())[:top_k])
    for index in range(len(rows)):
        if len(chosen) >= top_k:
            break
        chosen.add(index)
    picked = [rows[index] for index in sorted(chosen)]
    if {id(row) for row in picked} != {id(row) for row in rows[:top_k]}:
        logger.info(
            "kb 重排后按主体留名额：%s",
            "、".join(sorted(head_at)),
        )
    return picked

def _dedupe(rows: list[dict]) -> list[dict]:
    """按向量顺序去冗余：与任一已保留片段三字组重合度过高的直接丢。"""
    kept: list[dict] = []
    # 只在同一份文件里去重。两年的股东表结构很像，跨文件丢掉会把前一年整份抹掉。
    kept_grams: dict[str, list[set[str]]] = {}
    for row in rows:
        fid = str(row.get("file_id") or "")
        grams = _public()._trigrams(row.get("text") or "")
        prior = kept_grams.setdefault(fid, [])
        if any(_public()._jaccard(grams, other) >= _DUP_THRESHOLD for other in prior):
            continue
        kept.append(row)
        prior.append(grams)
    return kept

def _to_hit(row: dict) -> Hit:
    return Hit(
        file_id=str(row["file_id"]),
        filename=row.get("filename") or "",
        seq=int(row["seq"]),
        page=row.get("page"),
        kind=row.get("kind") or "text",
        text=_public().render_otsl(row.get("text") or ""),
        distance=_public()._vector_distance(row),
        page_end=row.get("page_end"),
        section=row.get("section"),
        scope=row.get("scope"),
        period=str(row.get("period") or ""),
    )

def _vector_distance(row: dict) -> float | None:
    raw = row.get("distance")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None

# 与已保留片段的三字组 Jaccard 超过这个值就当成重复（overlap 造成的近似片）
_DUP_THRESHOLD = 0.72

# RRF 融合平滑项（业界惯用 60）：排名越靠前贡献越大，两路都命中会叠加
_RRF_K = 60

# 关键词路在融合里的权重。向量路的排序是「语义相关性」，关键词路只是
# 「字面命中个数」——没有 IDF、没有长度归一，同分还按 (file_id, seq) 排，
# 于是封面页、目录、页眉这类靠前又短、恰好写着公司名的片段常常排到第一。
# 权重 1.0（两路同权）时，这种片段能和向量第一名打平，把真正带数据的
# 切片挤下去；降到 0.5 后它只算「补充证据」，压不过向量路的高名次。
_RRF_KEYWORD_WEIGHT = 0.5

# 关键词「全命中」（一条切片命中了本次检索的全部查询词，见 `_rrf_merge`
# 的 `n_keywords`）时恢复权重到 1.0。这类切片基本就是用户要的字面答案
# （如封面「证券代码：600916」同时含「证券」和「代码」）：原统一 0.5 会把
# 它压在向量第 16 名之下、被 RERANK_LIMIT 砍掉，于是「证券代码是多少」
# 答到其他地方。要压过向量第 16 名需 w/61 > 1/76 → w > 0.803，故取整值
# 1.0。只有「全命中」才恢复，部分命中的目录/页眉仍保持 0.5 折扣，以防
# 字面噪声挤掉真正带数据的切片。
_RRF_KEYWORD_WEIGHT_FULL = 1.0

# ——— 报表口径偏好 ———
# 同一指标在合并报表与母公司报表里各有一份数，值可能差几十倍（能建期末
# 现金：合并 77,436,294、母公司 2,721,444）。用户问「中国能建期末现金」
# 默认问的是集团口径，但母公司那张表往往更短、通篇就是这个指标名，
# 向量相似度反而更高，实测把错口径排到第 1。
#
# 系数为什么这么小：RRF 的 k=60 把头部分数压得很平（第 1 名 1/61 vs
# 第 3 名 1/63，只差 3%），1.06 足以让第 3 名反超第 1 名，又不至于把
# 一段相关性差一截的切片顶上来。
_SCOPE_MATCH = 1.06

_SCOPE_MISMATCH = 0.94

# 问「母公司/单体」才切到 parent，其余一律按集团口径理解
_PARENT_INTENT = re.compile(r"母公司|个别报表|单体报表|母公司口径")
