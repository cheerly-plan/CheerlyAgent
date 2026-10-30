"""图谱检索的候选打分：向量分之外再叠一层字面依据。

从 graph_store.py 拆出(S5)。向量模型对完全不相关的词也会给 0.45~0.52 的
相似度（模型基线就这么高），只靠绝对阈值切不干净：搜「天气预报」照样能召回
一堆芯片公司。所以这里在向量分之上叠加字面证据，并让「没有字面证据」的候选
自己过一道更高的门槛：

- 名称完全相同 +0.30（搜「HBM」命中实体「HBM」）
- 名称互相包含 +0.18（搜「长鑫」命中「长鑫存储」）
- 检索词 token 命中名称或描述 最多 +0.12

有字面证据的候选直接保留，避免因为描述写法不同漏掉真正相关的实体。
"""

from __future__ import annotations

import re



# ——— 检索打分：向量分 + 字面依据 ———
#
# 向量模型对完全不相关的词也会给 0.45~0.52 的相似度（模型基线就这么高），
# 只靠绝对阈值切不干净：搜「天气预报」照样能召回一堆芯片公司。所以在向量分
# 之上再叠一层字面证据，并让「没有字面证据」的候选自己过一道更高的门槛：
#   · 名称完全相同        +0.30（搜「HBM」命中实体「HBM」）
#   · 名称互相包含        +0.18（搜「长鑫」命中「长鑫存储」）
#   · 检索词 token 命中名称或描述  最多 +0.12
# 有字面证据的候选直接保留，避免因为描述写法不同漏掉真正相关的实体。

_STRIP_PUNCT = re.compile(
    r"""[\s\-_/·・,，。.、:：;；!！?？'"“”‘’()（）\[\]【】<>《》]+"""
)
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_WORD = re.compile(r"[a-z0-9][a-z0-9+#.]*")

NAME_BOOST_EXACT = 0.30
NAME_BOOST_PART = 0.18
KEYWORD_BOOST_MAX = 0.12
# token 命中到这个比例即视为「有字面依据」，不再受相似度门槛约束
KEYWORD_HIT_RATIO = 0.5
# 确认「问到点子上了」之后，其余候选的分数线可以比 min_score 低这么多
WEAK_SCORE_DROP = 0.12
# 同时和最强候选比：差得太远的（名字对上了但只是同话题）也一并切掉
RELATIVE_BAND = 0.3
# 主体锚类型：这类实体（公司/机构）命中后，跨文件同名 metric 要隔离，避免串公司
_ANCHOR_TYPES = {"org"}
# 受「主体锚 + file_id 收敛」约束的类型：指标/概念这类词家家都有，最易跨文件串入
_SCOPED_TYPES = {"metric", "concept", "other"}


def _norm_text(text: str) -> str:
    """与实体 norm_name 同口径：去掉空白与常见标点、转小写。"""
    return _STRIP_PUNCT.sub("", (text or "").lower())


def _query_tokens(query: str) -> list[str]:
    """检索词切词：英文数字按词切，中文按二元组（单字太泛、整串太严）。"""
    text = (query or "").lower()
    tokens = _LATIN_WORD.findall(text)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return [t for t in tokens if t]


def _literal_match(
    query_norm: str, tokens: list[str], name: str, description: str
) -> tuple[float, str, float]:
    """字面匹配：返回（加分, 依据类型, 关键词命中比例）。"""
    name_norm = _norm_text(name)
    if not query_norm or not name_norm:
        return 0.0, "", 0.0
    if query_norm == name_norm:
        return NAME_BOOST_EXACT, "name", 1.0
    if len(name_norm) >= 2 and (query_norm in name_norm or name_norm in query_norm):
        return NAME_BOOST_PART, "name", 1.0
    if not tokens:
        return 0.0, "", 0.0
    haystack = name_norm + _norm_text(description)
    ratio = sum(1 for token in tokens if token in haystack) / len(tokens)
    if ratio <= 0:
        return 0.0, "", 0.0
    return min(KEYWORD_BOOST_MAX, KEYWORD_BOOST_MAX * ratio), "keyword", ratio


# 查询包含别名时，别名太短容易误中（「国电」出现在很多公司名里）
_ALIAS_IN_QUERY_MIN = 4


def _score_candidate(
    query_norm: str,
    tokens: list[str],
    row: dict,
    alias_name: str = "",
) -> dict:
    """给一条实体候选打分。别名按名称口径匹配：对齐删掉的简称要能成为强命中。"""
    vector = float(row.get("score") or 0)
    boost, basis, ratio = _literal_match(
        query_norm, tokens, row.get("name") or "", row.get("description") or ""
    )
    if alias_name:
        a_boost, a_basis, a_ratio = _literal_match(
            query_norm, tokens, alias_name, ""
        )
        if a_basis == "name" and basis != "name":
            boost, basis, ratio = a_boost, "alias", a_ratio
        elif basis != "name" and a_boost > boost:
            boost, basis, ratio = (
                a_boost,
                "alias" if a_basis == "name" else a_basis,
                a_ratio,
            )
    literal = basis in {"name", "alias"} or ratio >= KEYWORD_HIT_RATIO
    return {
        **row,
        "score": min(1.0, vector + boost),
        "vector": vector,
        "literal": literal,
        "matched": basis or "vector",
    }


def _rank_candidates(
    scored: list[dict], *, min_score: float, top_k: int
) -> list[dict]:
    """把打分后的候选挑成种子列表。

    三条规则：
    1. 整批都没有「问到点子上」的证据（字面命中，或向量分过 min_score）→ 返回空。
       向量对无关词也会给 0.45~0.52，不这样切，搜「天气预报」也能拉出一串芯片公司。
    2. 一旦确认问到了，字面命中的全留；纯向量的候选过一条双重线：
       不低于 `min_score - WEAK_SCORE_DROP`，也不比最强候选低超过 `RELATIVE_BAND`。
       前者保证「相关但描述里没写这个词」的实体不被切掉，后者砍掉只是同话题的尾部。
    3. 同一实体在每份文档里各有一条记录，按 norm_name 合并；代表记录取提及次数
       最多的那条（与全库图合并同一规则），分数取组内最高。

    主体收敛（解决跨公司串数据）：当候选里出现「字面强命中 + org 类型」的主体锚
    （如明确问「安克创新」）时，同名的指标类实体（metric/concept）在各家文件里
    都有，向量召回会把别家公司的同名词也带进来。此时：
      - 与主体锚同文件（file_id 一致）的候选照常走放宽线；
      - 跨文件（file_id 不在锚集合里）的指标类候选不享受 WEAK_SCORE_DROP 放宽，
        必须字面命中或向量分 ≥ min_score 原值才保留。
    没有主体锚（只问「毛利率」没提公司）时保持现状，交给下游/LLM 消歧，不硬过滤。
    """
    if not scored:
        return []
    if not any(r["literal"] or float(r["vector"]) >= min_score for r in scored):
        return []

    # 主体锚：有字面强命中（matched 是 name/alias）且 type=org 的实体
    anchor_files: set[str] = {
        str(r["file_id"])
        for r in scored
        if r.get("type") in _ANCHOR_TYPES and r["literal"] and r.get("matched") in {"name", "alias"}
    }

    best_vector = max(float(r["vector"]) for r in scored)
    floor = max(min_score - WEAK_SCORE_DROP, best_vector - RELATIVE_BAND)

    # 有主体锚时，按 (norm_name, file_id) 分组，避免安克/顺丰的同名 metric 被合并
    # 成一个代表（取错文件）；无主体锚时仍按 norm_name 合并（保持原行为）。
    group_key = (
        (lambda r: (r["norm_name"] or r["name"], str(r["file_id"])))
        if anchor_files
        else (lambda r: r["norm_name"] or r["name"])
    )

    grouped: dict[str, list[dict]] = {}
    for row in scored:
        if row["literal"]:
            # 字面命中直接保留
            grouped.setdefault(group_key(row), []).append(row)
            continue
        vector = float(row["vector"])
        # 主体收敛：跨文件的指标类候选，不享受放宽，必须过 min_score 原值
        if (
            anchor_files
            and row.get("type") in _SCOPED_TYPES
            and str(row["file_id"]) not in anchor_files
        ):
            if vector < min_score:
                continue
        elif vector < floor:
            continue
        grouped.setdefault(group_key(row), []).append(row)

    picked: list[dict] = []
    for group in grouped.values():
        best = max(
            group,
            key=lambda r: (int(r["mention_count"] or 0), r["name"] or ""),
        )
        top = max(group, key=lambda r: float(r["score"] or 0))
        picked.append({**best, "score": top["score"], "matched": top["matched"]})

    picked.sort(key=lambda r: float(r["score"] or 0), reverse=True)
    return picked[:top_k]
