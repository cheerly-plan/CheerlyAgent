"""知识库检索编排：向量召回 → 去冗余 → LLM 重排 → 邻接扩展。

只负责"取哪几段",不负责拼给模型的文案（那是 tools/kb_search 的事）——
图谱检索、对话工具、以后的评测可以共用同一套检索策略。

四级处理各自的必要性：
1. 召回放大到 candidates：直接取最近邻 3 条，常常全是同一段话的变体，
   真正回答问题的片段排在第四位；
2. 去冗余：切片有 overlap，相邻切片内容本来就高度重合，按三字组
   Jaccard 去重比引入 MMR + 向量再计算便宜得多；
3. 重排：向量相似度只看"像不像"，重排看"能不能回答问题"，后者更准；
4. 邻接扩展：切片按长度切，跨段落的结论会被切断，命中片前后各带一段。
"""

from __future__ import annotations

import difflib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.meta import ENTITY_SUFFIX as _ENTITY_SUFFIX
from ai_agent.kb.meta import EXT_RE as _EXT_RE
from ai_agent.kb.meta import FILENAME_NOISE as _FILENAME_NOISE
from ai_agent.kb.meta import NAME_PUNCT as _NAME_PUNCT
from ai_agent.kb.otsl import render_otsl
from ai_agent.kb.parsing import FIGURE, TABLE
from ai_agent.kb.tables import SCOPE_CONSOLIDATED, SCOPE_PARENT
from ai_agent.memory.embeddings import embed_query
from ai_agent.memory.store import store
from ai_agent.prompts.kb import build_rerank_user

logger = get_logger(__name__)

# 与已保留片段的三字组 Jaccard 超过这个值就当成重复（overlap 造成的近似片）
_DUP_THRESHOLD = 0.72
# 送重排时单段文本截断长度：重排只看相关性，不需要读完整段
_RERANK_CLIP = 300
# 送重排的候选上限：再多提示词就太长，而且第 13 名之后本来就进不了 top_k
_RERANK_LIMIT = 12
# RRF 融合平滑项（业界惯用 60）：排名越靠前贡献越大，两路都命中会叠加
_RRF_K = 60
# 关键词路在融合里的权重。向量路的排序是「语义相关性」，关键词路只是
# 「字面命中个数」——没有 IDF、没有长度归一，同分还按 (file_id, seq) 排，
# 于是封面页、目录、页眉这类靠前又短、恰好写着公司名的片段常常排到第一。
# 权重 1.0（两路同权）时，这种片段能和向量第一名打平，把真正带数据的
# 切片挤下去；降到 0.5 后它只算「补充证据」，压不过向量路的高名次。
_RRF_KEYWORD_WEIGHT = 0.5

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

# 查询里的功能词/疑问词，几乎每条查询都有、对召回没有区分度
_STOPWORDS = {
    "怎么", "如何", "什么", "哪些", "多少", "是否", "请问", "查询",
    "查找", "检索", "相关", "关于", "介绍", "说明", "可以", "需要",
    "应该", "如果", "我们", "他们", "以及", "还有", "一下", "这些",
    "文档", "文件", "知识库", "资料",
}

_JSON_ARRAY = re.compile(r"\[[^\[\]]*\]")
# Excel 的 page 是工作表序号；PDF/Word 的 page 是页码。展示名不能混用「表N」。
_SHEET_EXTS = {".xlsx", ".xlsm", ".xls", ".csv"}

# ——— 文件名主体收窄（tags 为空时的兜底信号） ———
# 查询与文件主体的最长公共子串至少有 3 个字、且占主体核心名 40% 以上，
# 才认定「这一问在问这家公司」。实测「中国能建」×「中国能源建设」= 3/6，
# 「顺丰控股」×全称 = 4/4；库里没被问到的公司一律 0。
_SUBJECT_MIN_COMMON = 3
_SUBJECT_MIN_RATIO = 0.4


def is_sheet_file(filename: str = "", mime: str = "") -> bool:
    suffix = Path(filename or "").suffix.lower()
    ext = (mime or "").lower()
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    return suffix in _SHEET_EXTS or ext in _SHEET_EXTS


def format_where(
    *,
    kind: str,
    page: int | None,
    page_end: int | None = None,
    seq: int = 0,
    filename: str = "",
    mime: str = "",
) -> str:
    """人读的引用位置：PDF 表格落在页上，只有电子表才叫「表N」。"""
    if kind == FIGURE:
        return f"第{page}页图表" if page is not None else "图表"
    if kind == TABLE:
        if page is None:
            return "表格"
        if is_sheet_file(filename, mime):
            return f"表{page}"
        if page_end is not None and page_end != page:
            return f"第{page}–{page_end}页"
        return f"第{page}页"
    if page is None:
        return f"第{seq + 1}段"
    if page_end is not None and page_end != page:
        return f"第{page}–{page_end}页"
    return f"第{page}页"


@dataclass
class Hit:
    """一条命中：命中片本体 + 前后邻接片段（拼上下文用）。"""

    file_id: str
    filename: str
    seq: int
    page: int | None
    kind: str
    text: str
    # 向量余弦距离；关键词独有命中没有走过向量路，保持 None，不能填 0。
    distance: float | None
    # 跨页切片的末页（同节内聚合会一片跨两页）；单页片等于 page
    page_end: int | None = None
    # 所属一级节，如「第十节 管理层讨论与分析」
    section: str | None = None
    # 报表口径：consolidated / parent / None（非报表或表里没印表题）
    scope: str | None = None
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)

    @property
    def where(self) -> str:
        """引用位置：图表标图像识别，表格按文件类型区分页码/表号。"""
        return format_where(
            kind=self.kind,
            page=self.page,
            page_end=self.page_end,
            seq=self.seq,
            filename=self.filename,
        )

    @property
    def label(self) -> str:
        """完整出处：位置 + 章节（有章节时）。"""
        return f"{self.where} · {self.section}" if self.section else self.where


def search_knowledge(
    query: str,
    *,
    file_ids: list[uuid.UUID] | None = None,
    top_k: int = 3,
) -> list[Hit]:
    """知识库检索主入口。检索词向量化失败会抛异常，由调用方转成友好文案。

    `file_ids` 为空表示搜全库——检索范围完全由调用方决定（工具层传入模型
    指定的 files，或会话级 kb_scope），本层不再根据查询内容自动推断文件范围。
    这样文件范围由模型/用户显式圈定，检索行为透明、可解释，不会出现「系统
    替你选了哪几份」的隐式预检索；范围为空时退化为搜全库，安全但不偷偷猜。
    """
    embedding = embed_query(query)
    return _core_search(query, embedding, file_ids=file_ids, top_k=top_k)


def preview_search(
    query: str,
    *,
    file_ids: list[uuid.UUID] | None,
    top_k: int = 5,
) -> list[Hit]:
    """知识库页的检索演示。

    ``file_ids`` 为空表示全库。这里不走 ``search_knowledge`` 的文件名收窄：
    演示要的是「没勾选 = 全部」，不能再被标签推断缩成一部分文件。
    """
    embedding = embed_query(query)
    hits = _core_search(query, embedding, file_ids=file_ids, top_k=top_k)
    limit = settings.kb_search_max_distance
    return [hit for hit in hits if hit.distance is None or hit.distance <= limit]


def _core_search(
    query: str,
    embedding: list[float],
    *,
    file_ids: list[uuid.UUID] | None,
    top_k: int,
) -> list[Hit]:
    """真正的一轮检索：召回 → 融合 → 去冗余 → 重排 → 邻接。

    embedding 由调用方传入：范围推导会搜两次（收窄 + 兜底），共用一份
    查询向量，不必为同一条 query 付两次向量化。
    """
    candidates = max(top_k, settings.kb_search_candidates)
    rows = store.search_kb_chunks(
        embedding,
        file_ids=file_ids,
        top_k=candidates,
        max_distance=settings.kb_search_max_distance,
    )
    rows = _fuse_with_keywords(query, rows, file_ids)
    if not rows:
        return []
    # 口径偏好放在去冗余与重排之前：它改的是「谁先被看到」，重排是从
    # 这 12 条候选里挑，候选顺序本身会影响模型的注意力
    rows = _apply_scope_preference(query, rows)

    kept = _dedupe(rows)
    if settings.kb_rerank_enabled and len(kept) > top_k:
        # 只把前 _RERANK_LIMIT 条交给模型：候选再多，提示词变长、延迟也变高，
        # 而排在第 13 名之外的本来也进不了 top_k
        head = kept[:_RERANK_LIMIT]
        kept = _rerank(query, head, top_k) + kept[_RERANK_LIMIT:]
    hits = [_to_hit(row) for row in kept[:top_k]]
    return _attach_neighbors(hits)


def _extract_keywords(query: str) -> list[str]:
    """从查询里提取关键词：jieba TF-IDF 打头的中文词 + 英文/数字 token。

    后者单独补一轮：jieba 会把 "OT-01"、"L3" 这类编号切碎或丢掉，
    而它们恰恰是向量最容易丢、子串匹配最准的东西。
    """
    keywords: list[str] = []
    try:
        import jieba.analyse

        keywords.extend(
            jieba.analyse.extract_tags(query, topK=settings.kb_hybrid_keywords)
        )
    except Exception:
        # jieba 加载失败不该拖垮检索，退化为纯向量
        logger.warning("jieba 关键词提取失败，退回纯向量", exc_info=True)
        return []
    keywords.extend(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_.]{1,}", query))
    unique: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        kw = kw.strip()
        if len(kw) < 2 or kw.lower() in _STOPWORDS or kw.lower() in seen:
            continue
        seen.add(kw.lower())
        unique.append(kw)
        if len(unique) >= settings.kb_hybrid_keywords:
            break
    return unique


def resolve_files_by_name(names: list[str]) -> list[uuid.UUID] | None:
    """把模型给的文件名/简称解析成就绪文件 id。

    不要求与库里文件名一字不差：精确全名、去扩展名、去标点子串、主体简称
    都能命中。解析失败返回 None，调用方应退回会话范围或全库，不要报错。
    一个提示词命中同一家公司的多份报告时全部纳入。命中了全部就绪文件则
    等于没圈范围，也返回 None。
    """
    hints = [_clean_filename_hint(name) for name in names]
    hints = [h for h in hints if h]
    if not hints:
        return None
    try:
        files = store.list_kb_files()
    except Exception:
        logger.warning("按文件名解析范围失败，跳过", exc_info=True)
        return None
    ready = [item for item in files if item.get("status") == "ready"]
    if not ready:
        return None
    matched: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for hint in hints:
        hint_hit = False
        for item in ready:
            fid = item["file_id"]
            if fid in seen:
                continue
            if _filename_hint_matches(hint, item.get("filename") or ""):
                seen.add(fid)
                matched.append(fid)
                hint_hit = True
        # 文件名没命中这个提示词，再按元数据维度（subject 列 + 别名）匹配：
        # 多主体文档（行业研报）文件名里没有「宁德时代」，但它确实是正文主体，
        # 只有走 subject/别名才能把这种提示词路由到对应文件。
        if not hint_hit:
            for item in ready:
                fid = item["file_id"]
                if fid in seen:
                    continue
                if _subject_token_matches(
                    hint,
                    item.get("subject") or "",
                    item.get("subject_aliases") or [],
                ):
                    seen.add(fid)
                    matched.append(fid)
                    hint_hit = True
    if not matched or len(matched) == len(ready):
        return None
    return matched


def _subject_token_matches(tok: str, subject: str, aliases: list[str]) -> bool:
    """提示词里的主体 token（可简称）是否命中文件级 subject 或任一别名。

    复用 _query_mentions 的最长公共子串判据，所以「中国能建」既能命中
    subject「中国能源建设股份有限公司」，也能命中别名「中国能建」；反过来
    「季度报告」这类与任何主体名无公共子串的噪音不会误命中。
    """
    candidates = [subject] + list(aliases or [])
    for cand in candidates:
        if not cand:
            continue
        if _query_mentions(tok, cand):
            return True
    return False


def _clean_filename_hint(name: str) -> str:
    """模型可能把目录行整段贴进来（含「｜标签：…」），只留文件名部分。"""
    text = str(name or "").strip()
    if not text:
        return ""
    for sep in ("｜", " | ", "|"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    return text


def _filename_hint_matches(hint: str, filename: str) -> bool:
    if not hint or not filename:
        return False
    hint_l = hint.lower().strip()
    name_l = filename.lower().strip()
    if hint_l == name_l:
        return True
    hint_stem = _EXT_RE.sub("", hint_l)
    name_stem = _EXT_RE.sub("", name_l)
    if hint_stem == name_stem:
        return True
    hint_norm = _NAME_PUNCT.sub("", hint_stem)
    name_norm = _NAME_PUNCT.sub("", name_stem)
    if hint_norm and name_norm:
        if hint_norm == name_norm:
            return True
        if len(hint_norm) >= 4 and (
            hint_norm in name_norm or name_norm in hint_norm
        ):
            return True
    return _query_mentions(hint, _file_subject(filename))


def _file_subject(filename: str) -> str:
    """文件名去掉年份、报告类型、扩展名后的主体。

    「顺丰控股股份有限公司2026 年第三季度报告」→「顺丰控股股份有限公司」。
    年份和「季度报告」这类字样每份文件都有，留着只会制造假重合。
    """
    stem = _EXT_RE.sub("", filename or "")
    stem = _FILENAME_NOISE.sub("", stem)
    return _NAME_PUNCT.sub("", stem).lower()


def _subject_core(subject: str) -> str:
    """剥掉「股份有限公司」这类组织形式后缀，留下真正的主体名。"""
    return _ENTITY_SUFFIX.sub("", subject)


def query_looks_on_topic(
    query: str, file_ids: list[uuid.UUID] | None = None
) -> bool:
    """问句是否点名了库里某份文件的主体或标签。

    用来挡「深圳带伞」这类出行问题：字面能撞上「深圳市顺丰投资」，
    但问句并没有在问顺丰。点名了公司/标签才算在查库。
    """
    probe = (query or "").strip()
    if not probe:
        return False
    try:
        files = store.list_kb_files()
    except Exception:
        logger.warning("kb 主题判断拿文件列表失败", exc_info=True)
        return False
    ready = [item for item in files if item.get("status") == "ready"]
    if file_ids:
        want = {str(fid) for fid in file_ids}
        ready = [item for item in ready if str(item.get("file_id")) in want]
    for item in ready:
        if _query_mentions(probe, _file_subject(item.get("filename") or "")):
            return True
        for tag in item.get("tags") or []:
            text = str(tag).strip()
            if len(text) >= 2 and text in probe:
                return True
    return False


def _query_mentions(query: str, subject: str) -> bool:
    """查询里是否提到了这个主体（允许简称）。

    判据是最长公共子串而不是整体相似度：query 里除主体名外还带着一长串
    提问内容，"中国能建…同比是增长还是下降"与全称的整体重合率会被稀释到
    0.15 以下，但最长公共子串始终是「中国能」这 3 个字，足以判定。
    """
    core = _subject_core(subject)
    if len(core) < 2:
        return False
    text = _NAME_PUNCT.sub("", (query or "").lower())
    if not text:
        return False
    match = difflib.SequenceMatcher(None, text, core).find_longest_match(
        0, len(text), 0, len(core)
    )
    return (
        match.size >= _SUBJECT_MIN_COMMON
        and match.size >= len(core) * _SUBJECT_MIN_RATIO
    )


def _fuse_with_keywords(
    query: str,
    vector_rows: list[dict],
    file_ids: list[uuid.UUID] | None,
) -> list[dict]:
    """混合检索：向量召回 + 关键词召回做 RRF 融合。

    两路各按名次贡献 1/(60+名次) 的分数，都命中的片段叠加后天然排前。
    关键词路失败/为空时退回纯向量，绝不把主检索拖挂。
    """
    if not settings.kb_hybrid_enabled:
        return vector_rows
    keywords = _extract_keywords(query)
    if not keywords:
        return vector_rows
    keywords = _keep_distinctive_keywords(keywords, file_ids)
    if not keywords:
        return vector_rows
    try:
        keyword_rows = store.search_kb_chunks_keyword(
            keywords,
            file_ids=file_ids,
            top_k=settings.kb_hybrid_candidates,
        )
    except Exception:
        logger.warning("kb 关键词召回失败，退回纯向量", exc_info=True)
        return vector_rows
    if not keyword_rows:
        return vector_rows
    return _rrf_merge(vector_rows, keyword_rows)


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


def _rrf_merge(vector_rows: list[dict], keyword_rows: list[dict]) -> list[dict]:
    """两路召回按名次做 RRF 融合：score = Σ 权重/(60+名次)，双路命中叠加。

    关键词路带 `_RRF_KEYWORD_WEIGHT` 的折扣：它只数「命中了几个词」，
    不像向量路衡量语义相关性，同权会盖过真正带数据的切片。

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
        scores[key] = scores.get(key, 0.0) + _RRF_KEYWORD_WEIGHT / (_RRF_K + rank + 1)
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


def _dedupe(rows: list[dict]) -> list[dict]:
    """按向量顺序去冗余：与任一已保留片段三字组重合度过高的直接丢。"""
    kept: list[dict] = []
    kept_grams: list[set[str]] = []
    for row in rows:
        grams = _trigrams(row.get("text") or "")
        if any(_jaccard(grams, other) >= _DUP_THRESHOLD for other in kept_grams):
            continue
        kept.append(row)
        kept_grams.append(grams)
    return kept


def _to_hit(row: dict) -> Hit:
    return Hit(
        file_id=str(row["file_id"]),
        filename=row.get("filename") or "",
        seq=int(row["seq"]),
        page=row.get("page"),
        kind=row.get("kind") or "text",
        text=render_otsl(row.get("text") or ""),
        distance=_vector_distance(row),
        page_end=row.get("page_end"),
        section=row.get("section"),
        scope=row.get("scope"),
    )


def _vector_distance(row: dict) -> float | None:
    raw = row.get("distance")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _attach_neighbors(hits: list[Hit]) -> list[Hit]:
    radius = settings.kb_neighbor_expand
    if radius <= 0 or not hits:
        return hits
    try:
        pairs = [(uuid.UUID(hit.file_id), hit.seq) for hit in hits]
        grouped = store.kb_chunk_neighbors(pairs, radius=radius)
    except Exception:
        # 邻接只是锦上添花，取不到就按原样返回，不能把整个检索拖挂
        logger.warning("kb 邻接片段读取失败，跳过上下文扩展", exc_info=True)
        return hits
    for hit in hits:
        before: list[tuple[int, str]] = []
        after: list[tuple[int, str]] = []
        for delta in range(1, radius + 1):
            prev = grouped.get((hit.file_id, hit.seq - delta))
            if prev:
                before.append((hit.seq - delta, render_otsl(prev[0]["text"])))
            nxt = grouped.get((hit.file_id, hit.seq + delta))
            if nxt:
                after.append((hit.seq + delta, render_otsl(nxt[0]["text"])))
        hit.before = [text for _, text in sorted(before)]
        hit.after = [text for _, text in sorted(after)]
    return hits


_rerank_cache: dict[tuple, list[int]] = {}


def _rerank(query: str, rows: list[dict], top_k: int) -> list[dict]:
    """LLM 重排：只输出候选编号顺序，解析失败就退回向量顺序。

    同 (query, 候选序列) 直接复用首次重排结果：LLM 编号输出非确定，缓存可
    消除「同问不同序」，也省一次重排调用。候选序列用 file_id#seq 签名，
    重排候选变了（范围/阈值不同）即失效，不会串味。
    """
    cache_key = (query, tuple((r.get("file_id"), r.get("seq")) for r in rows))
    cached = _rerank_cache.get(cache_key)
    if cached is None:
        listing = "\n\n".join(
            f"[{i + 1}] {row.get('filename') or ''} · {_where(row)}\n{_clip(render_otsl(row.get('text') or ''))}"
            for i, row in enumerate(rows)
        )
        prompt = build_rerank_user(query, listing, min(top_k, len(rows)))
        try:
            answer = complete_chat_once(prompt)
        except Exception:
            logger.warning("kb 重排调用失败，退回向量顺序", exc_info=True)
            return rows
        order = _parse_order(answer, len(rows))
        if not order:
            return rows
        _rerank_cache[cache_key] = order
    else:
        order = cached
    # 重排结果里没提到的候选按原顺序补在后面：宁可多给一条，也别丢候选
    ordered = [rows[i - 1] for i in order]
    ordered.extend(row for i, row in enumerate(rows, start=1) if i not in set(order))
    return ordered


def complete_chat_once(prompt: str) -> str:
    """单独抽出便于测试打桩：重排是短任务，模型与超时都独立配置。

    enable_thinking 必须关：主模型带思考链，20 秒的预算全花在推理上，
    实测直接超时；重排只需要输出一组编号，不需要想。
    """
    from ai_agent.llm.client import complete_chat

    return complete_chat(
        [{"role": "user", "content": prompt}],
        model=settings.kb_rerank_model or None,
        timeout=settings.kb_rerank_timeout_seconds,
        max_retries=0,
        max_tokens=120,
        # 重排要的是可复现的排序，不是措辞多样性
        temperature=settings.kb_rerank_temperature,
        extra_body={
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def _parse_order(answer: str, size: int) -> list[int]:
    match = _JSON_ARRAY.search(answer or "")
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    order: list[int] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, int) and 1 <= item <= size and item not in order:
            order.append(item)
    return order


def _where(row: dict) -> str:
    return format_where(
        kind=row.get("kind") or "text",
        page=row.get("page"),
        page_end=row.get("page_end"),
        seq=int(row.get("seq") or 0),
        filename=row.get("filename") or "",
        mime=row.get("mime") or "",
    )


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _RERANK_CLIP else f"{text[:_RERANK_CLIP]}…"


def _trigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i : i + 3] for i in range(len(compact) - 2)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0
