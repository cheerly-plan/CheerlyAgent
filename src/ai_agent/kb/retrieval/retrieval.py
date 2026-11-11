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

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.ingest.chunking import TABLE_GROUP_CHARS, html_table_plain
from ai_agent.kb.ingest.lexicon import load_short_words, read_extra
from ai_agent.kb.ingest.meta import name_in_text
from ai_agent.kb.ingest.meta import ENTITY_SUFFIX as _ENTITY_SUFFIX
from ai_agent.kb.ingest.meta import EXT_RE as _EXT_RE
from ai_agent.kb.ingest.meta import FILENAME_NOISE as _FILENAME_NOISE
from ai_agent.kb.ingest.meta import NAME_PUNCT as _NAME_PUNCT
from ai_agent.kb.ingest.otsl import render_otsl
from ai_agent.kb.ingest.parsing import FIGURE, TABLE
from ai_agent.kb.ingest.tables import SCOPE_CONSOLIDATED, SCOPE_PARENT
from ai_agent.storage.embeddings import embed_query
from ai_agent.storage.store import store
from ai_agent.prompts.kb import build_rerank_user
from ai_agent.kb.retrieval.rerank import get_reranker

logger = get_logger(__name__)

# 与已保留片段的三字组 Jaccard 超过这个值就当成重复（overlap 造成的近似片）
_DUP_THRESHOLD = 0.72
# 送重排的正文上限，与 HTML 表行组是同一个字数：一组表重排能看完，
# 不会停在表头。正文切片本来就短于这个上限，放宽后整段也能读完。
_RERANK_CLIP = TABLE_GROUP_CHARS
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
    # 文件报告期，如 2025Q3。正文表格常常不写年份，回答时靠它区分两年。
    period: str = ""
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
    filters: dict | None = None,
    per_file: bool = False,
    diversify_query: str | None = None,
) -> list[Hit]:
    """知识库检索主入口。检索词向量化失败会抛异常，由调用方转成友好文案。

    `file_ids` 为空表示搜全库——检索范围完全由调用方决定（工具层传入模型
    指定的 files，或会话级 kb_scope），本层不再根据查询内容自动推断文件范围。
    `filters`（可选）是 chunk 元数据过滤（kinds/subjects/periods/scopes/
    sections），由调用方（工具层）构造后传入：scope 从 query 意图推断、
    kinds/sections 由模型显式指定。带过滤召回为空时自动回退「不过滤」再查
    一次，避免过滤过窄直接挂空。
    """
    embedding = embed_query(query)
    hits = _core_search(
        query,
        embedding,
        file_ids=file_ids,
        top_k=top_k,
        filters=filters,
        per_file=per_file,
        diversify_query=diversify_query,
    )
    if not hits and filters:
        # 元数据过滤后召回为空，多半是约束过窄（如 scope 强约束把母公司切片
        # 全排除了），回退不过滤再查一次，保证至少有结果而非直接挂空
        logger.info("kb 元数据过滤后召回为空，回退不过滤重查: %s", filters)
        hits = _core_search(
            query,
            embedding,
            file_ids=file_ids,
            top_k=top_k,
            filters=None,
            per_file=per_file,
            diversify_query=diversify_query,
        )
    return hits


def search_knowledge_multi(
    queries: list[str],
    *,
    file_ids: list[uuid.UUID] | None = None,
    top_k: int = 3,
    filters: dict | None = None,
    per_file: bool = False,
    diversify_query: str | None = None,
) -> list[Hit]:
    """多查询检索并融合（multi-query RRF）。

    把多条改写变体分别检索后，按 (file_id, seq) 去重并跨查询累加 RRF 分数
    （score = Σ 1/(60+rank)），再统一重排、截断、邻接。单条查询时退化为
    search_knowledge，保证行为一致。任一变体向量化失败仅跳过该变体。
    """
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        return []
    if len(queries) == 1:
        return search_knowledge(
            queries[0],
            file_ids=file_ids,
            top_k=top_k,
            filters=filters,
            per_file=per_file,
            diversify_query=diversify_query,
        )
    merged: dict[tuple[str, int], dict] = {}
    scores: dict[tuple[str, int], float] = {}
    for q in queries:
        try:
            embedding = embed_query(q)
        except Exception:
            logger.exception("multi-query 向量化失败，跳过变体：%s", q[:60])
            continue
        rows = _retrieve_candidates(
            q, embedding, file_ids=file_ids, top_k=top_k, filters=filters
        )
        if not rows and filters:
            rows = _retrieve_candidates(
                q, embedding, file_ids=file_ids, top_k=top_k, filters=None
            )
        for rank, row in enumerate(rows):
            key = (str(row["file_id"]), int(row["seq"]))
            if key not in merged:
                merged[key] = dict(row)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
    if not merged:
        return []
    ranked = sorted(
        merged.values(),
        key=lambda r: scores[(str(r["file_id"]), int(r["seq"]))],
        reverse=True,
    )
    if settings.kb_rerank_enabled and len(ranked) > top_k:
        # 重排用主查询（queries[0]，即改写后的原始问句）做排序
        head = ranked[:_RERANK_LIMIT]
        ranked = _rerank(queries[0], head, top_k) + ranked[_RERANK_LIMIT:]
    ranked = _diversify(
        diversify_query or queries[0], ranked, top_k, per_file=per_file
    )
    ranked = _merge_same_table_rows(ranked)
    hits = [_to_hit(row) for row in ranked[:top_k]]
    return _attach_neighbors(hits)


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


def _retrieve_candidates(
    query: str,
    embedding: list[float],
    *,
    file_ids: list[uuid.UUID] | None,
    top_k: int,
    filters: dict | None = None,
) -> list[dict]:
    """一轮检索的候选召回：召回 → 关键词融合 → 口径偏好 → 去重（不含重排/截断/邻接）。

    供 search_knowledge（单 query）与 search_knowledge_multi（多 query 融合）共用。
    """
    candidates = max(top_k, settings.kb_search_candidates)
    rows = store.search_kb_chunks(
        embedding,
        file_ids=file_ids,
        top_k=candidates,
        max_distance=settings.kb_search_max_distance,
        **(filters or {}),
    )
    rows = _fuse_with_keywords(query, rows, file_ids, filters=filters)
    if not rows:
        return []
    # 口径偏好放在去冗余与重排之前：它改的是「谁先被看到」，重排是从
    # 这 12 条候选里挑，候选顺序本身会影响模型的注意力
    rows = _apply_scope_preference(query, rows)
    return _dedupe(rows)


def _core_search(
    query: str,
    embedding: list[float],
    *,
    file_ids: list[uuid.UUID] | None,
    top_k: int,
    filters: dict | None = None,
    per_file: bool = False,
    diversify_query: str | None = None,
) -> list[Hit]:
    """真正的一轮检索：候选召回 → 重排 → 邻接。"""
    rows = _retrieve_candidates(
        query, embedding, file_ids=file_ids, top_k=top_k, filters=filters
    )
    if not rows:
        return []
    if settings.kb_rerank_enabled and len(rows) > top_k:
        # 只把前 _RERANK_LIMIT 条交给模型：候选再多，提示词变长、延迟也变高，
        # 而排在第 13 名之外的本来也进不了 top_k
        head = rows[:_RERANK_LIMIT]
        rows = _rerank(query, head, top_k) + rows[_RERANK_LIMIT:]
    rows = _diversify(
        diversify_query or query, rows, top_k, per_file=per_file
    )
    rows = _merge_same_table_rows(rows)
    hits = [_to_hit(row) for row in rows[:top_k]]
    return _attach_neighbors(hits)


def _extract_keywords(query: str) -> list[str]:
    """从查询里提取关键词：jieba TF-IDF 打头的中文词 + 英文/数字 token。

    后者单独补一轮：jieba 会把 "OT-01"、"L3" 这类编号切碎或丢掉，
    而它们恰恰是向量最容易丢、子串匹配最准的东西。
    """
    keywords: list[str] = []
    load_short_words()
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


# 相邻两个二字词才拼。单字（「前」）不参与，避免「前」+「三季度」变成
# 「前三季度」，对不上正文里的「第三季度」。两边都是二字，加起来正好四字。
_SHORT_WORD = re.compile(r"^[\u4e00-\u9fff]{2}$")


def _adjacent_short_phrases(query: str) -> list[tuple[str, str, str]]:
    """问句里相邻的两个二字词，拼成一个不超过四字的短词。

    返回 (短词, 左词, 右词)。已经是一个词的（「净利润」「注册资本」，以及
    短词表里的词）不会被拆开再拼。单字和停用词不参与。
    """
    load_short_words()
    try:
        import jieba

        tokens = [token.strip() for token in jieba.cut(query or "") if token.strip()]
    except Exception:
        logger.warning("jieba 短词切分失败，跳过短词", exc_info=True)
        return []
    found: list[tuple[str, str, str]] = []
    index = 0
    while index < len(tokens) - 1:
        left, right = tokens[index], tokens[index + 1]
        if (
            _SHORT_WORD.fullmatch(left)
            and _SHORT_WORD.fullmatch(right)
            and left not in _STOPWORDS
            and right not in _STOPWORDS
            and len(left) + len(right) <= 4
        ):
            found.append((left + right, left, right))
            index += 2
            continue
        index += 1
    return found


def _apply_short_phrases(
    query: str,
    keywords: list[str],
    file_ids: list[uuid.UUID] | None,
) -> list[str]:
    """两个二字在库里以词的形式出现过，就用整段替换拆开的两半。

    后面还紧跟汉字的不算出现过：「合并利润」只是「合并利润表」的前缀，
    仍退回「合并」和「利润」。一个边界命中都没有、过半切片都含有、
    或写在排除名单里的，都不替换。词频取不到时也不替换。
    """
    pairs = _adjacent_short_phrases(query)
    if not pairs:
        return keywords
    try:
        total, freq = store.keyword_boundary_freq(
            [phrase for phrase, _, _ in pairs], file_ids=file_ids
        )
    except Exception:
        logger.warning("kb 短词词频统计失败，保留拆开的词", exc_info=True)
        return keywords
    if total <= 0:
        return keywords
    blocked = read_extra()[1]
    chosen: list[str] = []
    drop: set[str] = set()
    for phrase, left, right in pairs:
        if phrase in blocked:
            continue
        hits = freq.get(phrase, 0)
        if hits <= 0 or hits * 2 > total:
            continue
        chosen.append(phrase)
        drop.update((left, right))
    if not chosen:
        return keywords
    logger.info("kb 关键词改用短词：%s", chosen)
    merged: list[str] = []
    seen: set[str] = set()
    for keyword in [*chosen, *[word for word in keywords if word not in drop]]:
        key = keyword.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(keyword)
        if len(merged) >= settings.kb_hybrid_keywords:
            break
    return merged


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
        # 「中科飞测」不在别名表里，但是全称里独有的一段，同样算点名。
        if not hint_hit:
            from ai_agent.kb.ingest.meta import fragment_subjects

            catalog = [
                (str(item.get("subject") or ""), list(item.get("subject_aliases") or []))
                for item in ready
            ]
            fragments = fragment_subjects(hint, catalog)
            for item in ready:
                fid = item["file_id"]
                if fid in seen:
                    continue
                subject = str(item.get("subject") or "")
                if subject in fragments or _subject_token_matches(
                    hint,
                    subject,
                    item.get("subject_aliases") or [],
                ):
                    seen.add(fid)
                    matched.append(fid)
                    hint_hit = True
    if not matched:
        return []  # 点名但一个都解析不到 → 上层按「未找到」处理
    # 仅在就绪文件多于 1 份时，「命中覆盖全部」才等于没缩小范围；
    # 单文件库（ready==1）命中那唯一一份就是成功圈定，不可短路。
    if len(ready) > 1 and len(matched) == len(ready):
        return None  # 点名无缩小意义 → 上层退回全库
    return matched


def _subject_token_matches(tok: str, subject: str, aliases: list[str]) -> bool:
    """提示词里的主体是否整段出现在文件 subject 或别名里。

    「中国能建」只在别名表有这四个字时命中，不再和全称做最长公共子串。
    """
    for cand in [subject, *list(aliases or [])]:
        if cand and name_in_text(tok, cand):
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
    return name_in_text(hint, _file_subject(filename))


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
        if name_in_text(probe, _file_subject(item.get("filename") or "")):
            return True
        for tag in item.get("tags") or []:
            text = str(tag).strip()
            if len(text) >= 2 and text in probe:
                return True
    return False


def mentioned_subjects(
    query: str, file_ids: list[uuid.UUID] | None = None
) -> set[str]:
    """query 里点名的文件主体（subject 核心名）集合。

    主体、别名、文件名去噪后的名字整段出现才算。别名表没有的简称，
    只要是某一家主体名里独有的一段（「中科飞测」），也算点名——
    对话靠这个决定要不要强制调用知识库工具。
    """
    text = (query or "").strip()
    if not text:
        return set()
    try:
        files = store.list_kb_files()
    except Exception:
        logger.warning("kb 主体提及检测拿文件列表失败", exc_info=True)
        return set()
    ready = [item for item in files if item.get("status") == "ready"]
    if file_ids:
        want = {str(fid) for fid in file_ids}
        ready = [item for item in ready if str(item.get("file_id")) in want]
    subjects: set[str] = set()
    for item in ready:
        names = [item.get("subject") or ""]
        names.extend(alias for alias in (item.get("subject_aliases") or []) if alias)
        names.append(_file_subject(item.get("filename") or ""))
        for name in names:
            if name and name_in_text(text, name):
                core = _subject_core(name)
                subjects.add(core or name)
                break
    from ai_agent.kb.ingest.meta import fragment_subjects

    catalog = [
        (str(item.get("subject") or ""), list(item.get("subject_aliases") or []))
        for item in ready
    ]
    for subject in fragment_subjects(text, catalog):
        core = _subject_core(subject)
        subjects.add(core or subject)
    return subjects


def mentioned_entities(
    query: str, file_ids: list[uuid.UUID] | None = None
) -> list[str]:
    """问句里按字面出现的图谱实体名或别名。

    文件主体的简称由 mentioned_subjects 负责。这里补人名、子公司、产品
    这类不在文件 subject 上的名字：实体名或别名整段出现在问句里才算命中。
    读库失败返回空，调用方退回「不强制检索」。
    """
    text = (query or "").strip()
    if len(text) < 2:
        return []
    try:
        names = store.query_mentions_entity(text, file_ids=file_ids)
    except Exception:
        logger.warning("kb 实体提及检测失败", exc_info=True)
        return []
    return [name for name in names if name]


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
    keywords = _extract_keywords(query)
    if not keywords:
        return vector_rows
    keywords = _apply_short_phrases(query, keywords, file_ids)
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
            **(filters or {}),
        )
    except Exception:
        logger.warning("kb 关键词召回失败，退回纯向量", exc_info=True)
        return vector_rows
    if not keyword_rows:
        return vector_rows
    return _rrf_merge(vector_rows, keyword_rows, n_keywords=len(keywords))


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
        return _diversify_by_file(query, rows, top_k)
    return _diversify_by_subject(rows, top_k)


def _focused_file_ids(query: str, rows: list[dict]) -> set[str]:
    """结果里被问句点名的文件。用文件名还原主体，不看切片上被改写过的主体。"""
    from ai_agent.kb.ingest.meta import extract_subject, fragment_subjects, name_in_text

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
        if name_in_text(query, subject):
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
        return _diversify_by_subject(rows, top_k)
    focus = _focused_file_ids(query, rows)
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
        grams = _trigrams(row.get("text") or "")
        prior = kept_grams.setdefault(fid, [])
        if any(_jaccard(grams, other) >= _DUP_THRESHOLD for other in prior):
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
        text=render_otsl(row.get("text") or ""),
        distance=_vector_distance(row),
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


# 同一张长表被切成多片时，命中常常落在几片不相邻的切片上（1 和 5）。
# 合并时把中间的空档一起补进来——模型要的数很可能正在空档里——但设上限，
# 免得一张表把整个上下文吃光。
_MERGE_MAX_GAP = 2        # 空档最多补 2 片
_MERGE_MAX_CHARS = 2000   # 合并后的正文上限


def _table_header(text: str) -> str:
    """表格切片的表头指纹：前 3 个非空行（标题 / 表名 / 列名）。"""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return "".join(re.sub(r"\s+", "", ln) for ln in lines[:3])


def _merge_same_table_rows(rows: list[dict]) -> list[dict]:
    """同一文件同一页、表头相同的多片表格合并成一条。

    长表每片都重印表头（刻意的，避免切片独立后失去列名），于是同表几片一起
    进 top_k，既浪费名额（模型只拿到碎片），又重复占上下文。

    命中不相邻时（seq 1 和 5）按 seq 顺序合并，**空档也补进来**：那是同一张
    表的中间行，答案可能就在里面。空档超过 ``_MERGE_MAX_GAP`` 片或合并后超过
    ``_MERGE_MAX_CHARS`` 字就放弃这次合并，保持原样——宁可多占名额，也不能
    把一张长表整块塞进上下文。
    """
    groups: dict[tuple[str, int, str], list[int]] = {}
    for index, row in enumerate(rows):
        if (row.get("kind") or "") != "table":
            continue
        header = _table_header(row.get("text") or "")
        if not header:
            continue
        key = (str(row.get("file_id")), int(row.get("page") or 0), header)
        groups.setdefault(key, []).append(index)

    drop: set[int] = set()
    merged_rows: dict[int, dict] = {}
    for (file_id, _page, _header), indexes in groups.items():
        if len(indexes) < 2:
            continue
        members = [rows[i] for i in indexes]
        seqs = sorted(int(m.get("seq") or 0) for m in members)
        gaps = [
            seq
            for seq in range(seqs[0], seqs[-1] + 1)
            if seq not in set(seqs)
        ]
        # 空档小就补进来（同表的中间行，答案可能就在里面）；
        # 空档大就只合并命中的那几片，中间用一行标注省略——既保住完整的表名
        # 与列名，也不至于为了一段推测的相关性把整张长表塞进上下文。
        extra: list[dict] = []
        if gaps and len(gaps) <= _MERGE_MAX_GAP:
            extra = _fetch_merged_gap(file_id, seqs, gaps)
        ordered = sorted([*members, *extra], key=lambda m: int(m.get("seq") or 0))
        text = _join_table_texts(
            [(m.get("text") or "", int(m.get("seq") or 0)) for m in ordered]
        )
        if len(text) > _MERGE_MAX_CHARS:
            continue
        best = min(indexes, key=lambda i: rows[i].get("distance") or 1.0)
        merged = dict(rows[best])
        merged["text"] = text
        merged["seq"] = seqs[0]
        merged["page_end"] = max(int(m.get("page") or 0) for m in members) or merged.get("page")
        distances = [m["distance"] for m in members if m.get("distance") is not None]
        if distances:
            merged["distance"] = min(distances)
        merged_rows[best] = merged
        drop.update(i for i in indexes if i != best)

    if not merged_rows and not drop:
        return rows
    out: list[dict] = []
    for index, row in enumerate(rows):
        if index in merged_rows:
            out.append(merged_rows[index])
        elif index in drop:
            continue
        else:
            out.append(row)
    return out


def _fetch_merged_gap(file_id: str, seqs: list[int], gaps: list[int]) -> list[dict]:
    """把空档里的切片取回来（同表中间行）。取不到就不补，不影响主流程。"""
    try:
        import uuid as _uuid

        pairs = [(_uuid.UUID(file_id), seq) for seq in seqs]
        grouped = store.kb_chunk_neighbors(pairs, radius=_MERGE_MAX_GAP + 1)
    except Exception:
        logger.warning("合并同表切片时读取空档失败，跳过补全", exc_info=True)
        return []
    found: list[dict] = []
    for seq in gaps:
        rows = grouped.get((file_id, seq)) or []
        if rows:
            found.append(dict(rows[0], file_id=file_id))
    return found


def _join_table_texts(segments: list[tuple[str, int]]) -> str:
    """按 seq 顺序拼接同表切片，重复表头只留第一份；空缺处标注省略。"""
    if not segments:
        return ""
    texts = [text for text, _seq in segments]
    seqs = [seq for _text, seq in segments]
    head_lines = [
        ln for ln in texts[0].splitlines() if ln.strip()
    ][:3]
    out = list(texts[0].splitlines())
    for index, text in enumerate(texts[1:], start=1):
        omitted = seqs[index] - seqs[index - 1] - 1
        if omitted > 0:
            out.append(f"…（同表另有 {omitted} 片未展开）…")
        lines = text.splitlines()
        # 跳过与前一片重复的表头行（标题 / 表名 / 列名 / 分隔线）
        start = 0
        head_set = {re.sub(r"\s+", "", ln) for ln in head_lines}
        while start < len(lines) and (
            not lines[start].strip()
            or re.sub(r"\s+", "", lines[start]) in head_set
            or re.fullmatch(r"\|[\s\-|]+\|", lines[start].strip() or "-")
        ):
            start += 1
        out.extend(lines[start:])
    return "\n".join(out).strip()


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


_RERANK_CACHE_MAX = 1024
_rerank_cache: dict[tuple, list[float]] = {}


def _candidate_text(row: dict) -> str:
    """喂给重排器的候选文本：文件名 + 位置 + 单元格文字。

    HTML 标签先去掉再截断。否则 800 字窗口全花在 `<td>` 上，表身还是看不见。
    """
    body = html_table_plain(render_otsl(row.get("text") or ""))
    return f"{row.get('filename') or ''} · {_where(row)}\n{_clip(body)}"


def _llm_rerank_scores(query: str, docs: list[str]) -> list[float]:
    """LLM 重排：让模型对候选编号排序，折算成分数（名次越靠前分越高）。

    仅作为 cross-encoder 不可用时的兜底；分数制与 cross-encoder 后端输出同构，
    `_rerank` 无需感知具体后端。
    """
    listing = "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(docs))
    prompt = build_rerank_user(query, listing, len(docs))
    try:
        answer = complete_chat_once(prompt)
    except Exception:
        logger.warning("kb LLM 重排调用失败", exc_info=True)
        raise
    order = _parse_order(answer, len(docs))
    if not order:
        raise ValueError("kb LLM 重排未解析出有效顺序")
    n = len(docs)
    scores = [0.0] * n
    for rank, idx in enumerate(order):
        if 1 <= idx <= n:
            scores[idx - 1] = float(n - rank)
    return scores


def _rerank(query: str, rows: list[dict], top_k: int) -> list[dict]:
    """重排：调用可插拔 Reranker 对候选打分，按分数降序返回。

    候选序列用 file_id#seq 签名，重排候选变了（范围/阈值不同）即失效，不会串味。
    后端由 settings.kb_rerank_backend 决定（见 ai_agent.kb.rerank）。失败或分数
    数量不符时退回向量顺序，绝不丢候选。
    """
    sig = tuple((r.get("file_id"), r.get("seq")) for r in rows)
    cache_key = (query, sig)
    scores = _rerank_cache.get(cache_key)
    if scores is None:
        docs = [_candidate_text(r) for r in rows]
        try:
            scores = get_reranker().rerank(query, docs)
        except Exception:
            logger.warning("kb 重排调用失败，退回向量顺序", exc_info=True)
            return rows
        if len(scores) != len(rows):
            logger.warning("kb 重排返回分数数量与候选不符，退回向量顺序")
            return rows
        _rerank_cache[cache_key] = scores
        if len(_rerank_cache) > _RERANK_CACHE_MAX:
            # dict 保插入序，弹出最旧一项，避免进程内无限增长（修复旧实现无淘汰）
            _rerank_cache.pop(next(iter(_rerank_cache)))
    adjusted = _boost_indicator_hits(query, rows, list(scores))
    order = sorted(range(len(adjusted)), key=lambda i: adjusted[i], reverse=True)
    return [rows[i] for i in order]


def _boost_indicator_hits(
    query: str, rows: list[dict], scores: list[float]
) -> list[float]:
    """问句里整段出现的行项目，给对应表格切片一点加分。

    只认指标名整段出现在问句里。「营收」不会因此顶上「营业收入」那一行。
    加分是分数的一成多，够把接近的两片换位，不够把明显不相关的片抬上来。
    """
    text = query or ""
    boosted: list[float] = []
    for row, score in zip(rows, scores):
        names = row.get("indicators") or []
        if not isinstance(names, list):
            names = []
        hit = any(
            isinstance(name, str) and len(name) >= 2 and name in text for name in names
        )
        boosted.append(score * 1.12 if hit else score)
    return boosted


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
