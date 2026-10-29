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

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.parsing import FIGURE, TABLE
from ai_agent.memory.embeddings import embed_query
from ai_agent.memory.store import store

logger = get_logger(__name__)

# 与已保留片段的三字组 Jaccard 超过这个值就当成重复（overlap 造成的近似片）
_DUP_THRESHOLD = 0.72
# 送重排时单段文本截断长度：重排只看相关性，不需要读完整段
_RERANK_CLIP = 300
# 送重排的候选上限：再多提示词就太长，而且第 13 名之后本来就进不了 top_k
_RERANK_LIMIT = 12
# RRF 融合平滑项（业界惯用 60）：排名越靠前贡献越大，两路都命中会叠加
_RRF_K = 60

# 查询里的功能词/疑问词，几乎每条查询都有、对召回没有区分度
_STOPWORDS = {
    "怎么", "如何", "什么", "哪些", "多少", "是否", "请问", "查询",
    "查找", "检索", "相关", "关于", "介绍", "说明", "可以", "需要",
    "应该", "如果", "我们", "他们", "以及", "还有", "一下", "这些",
    "文档", "文件", "知识库", "资料",
}

_JSON_ARRAY = re.compile(r"\[[^\[\]]*\]")


@dataclass
class Hit:
    """一条命中：命中片本体 + 前后邻接片段（拼上下文用）。"""

    file_id: str
    filename: str
    seq: int
    page: int | None
    kind: str
    text: str
    # 向量余弦距离；关键词独有命中没有走过向量路，保持 None。
    # 不能填 0：0 看起来像完美匹配，会把预检索闲聊门控打穿。
    distance: float | None
    # 跨页切片的末页（同节内聚合会一片跨两页）；单页片等于 page
    page_end: int | None = None
    # 所属一级节，如「第十节 管理层讨论与分析」
    section: str | None = None
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)

    @property
    def where(self) -> str:
        """引用位置：表格用表号、图表标图像识别、纯文本用页码/页码区间/段号。"""
        if self.kind == TABLE:
            return f"表{self.page}" if self.page is not None else "表格"
        if self.kind == FIGURE:
            return f"第{self.page}页图表" if self.page is not None else "图表"
        if self.page is None:
            return f"第{self.seq + 1}段"
        if self.page_end is not None and self.page_end != self.page:
            return f"第{self.page}–{self.page_end}页"
        return f"第{self.page}页"

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
    """知识库检索主入口。检索词向量化失败会抛异常，由调用方转成友好文案。"""
    embedding = embed_query(query)
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


def tag_file_scope(query: str) -> list[uuid.UUID] | None:
    """按查询关键词与文件标签的匹配推导一个候选文件范围。

    「先圈定范围再搜」的廉价实现：复用关键词提取（jieba + 编号 token，
    没有新增 LLM 调用），关键词与标签互为子串即算命中（"考勤" 命中
    "考勤制度"）。返回 None 表示不值得收窄：没提取到关键词、库里有
    文件没打标签、或全员命中（没有区分度）。调用方必须自带弱召回兜底
    ——标签判窄了不能变成漏召回。
    """
    if not settings.kb_tag_scope_enabled:
        return None
    keywords = _extract_keywords(query)
    if not keywords:
        return None
    try:
        files = store.list_kb_files()
    except Exception:
        logger.warning("kb 标签范围推导拿文件列表失败，跳过预过滤", exc_info=True)
        return None
    # 只有 ready 的文件才有切片可检索
    candidates = [f for f in files if f.get("status") == "ready"]
    return _tag_scope_ids(keywords, candidates)


def _tag_scope_ids(
    keywords: list[str], files: list[dict]
) -> list[uuid.UUID] | None:
    """纯函数：关键词 × 标签双向子串匹配，命中部分文件才收窄。"""
    matched: list[uuid.UUID] = []
    for item in files:
        tags = [str(t).strip().lower() for t in (item.get("tags") or []) if str(t).strip()]
        if not tags:
            return None  # 有文件没标签，tag 语义上覆盖不了它，收窄会漏
        hit = any(kw in tag or tag in kw for kw in keywords for tag in tags)
        if hit:
            matched.append(item["file_id"])
    if not matched or len(matched) == len(files):
        return None
    return matched


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


def _rrf_merge(vector_rows: list[dict], keyword_rows: list[dict]) -> list[dict]:
    """两路召回按名次做 RRF 融合：score = Σ 1/(60+名次)，双路命中叠加。

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
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
    ranked = sorted(merged.items(), key=lambda item: scores[item[0]], reverse=True)
    fused = [row for _, row in ranked]
    logger.info(
        "kb 混合检索 向量 %d 条 + 关键词 %d 条 → 融合 %d 条",
        len(vector_rows), len(keyword_rows), len(fused),
    )
    return fused


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
        text=row.get("text") or "",
        distance=_vector_distance(row),
        page_end=row.get("page_end"),
        section=row.get("section"),
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
                before.append((hit.seq - delta, prev[0]["text"]))
            nxt = grouped.get((hit.file_id, hit.seq + delta))
            if nxt:
                after.append((hit.seq + delta, nxt[0]["text"]))
        hit.before = [text for _, text in sorted(before)]
        hit.after = [text for _, text in sorted(after)]
    return hits


def _rerank(query: str, rows: list[dict], top_k: int) -> list[dict]:
    """LLM 重排：只输出候选编号顺序，解析失败就退回向量顺序。"""
    listing = "\n\n".join(
        f"[{i + 1}] {row.get('filename') or ''} · {_where(row)}\n{_clip(row.get('text') or '')}"
        for i, row in enumerate(rows)
    )
    prompt = (
        "你在做检索结果重排。下面是知识库候选片段，每个用编号标注。"
        "请按「能否直接回答用户问题」从高到低排序。\n\n"
        f"用户问题：{query}\n\n候选：\n{listing}\n\n"
        f"只输出一个 JSON 数组，元素是候选编号，例如 [3,1,2]；"
        f"最多输出 {min(top_k, len(rows))} 个编号，不要输出任何解释。"
    )
    try:
        answer = complete_chat_once(prompt)
    except Exception:
        logger.warning("kb 重排调用失败，退回向量顺序", exc_info=True)
        return rows
    order = _parse_order(answer, len(rows))
    if not order:
        return rows
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
    kind = row.get("kind") or "text"
    page = row.get("page")
    if kind == TABLE:
        return f"表{page}" if page is not None else "表格"
    if kind == FIGURE:
        return f"第{page}页图表" if page is not None else "图表"
    return f"第{page}页" if page is not None else f"第{int(row.get('seq') or 0) + 1}段"


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
