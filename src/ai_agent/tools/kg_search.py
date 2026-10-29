"""知识图谱检索工具。

与 kb_search 的分工：
- kb_search 是切片向量直查，命中「原文里出现过相关表述」的段落；
- kg_search 先匹配实体，再沿关系走一跳，能找回**原文没直接提到关键词、
  但通过关系相连**的内容（"长鑫存储" → 生产 → "DRAM" → 属于 → "存储芯片"）。

证据召回三路融合（GraphRAG）：
1. 种子实体本身的切片（关系最直接）；
2. 邻居实体的切片——关系把话题带到的"隔壁"内容，向量检索够不着；
3. 向量直查兜底——问题本身的语义相关片段，防图谱路径漏掉显式表述。

返回给模型的格式里显式带上关系路径，便于它在回答里说明依据。
引用走 citations 旁路（与 kb_search 相同），前端可点开原文。
"""

import logging
import uuid

from pydantic import BaseModel, Field

from ai_agent.core.config import settings
from ai_agent.kb.citations import Citation, add_citations
from ai_agent.kb.scope import kb_scope
from ai_agent.memory.embeddings import embed_query
from ai_agent.memory.store import store
from ai_agent.tools.base import Tool

logger = logging.getLogger(__name__)

# 邻居实体证据上限：图扩展最大 200 节点，全带证据会淹没正文；
# BFS 顺序即离种子由近到远，取前几个邻居够了
_MAX_NEIGHBOR_ENTITIES = 6
# 单次返回的证据切片总数
_MAX_EVIDENCE = 8
# 注入给模型的关系条数：再多挤掉原文，对话预检索与工具共用这个上限
_MAX_NEIGHBORHOOD_EDGES = 20


def format_neighborhood(
    seeds: list[dict],
    graph: dict,
    *,
    max_edges: int = _MAX_NEIGHBORHOOD_EDGES,
) -> list[str]:
    """命中实体 + 关系路径。预检索只注入这一段，不带原文（切片预检索已经有了）。"""
    name_by_id = {str(n["id"]): n["name"] for n in graph.get("nodes") or []}
    lines = [
        "命中实体："
        + "、".join(f"{s['name']}({s.get('type') or ''})" for s in seeds)
    ]
    edges = graph.get("edges") or []
    if edges:
        lines.append("关联关系：")
        for edge in edges[:max_edges]:
            src = name_by_id.get(str(edge["source_id"]), "?")
            dst = name_by_id.get(str(edge["target_id"]), "?")
            detail = f"：{edge['description']}" if edge.get("description") else ""
            lines.append(f"- {src} —{edge['relation']}→ {dst}{detail}")
    return lines


class KgSearchArgs(BaseModel):
    query: str = Field(
        ..., description="要检索的实体、主题或问题，用自然语言描述"
    )
    file_ids: list[str] | None = Field(
        default=None, description="限定检索的文件 id 列表，缺省检索全部文件"
    )
    depth: int = Field(default=1, ge=1, le=2, description="从命中实体向外扩展的跳数")
    top_k: int = Field(default=5, ge=1, le=10, description="参与扩展的种子实体数")


def _evidence_where(item: dict) -> str:
    """证据切片的引用位置（与 kb_search 的口径一致，含跨页区间与章节）。"""
    kind = item.get("kind") or "text"
    page = item.get("page")
    page_end = item.get("page_end")
    section = item.get("section")
    if kind == "figure":
        where = f"第{page}页图表" if page is not None else "图表"
    elif kind == "table":
        where = f"表{page}" if page is not None else "表格"
    elif page is None:
        where = f"第{int(item['chunk_seq']) + 1}段"
    elif page_end is not None and page_end != page:
        where = f"第{page}–{page_end}页"
    else:
        where = f"第{page}页"
    return f"{where} · {section}" if section else where


def _kg_search(
    query: str,
    file_ids: list[str] | None = None,
    depth: int = 1,
    top_k: int = 5,
) -> str:
    stats = store.graph_stats()
    if stats["entities"] == 0:
        return "知识图谱还没有数据（可能文件都在建图中或已关闭抽取），可以先改用 kb_search。"

    # 模型没显式指定文件时，跟随用户在输入框里选的会话范围（与 kb_search 同口径）
    raw_ids = file_ids if file_ids else list(kb_scope())
    parsed_ids: list | None = None
    if raw_ids:
        parsed_ids = []
        for raw in raw_ids:
            try:
                parsed_ids.append(uuid.UUID(raw))
            except ValueError:
                return f"ERROR: 非法的文件 id {raw}"

    try:
        embedding = embed_query(query)
    except Exception:
        return "ERROR: 向量化服务暂时不可用，请稍后重试"

    seeds = store.search_entities(
        embedding,
        query=query,
        top_k=top_k,
        file_ids=parsed_ids,
        min_score=settings.kb_graph_min_score,
    )
    if not seeds:
        return "图谱里没有找到相关实体，可以换个说法，或用 kb_search 直接检索原文。"

    graph = store.expand_subgraph(
        [s["id"] for s in seeds],
        depth=depth,
        max_nodes=settings.kb_graph_max_nodes,
    )
    lines: list[str] = format_neighborhood(seeds, graph)

    evidence = _collect_evidence(query, embedding, seeds, graph, parsed_ids)
    if evidence:
        lines.append("原文依据：")
        for item in evidence:
            lines.append(
                f"[{item['filename']} · {_evidence_where(item)}]\n{item['text']}"
            )
    # 引用旁路：图谱证据同样可以在前端点开核对（与 kb_search 同一套机制）
    add_citations(_citations_of(evidence))

    return "\n".join(lines)


def _collect_evidence(
    query: str,
    embedding: list[float],
    seeds: list[dict],
    graph: dict,
    file_ids: list | None,
) -> list[dict]:
    """三路证据融合，按 (file_id, chunk_seq) 去重，总量封顶。

    顺序即优先级：种子切片 → 邻居切片 → 向量直查。图扩展可能带回来
    一堆离题的远亲，所以邻居只取 BFS 前 _MAX_NEIGHBOR_ENTITIES 个、
    每个只给 1 条最常命中的切片。
    """
    evidence: list[dict] = []
    seen: set[tuple[str, int]] = set()

    def _add(items: list[dict]) -> None:
        for item in items:
            try:
                key = (str(item["file_id"]), int(item["chunk_seq"]))
            except (KeyError, TypeError, ValueError):
                continue
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)
            if len(evidence) >= _MAX_EVIDENCE:
                return

    grouped = store.get_entity_chunks([s["id"] for s in seeds], per_entity=2)
    for seed in seeds:
        _add(grouped.get(str(seed["id"]), []))
        if len(evidence) >= _MAX_EVIDENCE:
            return evidence

    seed_ids = {str(s["id"]) for s in seeds}
    neighbors = [
        node["id"] for node in graph["nodes"] if str(node["id"]) not in seed_ids
    ][:_MAX_NEIGHBOR_ENTITIES]
    if neighbors:
        neighbor_grouped = store.get_entity_chunks(neighbors, per_entity=1)
        for node_id in neighbors:
            _add(neighbor_grouped.get(str(node_id), []))
            if len(evidence) >= _MAX_EVIDENCE:
                return evidence

    # 向量直查兜底：图谱证据覆盖"关系近"，这一路补"问题本身相关"，
    # 复用上面已算好的 embedding，不额外调用向量化服务
    try:
        vec_rows = store.search_kb_chunks(
            embedding,
            file_ids=file_ids,
            top_k=2,
            max_distance=settings.kb_search_max_distance,
        )
    except Exception:
        logger.warning("kg_search 向量兜底召回失败，跳过", exc_info=True)
        vec_rows = []
    for row in vec_rows:
        if len(evidence) >= _MAX_EVIDENCE:
            break
        _add(
            [
                {
                    "file_id": row["file_id"],
                    "chunk_seq": int(row["seq"]),
                    "page": row.get("page"),
                    "page_end": row.get("page_end"),
                    "section": row.get("section"),
                    "kind": row.get("kind") or "text",
                    "text": row.get("text") or "",
                    "filename": row.get("filename") or "",
                }
            ]
        )
    return evidence


def _citations_of(evidence: list[dict]) -> list[Citation]:
    return [
        Citation(
            file_id=str(item["file_id"]),
            filename=item.get("filename") or "",
            seq=int(item["chunk_seq"]),
            page=item.get("page"),
            kind=item.get("kind") or "text",
            where=_evidence_where(item),
            text=item.get("text") or "",
        )
        for item in evidence
    ]


kg_search_tool = Tool(
    name="kg_search",
    display_name="图谱检索",
    description=(
        "按实体与关系检索知识库图谱。当问题涉及「某公司/技术/指标和谁有关、"
        "上下游、投资关系、产业链位置」这类关系型问题时优先使用；"
        "只查原文表述时用 kb_search。"
    ),
    func=_kg_search,
    args_model=KgSearchArgs,
)
