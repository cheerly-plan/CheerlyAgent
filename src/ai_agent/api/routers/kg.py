"""知识图谱接口：看图、检索、重建。

权限与知识库一致（读写分离）：看图、检索对所有登录用户开放；
触发抽取的写操作（建图/重建）只有管理员能做。
图谱检索是「实体向量匹配 → 邻域扩展 → 回捞切片原文」，返回可直接渲染的子图
和作为证据的原文片段，前端不必再拼一次请求。

  GET  /v1/kg/graph                        全库图谱（所有登录用户）
  GET  /v1/kg/files/{file_id}/graph        单文件图谱（所有登录用户）
  GET  /v1/kg/search?q=&top_k=&depth=      图谱检索（所有登录用户）
  GET  /v1/kg/entities/{entity_id}/chunks  实体的原文证据（所有登录用户）
  GET  /v1/kg/doc-types                    可选抽取配置（所有登录用户）
  POST /v1/kg/graph/build                  管理员选文件批量生成图谱（后台执行）
  POST /v1/kg/files/{file_id}/rebuild      重建该文件图谱（后台执行，管理员；
                                           body 里的 doc_type/level 可省略，
                                           省略时沿用该文件上次的抽取配置）
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException

from ai_agent.api.dependencies import get_current_user, require_admin
from ai_agent.api.schemas import KgBuildRequest, KgRebuildRequest
from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.community import detect_communities, summarize
from ai_agent.kb.doc_profiles import list_profiles
from ai_agent.kb.pipeline import kb_pipeline
from ai_agent.memory.db import UserRecord
from ai_agent.memory.embeddings import embed_query
from ai_agent.memory.store import store

router = APIRouter(prefix="/v1/kg", tags=["kg"])

logger = get_logger(__name__)


def _parse_uuid(raw: str, label: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"非法的{label} id") from e


def _parse_uuid_list(raw: str, label: str) -> list[uuid.UUID] | None:
    """逗号分隔的 id 列表 → uuid 列表；空串 = None（不限定范围）。"""
    parts = [p for p in (raw or "").split(",") if p.strip()]
    if not parts:
        return None
    return [_parse_uuid(p, label) for p in parts]


def _extract_config(doc_type: str, level: str) -> tuple[str, str]:
    """校验并归一化一次建图/重建携带的抽取配置。

    空串原样返回——「没选」本身是合法状态，落库存空、抽取时由 pipeline 回落
    全局默认的通用配置；未知 key 直接 400，不静默回落（用户以为选上了、实际
    跑的是另一套配置，比报错更糟）。level 空串也原样返回，抽取时归一。
    """
    doc_type = (doc_type or "").strip()
    level = (level or "").strip()
    known = {p.key for p in list_profiles()}
    if doc_type and doc_type not in known:
        raise HTTPException(status_code=400, detail=f"未知文档类型：{doc_type}")
    return doc_type, level



def _node(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "name": row["name"],
        # 规范名：跨文件同名实体的合并键。前端拿它把检索命中的实体对到画布节点上，
        # 只靠 id 会对不上——命中记录和图上保留的代表记录可能不是同一条
        "norm_name": row.get("norm_name") or "",
        "type": row["type"],
        "description": row.get("description") or "",
        "mentions": int(row.get("mention_count") or 1),
        # 在返回边集里的连接数：前端全库视图据此隐藏孤独点、挑枢纽节点。
        # 单文件图/检索子图没有这个字段，前端自己兜底
        "degree": int(row.get("degree") or 0),
        "file_id": str(row["file_id"]) if row.get("file_id") else "",
        "filename": row.get("filename") or "",
    }


def _edge(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "source": str(row["source_id"]),
        "target": str(row["target_id"]),
        "relation": row["relation"],
        "description": row.get("description") or "",
        "weight": float(row.get("weight") or 1.0),
        "chunk_seq": row.get("chunk_seq"),
    }


def _shape(graph: dict, *, seeds: list | None = None) -> dict:
    nodes = [_node(n) for n in graph["nodes"]]
    node_ids = {n["id"] for n in nodes}
    edges = [_edge(e) for e in graph["edges"]]
    edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]
    # 社区检测（簇视图）：结果随图一起给，前端不必自己算连通块。
    # 放在这里而不是建图时落库——图只有几百个点，标签传播毫秒级，
    # 而落库还得考虑"对齐后社区要重算"的失效问题
    mapping = detect_communities(nodes, edges)
    for item in nodes:
        item["community"] = mapping.get(item["id"], 0)
    payload = {"nodes": nodes, "edges": edges}
    payload["communities"] = summarize(nodes, mapping)
    if seeds is not None:
        payload["seeds"] = seeds
    return payload


@router.get("/graph")
def global_graph(
    file_ids: str = "",
    _user: UserRecord = Depends(get_current_user),
):
    """全库图谱。节点按提及次数取前 N 个，边只保留两端都在节点集里的。

    `file_ids`（逗号分隔）非空时只看这些文件的合并视图：管理员勾选
    同一行业的文件合并出图，范围外的文件不掺进来；空 = 全库。
    """
    scope = _parse_uuid_list(file_ids, "文件")
    graph = store.get_global_graph(
        limit=settings.kb_graph_max_nodes, file_ids=scope
    )
    stats = store.graph_stats(file_ids=scope)
    payload = _shape(graph)
    payload["stats"] = stats
    return payload


@router.get("/files/{file_id}/graph")
def file_graph(file_id: str, _user: UserRecord = Depends(get_current_user)):
    fid = _parse_uuid(file_id, "文件")
    record = store.get_kb_file(fid)
    if record is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    graph = store.get_file_graph(fid, limit=settings.kb_graph_max_nodes)
    payload = _shape(graph)
    payload["filename"] = record["filename"]
    # 概览统计（实体/关系/文件数）与全库视图同口径，页面标题后的统计标签用。
    # 单文件视图的 files 最多是 1，前端据此决定要不要显示「文件」那颗标签
    payload["stats"] = store.graph_stats(file_ids=[fid])
    # 文件表的图谱进度，前端据此显示「未构建 / 构建中 / 失败」并可触发重建
    payload.update(store.get_file_graph_status(fid))
    # 对齐进度与合并数（前端决定是否显示「也叫 …」与「对齐」按钮）
    payload.update(store.get_align_status(fid))
    # 这份图是按哪套抽取配置抽的（空 = 还没选过，用全局默认）
    payload["doc_type"] = (record.get("doc_type") or "").strip() or settings.kb_default_doc_type
    payload["extract_level"] = (record.get("extract_level") or "").strip() or "full"
    return payload


@router.get("/search")
def search_graph(
    q: str,
    top_k: int = 6,
    depth: int = 1,
    file_id: str = "",
    file_ids: str = "",
    _user: UserRecord = Depends(get_current_user),
):
    """图谱检索：query → 种子实体 → N 跳子图 + 关联切片。

    与 kb_search（切片向量直查）互补：这条按「实体与关系」召回，
    能找回原文里没直接出现关键词、但通过关系相连的内容。

    范围：`file_ids`（逗号分隔，合并视图用）优先；否则退回单文件
    `file_id`；都为空 = 全库。检索范围必须和画布图同源，否则命中的
    实体不在画布上，看起来就像「检索没过滤」。
    """
    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="检索词不能为空")
    top_k = min(max(top_k, 1), 20)
    depth = min(max(depth, 1), 2)
    scope = _parse_uuid_list(file_ids, "文件")
    if scope is None:
        scope = [_parse_uuid(file_id, "文件")] if file_id else None
    file_ids_arg = scope

    try:
        embedding = embed_query(query)
    except Exception as e:
        raise HTTPException(status_code=502, detail="向量化服务不可用") from e

    seeds = store.search_entities(
        embedding,
        query=query,
        top_k=top_k,
        file_ids=file_ids_arg,
        min_score=settings.kb_graph_min_score,
    )
    if not seeds:
        # 有候选但都被噪音门槛切掉了，也是「没找到」，交给前端提示换个说法
        return {
            "query": query,
            "nodes": [],
            "edges": [],
            "seeds": [],
            "chunks": [],
            "empty": True,
        }

    graph = store.expand_subgraph(
        [s["id"] for s in seeds],
        depth=depth,
        max_nodes=settings.kb_graph_max_nodes,
    )
    payload = _shape(
        graph,
        seeds=[
            {
                "id": str(s["id"]),
                "name": s["name"],
                "norm_name": s.get("norm_name") or "",
                "type": s["type"],
                "score": round(float(s["score"] or 0), 3),
                "filename": s.get("filename") or "",
            }
            for s in seeds
        ],
    )

    # 证据片段：按「规范名」聚合，而不是只认代表记录。
    # 命中的是一个实体名，它在每份文档里各存一条记录，正文也散在各份文件里；
    # 只取代表记录的话，两份文件都提到的 HBM 只会回一份的原文。
    norm_ids = store.entity_ids_by_norms(
        [s["norm_name"] for s in seeds if s.get("norm_name")],
        file_ids=file_ids_arg,
    )
    grouped = store.get_entity_chunks(
        [eid for ids in norm_ids.values() for eid in ids]
    )
    chunks: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for seed in seeds:
        ids = norm_ids.get(seed.get("norm_name") or "") or [seed["id"]]
        for eid in ids:
            for item in grouped.get(str(eid), []):
                key = (item["filename"], item["chunk_seq"])
                if key in seen:
                    continue
                seen.add(key)
                chunks.append({**item, "entity": seed["name"]})
    payload["chunks"] = chunks[:12]
    payload["empty"] = not payload["nodes"]
    return payload


@router.get("/entities/{entity_id}/chunks")
def entity_chunks(entity_id: str, _user: UserRecord = Depends(get_current_user)):
    eid = _parse_uuid(entity_id, "实体")
    grouped = store.get_entity_chunks([eid], per_entity=20)
    return {"chunks": grouped.get(str(eid), [])}


@router.post("/graph/build")
def build_graphs(req: KgBuildRequest, user: UserRecord = Depends(require_admin)):
    """管理员选文件批量生成图谱。

    上传后不再自动建图：管理员人工核对/修订切片，再在这里显式选择
    哪些文件进图谱。逐个文件校验并投递到图谱线程池，接口立即返回，
    进度看各文件的 graph_status。
    """
    if not settings.kb_graph_enabled:
        raise HTTPException(status_code=409, detail="图谱功能已关闭（KB_GRAPH_ENABLED=false）")

    # 批量建图可以顺手统一套一套配置（"这批都是年报"），配置是这次构建的参数
    doc_type, level = _extract_config(req.doc_type, req.level)

    results: list[dict] = []
    submitted = 0
    for raw in req.file_ids:
        fid = _parse_uuid(raw, "文件")
        record = store.get_kb_file(fid)
        if record is None:
            results.append({"file_id": raw, "status": "not_found"})
            continue
        if record["status"] != "ready":
            results.append({"file_id": raw, "status": "not_ready"})
            continue
        # 建图中的文件跳过，不重复投递（图谱线程池串行，重复投只会排队空等）
        if store.get_file_graph_status(fid)["graph_status"] == "building":
            results.append({"file_id": raw, "status": "building"})
            continue
        # 配置为空也写库：清掉这份文件之前可能存过的旧配置，抽时回落全局默认
        store.set_kb_extract_config(fid, doc_type=doc_type, level=level)
        store.update_graph_status(fid, "building")
        kb_pipeline.submit_graph(fid)
        submitted += 1
        results.append({"file_id": raw, "status": "building"})
    if submitted:
        logger.info(
            "kg 批量建图提交 admin=%s files=%d", user.username, submitted
        )
    return {"results": results}


@router.post("/files/{file_id}/rebuild")
def rebuild_graph(
    file_id: str,
    req: KgRebuildRequest | None = None,
    user: UserRecord = Depends(require_admin),
):
    """重建图谱：用库里已有的切片重新抽取（不重新解析源文件）。

    用于「切片修订后图谱过期（stale）」「升级前上传的老文件」
    或「上次抽取失败」的情况；任务丢进图谱线程池，接口立即返回，
    进度看 graph_status。

    `req` 可选：
    - 不带 body = 沿用该文件已保存的配置（脚本/过期刷新不该顺手改配置）；
    - 带 body = 把配置作为这次构建的参数落库（含空值，清成通用）再重抽。
    """
    if not settings.kb_graph_enabled:
        raise HTTPException(status_code=409, detail="图谱功能已关闭（KB_GRAPH_ENABLED=false）")
    fid = _parse_uuid(file_id, "文件")
    record = store.get_kb_file(fid)
    if record is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    if record["status"] != "ready":
        raise HTTPException(status_code=409, detail="文件还没解析完成，无法建图")
    # 已在建图中：直接返回，不重复投递
    if store.get_file_graph_status(fid)["graph_status"] == "building":
        return {"status": "building"}

    if req is not None:
        doc_type, level = _extract_config(req.doc_type, req.level)
        store.set_kb_extract_config(fid, doc_type=doc_type, level=level)

    store.update_graph_status(fid, "building")
    kb_pipeline.submit_graph(fid)
    logger.info("kg 重建提交 file_id=%s admin=%s", fid, user.username)
    return {"status": "building"}


@router.get("/doc-types")
def doc_types(_user: UserRecord = Depends(get_current_user)):
    """可选抽取配置：前端建图前选文档类型，选中后作为本次构建的参数提交。

    只给摘要（label / 实体范围 / 谓词前几条），不把整段 prompt 丢给前端——
    prompt 是服务端实现细节，前端需要的是「选哪个更合适」的判断依据。
    """
    return {
        "types": [p.brief() for p in list_profiles()],
        "default": settings.kb_default_doc_type,
    }


@router.post("/files/{file_id}/align")
def align_entities(file_id: str, user: UserRecord = Depends(require_admin)):
    """实体对齐（图谱质量 P2）：把同一实体的多种写法合并成一个点。

    对齐要逐对裁决，几百个实体是几分钟的活，丢进图谱线程池后台跑，
    接口立即返回；进度看 align_status，结果看 alias_count。
    """
    if not settings.kb_graph_enabled:
        raise HTTPException(status_code=409, detail="图谱功能已关闭（KB_GRAPH_ENABLED=false）")
    fid = _parse_uuid(file_id, "文件")
    record = store.get_kb_file(fid)
    if record is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    status = store.get_file_graph_status(fid)
    if status["graph_status"] != "ready" or not status["entity_count"]:
        raise HTTPException(status_code=409, detail="该文件还没有可对齐的图谱")
    if store.get_align_status(fid)["align_status"] == "aligning":
        raise HTTPException(status_code=409, detail="该文件正在对齐中")
    store.set_align_status(fid, "aligning")
    kb_pipeline.submit_align(fid)
    logger.info("kg 实体对齐提交 file_id=%s admin=%s", fid, user.username)
    return {"status": "aligning"}


@router.get("/files/{file_id}/aliases")
def list_aliases(file_id: str, _user: UserRecord = Depends(get_current_user)):
    """该文件里被合并掉的写法：前端在实体详情里展示「也叫 …」。"""
    fid = _parse_uuid(file_id, "文件")
    return {
        "aliases": store.list_aliases(fid),
        **store.get_align_status(fid),
    }
