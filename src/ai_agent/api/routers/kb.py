"""知识库文件管理：分片上传、处理进度、列表与删除。

权限模型：知识库是全站共享文档库，**读写分离**——
所有登录用户都能浏览文件列表、查看切片明细、看图谱（查询能力人人有）；
管理操作（上传、删除、修订切片、重建图谱）只有管理员能做。

上传协议（大文件分片）：
  POST   /v1/kb/files                      建档，返回 file_id（管理员）
  PUT    /v1/kb/files/{id}/parts/{index}   逐片传二进制（管理员）
  POST   /v1/kb/files/{id}/complete        合并分片并触发后台解析（管理员）；
                                           内容指纹与库内文件相同则 409 拒绝
  GET    /v1/kb/files                      文件列表（所有登录用户；?q= 文件名
                                           模糊 / ?tag= 标签筛选）
  PATCH  /v1/kb/files/{id}/tags            全量覆盖设置标签（管理员）
  PATCH  /v1/kb/files/{id}                 重命名（管理员）：DB + 磁盘同步改，
                                           stem 变化触发后台整份重嵌
  GET    /v1/kb/files/{id}/download        下载原始文件（所有登录用户）
  GET    /v1/kb/files/{id}/chunks          切片明细（所有登录用户）
  PATCH  /v1/kb/files/{id}/chunks/{seq}    修订切片正文，单片重嵌入（管理员）
  DELETE /v1/kb/files/{id}                 删除文件与切片（管理员）

文件本体放文件系统（kb_upload_dir/{file_id}/），DB 只存元数据；
分片先落 parts/ 临时目录，complete 时按序号合并并校验总大小。
P1 不做断点续传（刷新重传整文件），协议已留好按 index 补片的扩展位。
"""

import hashlib
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from ai_agent.api.dependencies import get_current_user, require_admin
from ai_agent.api.schemas import (
    KbChunkEditRequest,
    KbInitRequest,
    KbRenameRequest,
    KbTagsRequest,
)
from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.chunking import embedding_input
from ai_agent.kb.parsing import supported, supported_label
from ai_agent.kb.pipeline import kb_pipeline
from ai_agent.memory.db import UserRecord
from ai_agent.memory.embeddings import embed_texts
from ai_agent.memory.store import store

router = APIRouter(prefix="/v1/kb", tags=["kb"])

logger = get_logger(__name__)

# 标签上限：单个 24 字、每文件 8 个——够组织文档分类，又不会被滥用成自由文本
_TAG_MAX_LEN = 24
_TAG_MAX_COUNT = 8


def _clean_tags(raw: list[str]) -> list[str]:
    """标签规整：去空白、去空、去重、截断。建档和改标签共用同一套规则。"""
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw:
        tag = item.strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        cleaned.append(tag[:_TAG_MAX_LEN])
        if len(cleaned) >= _TAG_MAX_COUNT:
            break
    return cleaned


def _parse_file_id(file_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(file_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="非法的文件 id") from e


def _upload_dir(file_id: str) -> Path:
    # 磁盘路径用 uuid 做目录名，天然防路径穿越；文件名只入库展示
    return Path(settings.kb_upload_dir) / file_id


def _safe_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name.strip()
    return name or "file"


def _get_file(file_id: str) -> dict:
    record = store.get_kb_file(_parse_file_id(file_id))
    if record is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    return record


def _file_sha256(path: Path) -> str:
    """分块算内容指纹：单文件上限 100MB，整读会让内存峰值随并发上传线性上涨。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@router.post("/files")
def init_upload(
    req: KbInitRequest, user: UserRecord = Depends(require_admin)
):
    max_bytes = settings.kb_max_file_mb * 1024 * 1024
    if req.size_bytes <= 0 or req.size_bytes > max_bytes:
        raise HTTPException(
            status_code=400, detail=f"文件大小需在 1B ~ {settings.kb_max_file_mb}MB 之间"
        )
    suffix = Path(req.filename).suffix.lower()
    if not supported(suffix):
        raise HTTPException(
            status_code=400, detail=f"暂不支持该文件类型，支持：{supported_label()}"
        )
    file_id = store.create_kb_file(
        user.user_id,
        _safe_filename(req.filename),
        req.size_bytes,
        suffix,
        tags=_clean_tags(req.tags),
    )
    logger.info(
        "kb 上传建档 admin=%s file_id=%s file=%s",
        user.username,
        file_id,
        req.filename,
    )
    return {"file_id": str(file_id)}


@router.put("/files/{file_id}/parts/{index}")
async def upload_part(
    file_id: str,
    index: int,
    request: Request,
    _user: UserRecord = Depends(require_admin),
):
    record = _get_file(file_id)
    if record["status"] != "uploaded":
        raise HTTPException(status_code=409, detail="文件已开始处理，不能继续上传")
    if index < 0 or index >= 100000:
        raise HTTPException(status_code=400, detail="分片序号非法")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="分片内容为空")
    if len(body) > settings.kb_part_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail=f"单片不能超过 {settings.kb_part_size_mb}MB"
        )
    part_dir = _upload_dir(file_id) / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    (part_dir / f"{index:06d}.part").write_bytes(body)
    return {"ok": True}


@router.post("/files/{file_id}/complete")
def complete_upload(
    file_id: str, user: UserRecord = Depends(require_admin)
):
    record = _get_file(file_id)
    if record["status"] != "uploaded":
        raise HTTPException(status_code=409, detail="文件已在处理中，请勿重复提交")
    part_dir = _upload_dir(file_id) / "parts"
    parts = sorted(part_dir.glob("*.part")) if part_dir.is_dir() else []
    if not parts:
        raise HTTPException(status_code=400, detail="没有收到任何分片，请重新上传")

    final_dir = _upload_dir(file_id)
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / record["filename"]
    with final_path.open("wb") as out:
        for part in parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src, out)
    size = final_path.stat().st_size
    if size != record["size_bytes"]:
        final_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail="分片合计大小与声明不符，请重新上传"
        )
    shutil.rmtree(part_dir, ignore_errors=True)

    # 内容指纹去重：同内容文件拒绝重复入库（向量切片、图谱抽取都是真金白银）。
    # 在提交解析之前拦，干净地删掉档案与已合并文件，让管理员直接用已有的那份
    digest = _file_sha256(final_path)
    _check_duplicate(digest, record["id"], record["filename"], user.username)
    store.update_kb_file_hash(record["id"], digest)

    kb_pipeline.submit(_parse_file_id(file_id), user.user_id, final_path)
    return {"status": "parsing"}


def _check_duplicate(
    digest: str, file_id: uuid.UUID, filename: str, admin: str
) -> None:
    """同内容已有文件时拒绝入库：删档案与文件本体，409 指向已存在的那份。"""
    duplicate = store.find_kb_file_by_hash(digest, exclude_file_id=file_id)
    if duplicate is None:
        return
    store.delete_kb_file(file_id)
    shutil.rmtree(_upload_dir(str(file_id)), ignore_errors=True)
    logger.info(
        "kb 上传去重 admin=%s file=%s 与 file_id=%s 内容相同",
        admin,
        filename,
        duplicate["file_id"],
    )
    raise HTTPException(
        status_code=409,
        detail=f"与已有文件《{duplicate['filename']}》内容完全相同，已拒绝重复入库",
    )


@router.get("/files")
def list_files(
    q: str = "",
    tag: str = "",
    _user: UserRecord = Depends(get_current_user),
):
    """文件列表对所有登录用户开放（普通用户只读浏览 + 对话检索）。

    q 按文件名模糊筛选，tag 按标签筛选；两者可叠加。
    """
    return {"files": store.list_kb_files(q=q.strip(), tag=tag.strip())}


@router.patch("/files/{file_id}/tags")
def set_file_tags(
    file_id: str, req: KbTagsRequest, _user: UserRecord = Depends(require_admin)
):
    """全量覆盖式设置标签（传空数组清空），管理员操作。"""
    fid = _parse_file_id(file_id)
    if store.get_kb_file(fid) is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    tags = _clean_tags(req.tags)
    store.set_kb_file_tags(fid, tags)
    return {"ok": True, "tags": tags}


@router.patch("/files/{file_id}")
def rename_file(
    file_id: str, req: KbRenameRequest, user: UserRecord = Depends(require_admin)
):
    """管理员重命名：DB 展示名 + 磁盘文件同步改，stem 变了再后台重嵌。

    两个不能省的联动：
    - 磁盘文件按 filename 落盘（complete_upload），只改 DB 下载就 404；
    - 切片向量在嵌入时烧进了《文件名》语境前缀，改名不重嵌，这份文件的
      向量语境就永远停在旧名上。
    """
    fid = _parse_file_id(file_id)
    record = _get_file(file_id)
    if record["status"] != "ready":
        raise HTTPException(status_code=409, detail="文件处理完成后才能改名")

    old_name = record["filename"]
    new_name = _safe_filename(req.filename)
    old_suffix = Path(old_name).suffix.lower()
    if Path(new_name).suffix.lower() != old_suffix:
        raise HTTPException(
            status_code=400,
            detail=f"扩展名需保持 {old_suffix} 不变（解析器与下载类型都按它识别）",
        )
    if new_name == old_name:
        return {"ok": True, "filename": old_name, "reembedding": False}

    stem_changed = Path(new_name).stem != Path(old_name).stem
    store.rename_kb_file(fid, new_name)
    old_path = _upload_dir(file_id) / old_name
    if old_path.is_file():
        old_path.rename(_upload_dir(file_id) / new_name)
    if stem_changed:
        kb_pipeline.submit_reembed(fid)
    logger.info(
        "kb 改名 admin=%s file_id=%s %s -> %s reembed=%s",
        user.username,
        fid,
        old_name,
        new_name,
        stem_changed,
    )
    return {"ok": True, "filename": new_name, "reembedding": stem_changed}


@router.get("/files/{file_id}/download")
def download_file(file_id: str, _user: UserRecord = Depends(get_current_user)):
    """下载原始文件。内容对全站可读（切片都能看），下载同步开放给登录用户。"""
    record = _get_file(file_id)
    if record["status"] == "uploaded":
        # 只有分片、还没合并出成品文件
        raise HTTPException(status_code=409, detail="文件尚未上传完成，暂不能下载")
    path = _upload_dir(file_id) / record["filename"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件本体不存在（可能已损坏）")
    return FileResponse(path, filename=record["filename"])


@router.get("/files/{file_id}/chunks")
def list_chunks(
    file_id: str,
    limit: int = 50,
    offset: int = 0,
    _user: UserRecord = Depends(get_current_user),
):
    """查看某文件的切片明细（seq 顺序，分页），所有登录用户可读。"""
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)
    fid = _parse_file_id(file_id)
    if store.get_kb_file(fid) is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    chunks, total = store.list_kb_chunks(fid, limit=limit, offset=offset)
    return {"chunks": chunks, "total": total}


@router.patch("/files/{file_id}/chunks/{seq}")
def edit_chunk(
    file_id: str,
    seq: int,
    req: KbChunkEditRequest,
    user: UserRecord = Depends(require_admin),
):
    """人工修订切片正文：保存即对这一片重新向量化。

    切片向量彼此独立，重嵌一整份文件纯属浪费；但图谱是从切片文本抽出来的，
    改完文本图谱就基于旧内容了，这里把 graph_status 置为 stale，
    由管理员决定何时重建（POST /v1/kg/graph/build）。
    """
    fid = _parse_file_id(file_id)
    if seq < 0:
        raise HTTPException(status_code=400, detail="切片序号非法")
    record = _get_file(file_id)
    if record["status"] != "ready":
        raise HTTPException(status_code=409, detail="文件处理完成后才能编辑切片")

    # 只修文本：不允许借编辑增删/拆分切片，seq 是引用与 mention 回溯的身份
    text = req.text.strip()
    max_len = int(settings.kb_chunk_size * 1.5)
    if not text:
        raise HTTPException(status_code=400, detail="切片内容不能为空")
    if len(text) > max_len:
        raise HTTPException(
            status_code=400,
            detail=f"切片长度上限 {max_len} 字；内容太长请拆分后分别保存到相邻切片",
        )

    chunk = store.get_kb_chunk(fid, seq)
    if chunk is None:
        raise HTTPException(status_code=404, detail="切片不存在")
    if text == chunk["text"]:
        return {"ok": True, "changed": False, "graph_status": chunk["graph_status"]}

    try:
        # 与入库时保持一致：嵌入带「文档名 + 页码/表号」前缀，
        # 否则人工修订过的切片在向量空间里跟其他切片不同源，召回会偏
        vector = embed_texts(
            [
                embedding_input(
                    text,
                    filename=chunk.get("filename") or "",
                    page=chunk["page"],
                    kind=chunk["kind"],
                    section=chunk.get("section"),
                )
            ]
        )[0]
    except Exception as e:
        raise HTTPException(status_code=502, detail="向量化服务不可用，请稍后重试") from e

    store.update_kb_chunk_text(chunk["id"], text, vector)
    graph_status = chunk["graph_status"]
    if graph_status == "ready":
        store.update_graph_status(fid, "stale")
        graph_status = "stale"
    logger.info(
        "kb 切片修订 admin=%s file_id=%s seq=%d len=%d graph=%s",
        user.username,
        fid,
        seq,
        len(text),
        graph_status,
    )
    return {"ok": True, "changed": True, "graph_status": graph_status}


@router.delete("/files/{file_id}")
def delete_file(
    file_id: str, _user: UserRecord = Depends(require_admin)
):
    deleted = store.delete_kb_file(_parse_file_id(file_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="文件不存在")
    shutil.rmtree(_upload_dir(file_id), ignore_errors=True)
    return {"ok": True}
