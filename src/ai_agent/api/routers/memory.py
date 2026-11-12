"""渐进式记忆 · 第 5 步：L3 候选人工审核入口 + L4 归档查询。

第 4 步落地的候选层只有 ``approve_candidate(id)`` 这个函数，没有可调用的
入口，人工审核实际上无处落点。本路由把「列队列 → 核对原文 → 批准/驳回」
补成闭环，同时给 L4 归档提供只读查询与手动蒸馏触发。

安全约束（与研报 M2 审核同一套纪律）：

- **默认关闭**：``memory_promotion_enabled=False`` 时候选端点一律 409、
  零 DB 访问；归档端点由 ``memory_archive_enabled`` 单独控制。新增路由
  本身不影响任何既有链路。
- **按 user_id 严格隔离**：普通用户只能看/审自己的记忆，请求里的
  ``user_id`` 会被强制忽略；只有管理员可以指定别的用户。
- **出处由人填，绝不由 LLM 补**：含数字/量化词的候选必须带 ``source``
  才能批准，服务端只接受审核人核对原文后手工录入的字符串。
- **人工批准不绕过事实校验**：最终仍走 ``evaluate_candidate`` 的出处门禁
  与 ``apply_profile_action`` 既有的确定性 + Jev 去重。
- **归档永不进 system 人格块**：只在被显式查询/展开时出现。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ai_agent.api.dependencies import get_current_user
from ai_agent.api.schemas import MemoryApproveRequest, MemoryRejectRequest
from ai_agent.core.config import settings
from ai_agent.memory.promotion import has_source, needs_source
from ai_agent.storage.db import UserRecord
from ai_agent.storage.store import store

router = APIRouter(prefix="/v1/memory", tags=["memory"])

_CANDIDATE_STATUS = r"^(pending|promoted|rejected)$"


def _require_promotion() -> None:
    if not settings.memory_promotion_enabled:
        raise HTTPException(
            status_code=409, detail="候选记忆审核未启用（MEMORY_PROMOTION_ENABLED=false）"
        )


def _require_archive() -> None:
    if not settings.memory_archive_enabled:
        raise HTTPException(
            status_code=409, detail="L4 记忆归档未启用（MEMORY_ARCHIVE_ENABLED=false）"
        )


def _is_admin(user: UserRecord) -> bool:
    return settings.is_admin_username(user.username)


def _owner(user: UserRecord, requested: str | None) -> str:
    """普通用户强制看自己；管理员可指定 user_id，未指定时默认自己。"""
    if requested and requested != user.user_id:
        if not _is_admin(user):
            raise HTTPException(status_code=403, detail="只能审核自己的记忆")
        return requested
    return user.user_id


@router.get("/candidates/users")
def list_candidate_users(user: UserRecord = Depends(get_current_user)) -> dict:
    """管理员列出有待审候选的用户 id，供按用户逐个审核。"""
    _require_promotion()
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="仅管理员可查看跨用户审核队列")
    try:
        user_ids = store.list_candidate_users(status="pending")
    except Exception:
        # 表不存在 / DB 抖动都只报 502，不把异常栈漏给前端
        raise HTTPException(status_code=502, detail="候选队列读取失败")
    return {"total": len(user_ids), "user_ids": user_ids}


@router.get("/candidates")
def list_candidates(
    status: str = Query(default="pending", pattern=_CANDIDATE_STATUS),
    limit: int = Query(default=20, ge=1, le=100),
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """候选队列。普通用户只看自己；管理员可用 user_id 指定用户。"""
    _require_promotion()
    owner = _owner(user, user_id)
    try:
        items = store.list_candidates(owner, status=status, limit=limit)
    except Exception:
        raise HTTPException(status_code=502, detail="候选记忆读取失败")
    # 前端要能直接看出哪条卡在「缺出处」，避免审核人反复试错
    for item in items:
        item["requires_source"] = needs_source(str(item.get("text") or ""))
    return {"total": len(items), "user_id": owner, "items": items}


@router.post("/candidates/{candidate_id}/approve")
def approve_candidate(
    candidate_id: str,
    req: MemoryApproveRequest,
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """人工批准：升 L1 画像 + 记录审核人/备注。含数字候选须补人工出处。"""
    _require_promotion()
    owner = _owner(user, user_id)
    try:
        row = store.get_candidate(candidate_id, user_id=owner)
    except Exception:
        raise HTTPException(status_code=502, detail="候选记忆读取失败")
    if not row or row.get("status") != "pending":
        raise HTTPException(status_code=404, detail="没有该待审候选（或已被处置）")

    text = str(row.get("text") or "")
    effective_source = (req.source or "").strip() or str(row.get("source") or "")
    if needs_source(text) and not has_source(effective_source):
        raise HTTPException(
            status_code=409,
            detail="该条含数字/量化事实，请人工核对原文后在 source 中填写出处再批准",
        )

    from ai_agent.memory.promotion import approve_candidate as do_approve

    ok = do_approve(
        candidate_id,
        user_id=owner,
        source=req.source or None,
        reviewed_by=user.username,
        note=req.note,
    )
    if not ok:
        raise HTTPException(
            status_code=409, detail="批准未完成：候选可能已被处置，或被画像去重门禁拦下"
        )
    return {
        "candidate_id": candidate_id,
        "status": "promoted",
        "user_id": owner,
        "reviewed_by": user.username,
        "source": effective_source,
    }


@router.post("/candidates/{candidate_id}/reject")
def reject_candidate(
    candidate_id: str,
    req: MemoryRejectRequest,
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """人工驳回：候选标 rejected 保留追溯，不写画像。"""
    _require_promotion()
    owner = _owner(user, user_id)

    from ai_agent.memory.promotion import reject_candidate as do_reject

    ok = do_reject(
        candidate_id, user_id=owner, reviewed_by=user.username, note=req.note
    )
    if not ok:
        raise HTTPException(
            status_code=404, detail="没有该待审候选（或已被处置、读取失败）"
        )
    return {
        "candidate_id": candidate_id,
        "status": "rejected",
        "user_id": owner,
        "reviewed_by": user.username,
    }


@router.get("/archive")
def search_archive(
    query: str = Query(default="", max_length=200),
    limit: int = Query(default=5, ge=1, le=50),
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """按需展开 L4 归档（确定性相关性排序，零 LLM / 零向量调用）。

    归档**不会**自动进 system，只能被这里或 ``expand_memory`` 显式取回。
    """
    _require_archive()
    owner = _owner(user, user_id)
    from ai_agent.memory.archive import search_archive as do_search

    try:
        items = do_search(query, user_id=owner, top_k=limit)
    except Exception:
        raise HTTPException(status_code=502, detail="归档记忆读取失败")
    return {"total": len(items), "user_id": owner, "items": items}


@router.post("/archive/distill")
def distill_archive(
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """手动触发一次蒸馏。后台清理线程也会定期跑同一函数（幂等）。"""
    _require_archive()
    owner = _owner(user, user_id)
    from ai_agent.memory.archive import distill_memory

    try:
        report = distill_memory(user_id=owner)
    except Exception:
        raise HTTPException(status_code=502, detail="归档蒸馏失败")
    return {"user_id": owner, **report}
