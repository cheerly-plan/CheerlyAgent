"""渐进式记忆 · 第 5 步：L3 候选人工审核入口 + L4 归档查询。

第 4 步落地的候选层只有 ``approve_candidate(id)`` 这个函数，没有可调用的
入口，人工审核实际上无处落点。本路由把「列队列 → 核对原文 → 批准/驳回」
补成闭环，同时给 L4 归档提供只读查询与手动蒸馏触发。

安全约束（与研报 M2 审核同一套纪律）：

- **恒定可用**：候选层与归档层常驻（渐进式记忆没有总开关），端点的
  权限隔离与出处门禁是唯一的防线。
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
from ai_agent.api.schemas import (
    MemoryApproveRequest,
    MemoryProfileWriteRequest,
    MemoryRejectRequest,
    MemorySettingsRequest,
)
from ai_agent.core.config import settings
from ai_agent.memory.profile import apply_profile_text
from ai_agent.memory.promotion import has_source, needs_source
from ai_agent.storage.db import UserRecord
from ai_agent.storage.store import store

router = APIRouter(prefix="/v1/memory", tags=["memory"])

_CANDIDATE_STATUS = r"^(pending|promoted|rejected)$"


def _is_admin(user: UserRecord) -> bool:
    return settings.is_admin_username(user.username)


def _owner(user: UserRecord, requested: str | None) -> str:
    """普通用户强制看自己；管理员可指定 user_id，未指定时默认自己。"""
    if requested and requested != user.user_id:
        if not _is_admin(user):
            raise HTTPException(status_code=403, detail="只能操作自己的记忆")
        return requested
    return user.user_id


@router.get("/candidates/users")
def list_candidate_users(user: UserRecord = Depends(get_current_user)) -> dict:
    """管理员列出有待审候选的用户 id，供按用户逐个审核。"""
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


@router.get("/settings")
def get_memory_settings(
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """读取用户级记忆开关。默认开启。"""
    owner = _owner(user, user_id)
    return {
        "user_id": owner,
        "enabled": store.memory_enabled(owner),
    }


@router.put("/settings")
def update_memory_settings(
    req: MemorySettingsRequest,
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """打开/关闭用户级记忆。

    关闭后：不写新记忆、不注入旧记忆、不挂 memory 工具；已有数据保留，
    用户仍可通过本路由的 profile/episodes/export 查看、导出、删除。
    """
    owner = _owner(user, user_id)
    try:
        enabled = store.set_memory_enabled(req.enabled, user_id=owner)
    except Exception:
        raise HTTPException(status_code=502, detail="记忆开关保存失败")
    return {"user_id": owner, "enabled": enabled}


@router.get("/profile")
def list_profile(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_inactive: bool = Query(default=False),
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """分页查看 L1 画像；include_inactive 可看到被覆盖/软过期的留档。"""
    owner = _owner(user, user_id)
    try:
        items, has_more = store.list_profile_page(
            owner,
            limit=limit,
            offset=offset,
            include_inactive=include_inactive,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="画像记忆读取失败")
    return {
        "user_id": owner,
        "total": len(items),
        "items": items,
        "has_more": has_more,
        "next_offset": offset + len(items) if has_more else None,
    }


@router.post("/profile")
def write_profile_from_text(
    req: MemoryProfileWriteRequest,
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """设置页用自然语言管理 L1 画像：新增、修改、删除都走同一条写入链。

    复用对话里 ``write_memory`` 工具的 ``apply_profile_text``：
    ``resolve_profile_write`` 把一句话解析成 add/update/delete/skip actions，
    ``apply_profile_action`` 再逐条过 facet 归一化、冗余与 Jev 语义去重、
    冲突消解（旧值标覆盖而非物理删）、软过期 TTL；delete 还会连带清掉
    同文 pending 候选。判断为「不该写」时返回说明，不碰任何一行。

    这里刻意不提供「按 id 直接改某一行」的入口——那会绕开上述判断。
    """
    owner = _owner(user, user_id)
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="记忆内容不能为空")
    try:
        message = apply_profile_text(text, user_id=owner)
    except Exception:
        raise HTTPException(status_code=502, detail="记忆更新失败")
    return {"user_id": owner, "message": message}


@router.delete("/profile/{item_id}")
def delete_profile_item(
    item_id: str,
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """删除一条 L1 画像（同时清掉同文候选与归档副本，避免重新出现）。"""
    owner = _owner(user, user_id)
    try:
        deleted = store.delete_profile_item(item_id, user_id=owner)
    except Exception:
        raise HTTPException(status_code=502, detail="画像记忆删除失败")
    if not deleted:
        raise HTTPException(status_code=404, detail="没有该画像条目")
    return {"user_id": owner, "item_id": item_id, "deleted": True}


@router.delete("/profile")
def clear_profile(
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """清空该用户的 L1 画像，返回删除条数。"""
    owner = _owner(user, user_id)
    try:
        deleted = store.clear_profile(owner)
    except Exception:
        raise HTTPException(status_code=502, detail="画像记忆清空失败")
    return {"user_id": owner, "deleted": deleted}


@router.get("/episodes")
def list_episodes(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_expired: bool = Query(default=False),
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """分页查看 L2 情景记忆；include_expired 可看到已过 TTL 尚未物理删除的。"""
    owner = _owner(user, user_id)
    try:
        items, has_more = store.list_episodes_page(
            owner,
            limit=limit,
            offset=offset,
            include_expired=include_expired,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="情景记忆读取失败")
    return {
        "user_id": owner,
        "total": len(items),
        "items": items,
        "has_more": has_more,
        "next_offset": offset + len(items) if has_more else None,
    }


@router.delete("/episodes/{episode_id}")
def delete_episode(
    episode_id: str,
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """删除一条 L2 情景记忆（含归档 episode 副本与 pending 候选）。"""
    owner = _owner(user, user_id)
    try:
        deleted = store.delete_episode(episode_id, user_id=owner)
    except Exception:
        raise HTTPException(status_code=502, detail="情景记忆删除失败")
    if not deleted:
        raise HTTPException(status_code=404, detail="没有该情景记忆")
    return {"user_id": owner, "episode_id": episode_id, "deleted": True}


@router.delete("/episodes")
def clear_episodes(
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """清空该用户的 L2 情景记忆，返回删除条数。"""
    owner = _owner(user, user_id)
    try:
        deleted = store.clear_episodes(owner)
    except Exception:
        raise HTTPException(status_code=502, detail="情景记忆清空失败")
    return {"user_id": owner, "deleted": deleted}


@router.get("/export")
def export_memory(
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """导出 L1–L4 全量记忆（含开关状态与候选/归档），用于备份或迁移。"""
    owner = _owner(user, user_id)
    try:
        return store.export_memory(owner)
    except Exception:
        raise HTTPException(status_code=502, detail="记忆导出失败")


@router.delete("")
def clear_memory(
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """删除该用户全部长期记忆（画像/情景/候选/归档），不可恢复。"""
    owner = _owner(user, user_id)
    try:
        store.clear_long_term(owner)
    except Exception:
        raise HTTPException(status_code=502, detail="记忆删除失败")
    return {"user_id": owner, "deleted": True}


@router.get("/archive")
def search_archive(
    query: str = Query(default="", max_length=200),
    limit: int = Query(default=5, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """按需展开 L4 归档（确定性相关性排序，零 LLM / 零向量调用）。

    归档**不会**自动进 system，只能被这里或 ``expand_memory`` 显式取回。
    query 为空时按 archived_at 倒序分页；有 query 时按相关性排序后分页。
    """
    owner = _owner(user, user_id)
    try:
        if query.strip():
            from ai_agent.memory.archive import search_archive as do_search

            items = do_search(
                query,
                user_id=owner,
                top_k=limit + 1,
                offset=offset,
            )
        else:
            items = store.list_archive_rows(
                owner, limit=limit + 1, offset=offset
            )
            for item in items:
                item["score"] = 0.0
    except Exception:
        raise HTTPException(status_code=502, detail="归档记忆读取失败")
    has_more = len(items) > limit
    items = items[:limit]
    return {
        "user_id": owner,
        "total": len(items),
        "items": items,
        "has_more": has_more,
        "next_offset": offset + len(items) if has_more else None,
    }


@router.post("/archive/distill")
def distill_archive(
    user_id: str | None = Query(default=None),
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """手动触发一次蒸馏。后台清理线程也会定期跑同一函数（幂等）。"""
    owner = _owner(user, user_id)
    from ai_agent.memory.archive import distill_memory

    try:
        report = distill_memory(user_id=owner)
    except Exception:
        raise HTTPException(status_code=502, detail="归档蒸馏失败")
    return {"user_id": owner, **report}
