"""渐进式记忆 · 第 5 步：L4 蒸馏归档（决策 + 编排层）。

第 4 步把「观察 → 跨会话复现 → 候选 → 证据达标自动晋升」打通了，但晋升后的结论和
长期高热度近况没有沉淀去向：episode 仍会因 TTL 慢慢降权退场，候选表只留
状态。L4 做两件事：

1. **蒸馏**：把「已晋升的候选」和「跨会话、高热度 episode」写成
   归档副本（幂等、带出处），供事后回溯；
2. **按需展开**：归档**永不自动进 system 人格块**，只能被
   ``GET /v1/memory/archive`` 或 ``expand_memory`` 显式取回，排序用
   ``progressive.relevance`` 的确定性 bigram（零 LLM / 零向量）。

纪律与前面各步一致：

- **不搬空热库**：归档只是副本，episode / 画像 / 检索结果一字不改；
- **不自动放行事实**：含数字条目能进归档，不代表能进画像，出处门禁仍在
  第 4 步的 ``evaluate_candidate`` 里，本模块不做任何事实判定；
- **恒定开启**：没有总开关；归档的准入由证据门槛（``memory_archive_min_*``）
  决定，任一 DB 异常只返回空结果；
- **fail-open**：任何 DB 异常只记日志，后台任务/接口返回空结果，不影响主对话。
"""

from __future__ import annotations

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.memory.progressive import relevance
from ai_agent.storage.db import MemoryItem, current_user_id
from ai_agent.storage.store import store

logger = get_logger(__name__)

# 归档条目文本上限：避免把一整段会话摘要塞进冷存储，也让查询输出可控。
_ARCHIVE_TEXT_MAX = 800


def _as_memory_item(row: dict, user_id: str) -> MemoryItem:
    return MemoryItem(
        id=str(row.get("id") or ""),
        user_id=user_id,
        text=str(row.get("text") or ""),
        created_at=str(row.get("archived_at") or ""),
        facet=str(row.get("facet") or "other"),
    )


def search_archive(
    query: str,
    *,
    user_id: str | None = None,
    top_k: int | None = None,
    offset: int = 0,
) -> list[dict]:
    """按当前问句展开归档。确定性排序、零模型调用、异常返回空列表。

    ``offset`` 用于管理页翻页：先按相关性排全部候选，再切片。为了保证
    深页也能拿到结果，读取窗口会随 offset 扩大（上限仍受归档批量控制）。
    """
    uid = user_id or current_user_id()
    k = settings.memory_archive_top_k if top_k is None else max(1, top_k)
    start = max(0, int(offset))
    try:
        rows = store.list_archive_rows(
            uid,
            limit=settings.memory_archive_batch + start,
        )
    except Exception:
        logger.exception("L4 归档读取失败 user=%s", uid)
        return []
    ranked = sorted(
        rows,
        key=lambda row: (
            -relevance(query, _as_memory_item(row, uid)),
            str(row.get("id") or ""),
        ),
    )
    out: list[dict] = []
    for row in ranked[start : start + k]:
        item = dict(row)
        item["score"] = round(relevance(query, _as_memory_item(row, uid)), 4)
        out.append(item)
    return out


def distill_memory(
    *, user_id: str | None = None, limit: int | None = None
) -> dict:
    """蒸馏一个用户的稳定记忆为归档副本。幂等，可反复调用。"""
    uid = user_id or current_user_id()
    batch = settings.memory_archive_batch if limit is None else max(1, limit)

    candidates: list[dict] = []
    episodes: list[dict] = []
    try:
        candidates = store.list_stable_candidates(uid, limit=batch)
    except Exception:
        logger.exception("L4 归档：已批准候选读取失败 user=%s", uid)
    try:
        episodes = store.list_stable_episodes(
            uid,
            min_hits=settings.memory_archive_min_hits,
            min_sessions=settings.memory_archive_min_sessions,
            limit=batch,
        )
    except Exception:
        logger.exception("L4 归档：稳定情景读取失败 user=%s", uid)

    entries: list[dict] = []
    for row in candidates:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        entries.append(
            {
                "kind": "promoted",
                "origin_id": str(row.get("id") or ""),
                "text": text[:_ARCHIVE_TEXT_MAX],
                "facet": str(row.get("facet") or "other"),
                "source": str(row.get("source") or ""),
                "hit_count": int(row.get("hit_count") or 0),
                "session_seen": int(row.get("session_seen") or 0),
                "reviewed_by": str(row.get("reviewed_by") or ""),
            }
        )
    for row in episodes:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        entries.append(
            {
                "kind": "episode",
                "origin_id": str(row.get("id") or ""),
                "text": text[:_ARCHIVE_TEXT_MAX],
                "facet": "other",
                "source": "",
                "hit_count": int(row.get("hit_count") or 0),
                "session_seen": int(row.get("session_seen") or 0),
                "reviewed_by": "",
            }
        )
    try:
        written = store.write_archive_entries(entries, user_id=uid)
    except Exception:
        logger.exception("L4 归档写入失败 user=%s", uid)
        written = 0
    logger.info(
        "memory_archive user=%s candidates=%d episodes=%d written=%d",
        uid,
        len(candidates),
        len(episodes),
        written,
    )
    return {
        "archived": written,
        "scanned_candidates": len(candidates),
        "scanned_episodes": len(episodes),
    }


def distill_all_users(*, limit_per_user: int | None = None) -> int:
    """后台清理线程入口：按用户遍历蒸馏，返回本轮新增归档条数。"""
    users: set[str] = set()
    try:
        users.update(store.list_episode_users(enabled_only=True))
    except Exception:
        logger.exception("L4 归档：episode 用户列表读取失败")
    try:
        users.update(
            store.list_candidate_users(status="promoted", enabled_only=True)
        )
    except Exception:
        logger.exception("L4 归档：候选用户列表读取失败")
    total = 0
    for uid in sorted(users):
        batch = {"user_id": uid}
        if limit_per_user is not None:
            batch["limit"] = limit_per_user
        total += int(distill_memory(**batch).get("archived") or 0)
    return total
