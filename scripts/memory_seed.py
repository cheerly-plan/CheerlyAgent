"""造一批可复现的记忆数据，用来验证排序 / 冲突消解 / 晋升梯。

为什么需要它：渐进式记忆的门槛是「跨 3 个会话 + 命中 3 次」，真实对话要跑
好几天才攒得出来。这个脚本按同样的路径写入（走 store 的真接口，不是直接
INSERT），所以验的是**真实代码路径**，不是测试替身。

    uv run python scripts/memory_seed.py            # 造数据 + 打印体检
    uv run python scripts/memory_seed.py --clean    # 清掉这个种子用户

种子用户在 ``--user`` 指定（默认 ``seed:memory_demo``），与真实账号隔离。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_agent.core.config import settings  # noqa: E402
from ai_agent.core.logging import get_logger  # noqa: E402
from ai_agent.memory.profile import ProfileAction, apply_profile_action  # noqa: E402
from ai_agent.memory.progressive import build_memory_index  # noqa: E402
from ai_agent.memory.promotion import (  # noqa: E402
    evaluate_episode,
    review_pending_candidates,
)
from ai_agent.storage.db import user_scope  # noqa: E402
from ai_agent.storage.store import store  # noqa: E402

logger = get_logger("memory_seed")

DEFAULT_USER = "seed:memory_demo"

# (facet, 文本, 多少天前写的)
PROFILE_SEED: list[tuple[str, str, int]] = [
    ("name", "张伟", 400),
    ("occupation", "半导体行业研究员", 200),
    ("locale", "深圳", 300),
    ("preference", "喜欢简洁直接的回答，不要客套话", 30),
    ("preference", "财报数据默认用万元", 60),
    ("constraint", "回答不要用表格，用列表", 15),
]

# 会覆盖「财报数据默认用万元」的那条
CORRECTION = ("preference", "以后财报数据都用亿元", 0)

# (文本, 多少天前, 命中次数, 出现在几个会话)
# episode 的 TTL 是 7 天，超过就检索不到——最后一条故意放到 120 天前，
# 用来验证「过期即隐身」
EPISODE_SEED: list[tuple[str, int, int, int]] = [
    ("在做一个 A 股半导体设备公司的深度研报，章节按竞争格局展开", 1, 4, 3),
    ("约定深度层结论必须人工审核后再进正文", 3, 5, 3),
    ("长电科技 2026Q3 净利润对比用亿元口径，同比口径沿用报告原文", 1, 2, 1),
    ("偏好先给结论再给依据，依据用要点列表", 5, 1, 1),
    ("那个双 PDF 的解析问题已经用 OCR 兜底解决了", 120, 0, 0),
]


def _age(kind: str, table: str, item_id: str, days: int) -> None:
    """把创建时间（episode 还有 last_seen_at）推回过去，模拟真实时间跨度。"""
    extra = (
        ", last_seen_at = now() - (%s * interval '1 day')"
        if table == "agent_episode"
        else ""
    )
    params: list[object] = [days, days, item_id, _UID[0]] if extra else [days, item_id, _UID[0]]
    with store._get_pool().connection() as conn:
        conn.execute(
            f"UPDATE {table} SET created_at = now() - (%s * interval '1 day'){extra} "
            f"WHERE id::text = %s AND user_id = %s",
            params,
        )
        conn.commit()


def _set_evidence(episode_id: str, hits: int, sessions: int) -> None:
    """直接写入命中热度与会话数。

    `record_episode_hits` 有 300 秒时间窗去重（防止同一轮里两次检索把热度
    刷成双倍），脚本里连续调用只能 +1，所以造数据时直接置值。
    """
    with store._get_pool().connection() as conn:
        conn.execute(
            "UPDATE agent_episode SET hit_count = %s, sessions_seen = %s "
            "WHERE id::text = %s AND user_id = %s",
            (hits, [f"seed-session-{i}" for i in range(sessions)], episode_id, _UID[0]),
        )
        conn.commit()


_UID = [DEFAULT_USER]


def clean(user_id: str) -> None:
    with user_scope(user_id):
        store.clear_long_term(user_id=user_id)
        with store._get_pool().connection() as conn:
            conn.execute("DELETE FROM agent_memory_candidate WHERE user_id = %s", (user_id,))
            conn.execute("DELETE FROM agent_memory_archive WHERE user_id = %s", (user_id,))
            conn.commit()
    print(f"已清空 {user_id} 的画像 / 情景 / 候选 / 归档")


def seed(user_id: str) -> None:
    with user_scope(user_id):
        for facet, text, days in PROFILE_SEED:
            item = apply_profile_action(ProfileAction("add", facet, text), user_id=user_id)
            if item:
                _age("profile", "agent_user_profile", item.id, days)

        # 覆盖式更正：旧条目应被标 superseded
        facet, text, days = CORRECTION
        apply_profile_action(ProfileAction("add", facet, text), user_id=user_id)

        for text, days, hits, sessions in EPISODE_SEED:
            item = store.write_episode(text, user_id=user_id)
            _age("episode", "agent_episode", item.id, days)
            _set_evidence(item.id, hits, sessions)
            decision = evaluate_episode(
                text=item.text,
                hit_count=hits,
                session_seen=sessions,
                min_sessions=settings.memory_promotion_min_sessions,
                min_hits=settings.memory_promotion_min_hits,
            )
            if decision.stage == "candidate":
                store.upsert_candidate(
                    text=item.text,
                    hit_count=hits,
                    session_seen=sessions,
                    user_id=user_id,
                )

        promoted = review_pending_candidates(user_id=user_id)
    print(f"已写入 {user_id}：{len(PROFILE_SEED) + 1} 条画像、{len(EPISODE_SEED)} 条情景")
    if promoted:
        print(f"  自动晋升进画像：{promoted} 条")


def report(user_id: str) -> None:
    with user_scope(user_id):
        active = store.list_profile(user_id)
        allrows = store.list_profile(user_id, include_inactive=True)
        episodes = store.search_episode("研报 结论 口径", k=10, user_id=user_id)
        candidates = store.list_candidates(user_id, status="pending")

    print("\n画像（有效）")
    for item in active:
        print(f"  [{item.facet:10s}] {item.text}")
    stale = [item for item in allrows if item not in active]
    if stale:
        print("画像（已失效，留档）")
        for item in stale:
            print(f"  [{item.facet:10s}] {item.text}")

    print(f"\n情景记忆 {len(episodes)} 条（问句「研报 结论 口径」）")
    for item in episodes:
        print(f"  hits={item.hits} score={item.score:.3f} {item.text[:38]}")

    print(f"\n待审候选 {len(candidates)} 条")

    block, trace = build_memory_index(
        active, user_input="财报用什么单位", top_k=5, max_chars=800
    )
    print("\n记忆索引（问句「财报用什么单位」）")
    print(f"  mode={trace.mode} injected={len(trace.injected)} chars={trace.chars}")
    for mid, score in trace.scores:
        text = next((i.text for i in active if i.id == mid), "")
        print(f"  {score:.3f} {text[:40]}")
    del block


def main() -> int:
    parser = argparse.ArgumentParser(description="造一批可复现的记忆数据")
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--clean", action="store_true", help="先清空再写入")
    args = parser.parse_args()
    _UID[0] = args.user

    if args.clean:
        clean(args.user)
        return 0
    seed(args.user)
    report(args.user)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
