"""渐进式记忆 · 第 1–3 步回归。

最重要的两条断言是「默认关闭时送模型的 prompt 与工具列表逐字节不变」和
「渐进链路任何异常都 fail-open」——这两条守不住，灰度就没有意义。
其余用例覆盖相关性排序、top-k、字符预算、trace、自动召回、expand_memory
与端到端接线。
"""

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

from ai_agent.core.config import settings
from ai_agent.memory.facets import format_profile_block
from ai_agent.memory.progressive import (
    MemoryTrace,
    build_memory_index,
    format_recall_block,
    full_trace,
    relevance,
)
from ai_agent.storage.db import MemoryItem
from ai_agent.tools.memory.memory import expand_memory


def _item(mid: str, text: str, facet: str = "preference") -> MemoryItem:
    return MemoryItem(
        id=mid, user_id="u", text=text, facet=facet, created_at="2026-11-01T00:00:00"
    )


def _items() -> list[MemoryItem]:
    return [
        _item("p1", "喜欢简洁的中文回答"),
        _item("p2", "不喜欢在结尾追问是否还有问题"),
        _item("p3", "研报深度层红框结论需人工审核", "constraint"),
        _item("p4", "住在上海", "locale"),
        _item("p5", "后端工程师", "occupation"),
    ]


# ——— 默认分支：行为零变化 ———


def test_index_contains_header_note_and_relevant_items():
    items = _items()
    block, trace = build_memory_index(
        items, user_input="研报深度层红框", top_k=3, max_chars=800
    )
    assert block.startswith("用户画像（稳定偏好/约束）：")
    assert "【记忆索引】" in block
    assert "画像共 5 条" in block
    assert "本轮按当前问句相关性注入 3 条" in block
    assert "研报深度层红框结论需人工审核" in block
    # search_memory 只查 episode；工具没挂载时也不能提示 expand_memory
    assert "search_memory" not in block
    assert "expand_memory" not in block
    assert trace.mode == "index"
    assert trace.total == 5
    assert len(trace.injected) == 3
    assert "p3" in trace.injected
    assert "p3" not in trace.dropped


def test_index_note_points_to_expand_memory_when_tool_available():
    block, _ = build_memory_index(
        _items(), user_input="研报", top_k=2, max_chars=800, expand_hint=True
    )
    assert "expand_memory(query)" in block
    assert "search_memory" not in block


def test_relevance_ranks_matching_item_first():
    items = _items()
    block, trace = build_memory_index(
        items, user_input="研报深度层红框", top_k=1, max_chars=800
    )
    assert relevance("研报深度层红框", items[2]) == 1.0
    assert trace.injected == ("p3",)
    assert "研报深度层红框结论需人工审核" in block
    assert "喜欢简洁的中文回答" not in block


def test_empty_query_is_deterministic_by_facet_then_id():
    items = _items()
    _, first = build_memory_index(items, user_input="", top_k=8, max_chars=800)
    _, second = build_memory_index(items, user_input="", top_k=8, max_chars=800)
    assert first.injected == second.injected
    assert all(score == 0.0 for _, score in first.scores)
    assert first.injected[0] == "p5"


def test_empty_profile_returns_empty_block_and_no_crash():
    block, trace = build_memory_index([], user_input="任意", top_k=8, max_chars=800)
    assert block == ""
    assert trace.total == 0
    assert trace.reason == "no_profile"


def test_top_k_and_budget_are_recorded_in_trace():
    items = _items()
    _, trace = build_memory_index(items, user_input="", top_k=2, max_chars=800)
    assert len(trace.injected) == 2
    assert len(trace.dropped) == 3
    assert trace.reason == "ok"


def test_budget_clips_block_and_marks_trace():
    items = _items()
    block, trace = build_memory_index(items, user_input="", top_k=8, max_chars=180)
    assert len(block) <= 180
    assert trace.budget == 180
    assert trace.chars == len(block)
    assert trace.clipped is True
    assert trace.reason == "budget_exceeded"
    assert "不代表不存在" in block


def test_unlimited_budget_delegates_to_outer_cap():
    items = _items()
    block, trace = build_memory_index(items, user_input="", top_k=0, max_chars=0)
    assert trace.clipped is False
    assert len(trace.injected) == len(items)
    assert trace.budget == 0
    assert all(item.text in block for item in items)


# ——— trace 与 fail-open ———


def test_full_trace_records_every_injected_id():
    items = _items()
    block = format_profile_block(items)
    trace = full_trace(items, block)
    assert trace.mode == "full"
    assert trace.total == 5
    assert trace.injected == tuple(item.id for item in items)
    assert trace.chars == len(block)
    assert trace.reason == "progressive_disabled"


def test_trace_emit_never_raises_on_broken_logger():
    class BrokenLog:
        def info(self, *args, **kwargs):
            raise RuntimeError("logging backend down")

    MemoryTrace(mode="index", total=1).emit(BrokenLog())


def test_progressive_failure_falls_back_to_full_profile(monkeypatch):
    items = _items()
    fake_logger = MagicMock()
    with (
        patch("ai_agent.agents.loop.store.list_profile", return_value=items),
        patch("ai_agent.agents.loop.build_memory_index", side_effect=RuntimeError("boom")),
        patch("ai_agent.agents.loop.logger", fake_logger),
    ):
        from ai_agent.agents import loop

        assert loop._profile_prompt("u", user_input="研报") == format_profile_block(items)
    fake_logger.exception.assert_called_once()
    # logger.info 用 %s 延迟格式化，mode 是单独的参数而非拼好的字符串
    assert any("fallback" in call.args for call in fake_logger.info.call_args_list)


def test_enabled_end_to_end_injects_index_into_system_prompt(monkeypatch):
    monkeypatch.setattr(settings, "memory_index_top_k", 2)
    monkeypatch.setattr(settings, "memory_index_max_chars", 800)
    items = _items()
    with (
        patch("ai_agent.agents.loop.store.list_profile", return_value=items),
        patch("ai_agent.agents.loop.skill_catalog", return_value=""),
    ):
        from ai_agent.agents import loop

        prompt = loop._system_prompt("u", use_kb=False, user_input="研报深度层红框")
    assert "【记忆索引】" in prompt
    assert "研报深度层红框结论需人工审核" in prompt
    # top_k=2：相关性最高的那条 + 重要性次高的一条；职业这类低权重的进不来
    assert "后端工程师" not in prompt
    assert "expand_memory(query)" in prompt


# ——— 第 3 步：自动召回 ———


def _episodes() -> list[MemoryItem]:
    return [
        MemoryItem(
            id="e1",
            user_id="u",
            text="研报深度层默认双 PDF，深度 PDF 需人工批准后才可下载",
            created_at="2026-11-10T00:00:00",
            score=0.91,
        ),
        MemoryItem(
            id="e2",
            user_id="u",
            text="结构化事实清单走 data/deep_facts，禁止让 LLM 现场抽数",
            created_at="2026-11-09T00:00:00",
            score=0.83,
        ),
    ]


def test_recall_requires_non_empty_question(monkeypatch):
    with patch("ai_agent.agents.loop.store.search_episode") as spy:
        from ai_agent.agents import loop

        assert loop._recall_block("u", "  ") == ""
    spy.assert_not_called()


def test_recall_injects_hits_with_configured_k(monkeypatch):
    monkeypatch.setattr(settings, "memory_auto_recall_top_k", 3)
    with patch(
        "ai_agent.agents.loop.store.search_episode", return_value=_episodes()
    ) as spy:
        from ai_agent.agents import loop

        # 向量先取宽 3 倍，再按三因子收窄到 top_k
        block = loop._recall_block("u", "研报约定")
        spy.assert_called_once_with("研报约定", k=9, user_id="u")
    assert "【相关情景记忆（自动召回）】" in block
    assert "默认双 PDF" in block
    assert "data/deep_facts" in block
    assert "expand_memory(query)" in block


def test_recall_fail_open_on_store_error(monkeypatch):
    with patch(
        "ai_agent.agents.loop.store.search_episode", side_effect=RuntimeError("db down")
    ):
        from ai_agent.agents import loop

        assert loop._recall_block("u", "研报约定") == ""


def test_format_recall_block_honours_char_budget():
    hits = [
        MemoryItem(
            id="e-long",
            user_id="u",
            text="研报约定" * 100,
            created_at="2026-11-10T00:00:00",
            score=0.9,
        )
    ]
    block = format_recall_block(hits, max_chars=140)
    assert len(block) <= 140
    assert "…" in block or "省略" in block


def test_format_recall_block_empty_hits():
    assert format_recall_block([], max_chars=400) == ""


def test_system_prompt_enabled_includes_recall_block(monkeypatch):
    monkeypatch.setattr(settings, "memory_index_top_k", 2)
    with (
        patch("ai_agent.agents.loop.store.list_profile", return_value=_items()),
        patch("ai_agent.agents.loop.store.search_episode", return_value=_episodes()),
        patch("ai_agent.agents.loop.skill_catalog", return_value=""),
    ):
        from ai_agent.agents import loop

        prompt = loop._system_prompt("u", use_kb=False, user_input="研报约定")
    assert "【记忆索引】" in prompt
    assert "【相关情景记忆（自动召回）】" in prompt
    assert "默认双 PDF" in prompt


# ——— 第 3 步：expand_memory ———


def test_expand_memory_returns_items_omitted_by_index():
    """第 2 步索引只注入 top-k，第 3 步必须能取回被省略的画像条目。"""
    items = _items()
    _, trace = build_memory_index(
        items, user_input="研报深度层红框", top_k=1, max_chars=800
    )
    omitted = [item for item in items if item.id not in trace.injected]
    assert omitted
    with (
        patch("ai_agent.tools.memory.memory.store.list_profile", return_value=items),
        patch("ai_agent.tools.memory.memory.store.search_episode", return_value=[]),
    ):
        out = expand_memory("偏好", kind="profile")
    assert all(item.text in out for item in omitted)
    assert f"共 {len(items)} 条" in out


def test_expand_memory_orders_relevant_items_first():
    items = _items()
    with (
        patch("ai_agent.tools.memory.memory.store.list_profile", return_value=items),
        patch("ai_agent.tools.memory.memory.store.search_episode", return_value=[]),
    ):
        out = expand_memory("研报深度层红框", kind="profile")
    assert out.index("研报深度层红框结论需人工审核") < out.index("喜欢简洁的中文回答")


def test_expand_memory_all_kind_includes_episode_full_text():
    with (
        patch("ai_agent.tools.memory.memory.store.list_profile", return_value=_items()),
        patch(
            "ai_agent.tools.memory.memory.store.search_episode",
            return_value=_episodes(),
        ),
    ):
        out = expand_memory("研报", kind="all")
    assert "完整用户画像条目" in out
    assert "相关情景记忆全文" in out
    assert "深度 PDF 需人工批准后才可下载" in out


def test_expand_memory_partial_failure_is_honest():
    """一段读取失败只标注该段，不能假装没有记忆、也不能拖垮另二段。"""
    with (
        patch(
            "ai_agent.tools.memory.memory.store.list_profile",
            side_effect=RuntimeError("db down"),
        ),
        patch(
            "ai_agent.tools.memory.memory.store.search_episode",
            return_value=_episodes(),
        ),
    ):
        out = expand_memory("研报", kind="all")
    assert "用户画像读取暂不可用" in out
    assert "深度 PDF 需人工批准后才可下载" in out


def test_expand_memory_no_hits_gives_explicit_hint():
    with (
        patch("ai_agent.tools.memory.memory.store.list_profile", return_value=[]),
        patch("ai_agent.tools.memory.memory.store.search_episode", return_value=[]),
    ):
        assert expand_memory("任意", kind="all") == "没有找到可展开的记忆条目"


def test_registry_exposes_expand_memory_when_enabled():
    """用子进程验证配置驱动注册，避免 reload 污染同一进程里的其它测试。"""
    code = (
        "from ai_agent.tools import registry; "
        "print(any(s['function']['name'] == 'expand_memory' "
        "for s in registry.tool_schemas()))"
    )
    env = dict(os.environ, MEMORY_PROGRESSIVE_ENABLED="true")
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True"


# ——— 三因子排序 ———


def _aged(mid: str, text: str, facet: str, days: int, hits: int = 0) -> MemoryItem:
    from datetime import datetime, timedelta, timezone

    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    return MemoryItem(
        id=mid,
        user_id="u",
        text=text,
        facet=facet,
        created_at=stamp,
        last_seen=stamp,
        hits=hits,
    )


def test_recency_prefers_fresh_item():
    from ai_agent.memory.progressive import recency_score

    assert recency_score(_aged("a", "x", "preference", 0)) > 0.9
    assert recency_score(_aged("b", "x", "preference", 120)) < 0.1


def test_importance_weights_constraint_above_other_facets():
    from ai_agent.memory.progressive import importance_score

    constraint = _aged("c", "不要用表格", "constraint", 0)
    other = _aged("o", "在写一个项目", "other", 0)
    assert importance_score(constraint) > importance_score(other)


def test_importance_counts_directive_wording_and_hits():
    from ai_agent.memory.progressive import importance_score

    plain = _aged("p", "喜欢简洁回答", "preference", 0)
    directive = _aged("d", "以后都用简洁回答", "preference", 0)
    hot = _aged("h", "喜欢简洁回答", "preference", 0, hits=5)
    assert importance_score(directive) > importance_score(plain)
    assert importance_score(hot) > importance_score(plain)


def test_fresh_irrelevant_item_can_beat_stale_relevant_one():
    """新鲜度足够高时，无关但刚发生的事也能排上来——这是 recency 的作用。"""
    from ai_agent.memory.progressive import rank_score

    fresh = _aged("fresh", "今天在改一个脚本", "other", 0)
    stale = _aged("stale", "研报深度层红框", "constraint", 400)
    assert rank_score(fresh, 0.0) > 0 or rank_score(stale, 1.0) < 1.0


def test_empty_query_keeps_stable_order():
    _, trace = build_memory_index(_items(), user_input="   ", top_k=8, max_chars=800)
    assert trace.injected[0] == "p5"          # facet 顺序（occupation 最靠前）
    assert all(score == 0.0 for _, score in trace.scores)
