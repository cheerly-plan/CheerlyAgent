"""长期记忆噪声回归。

样本直接取自真实库里已经写坏的行（2026-11 生产数据）：
一次性查询、任务状态、文档事实、由查询推断的身份、短期行程写成居住地。
这些句子一旦入库会随画像/召回长期注入，必须在抽取阶段挡住。
"""

import pytest

from ai_agent.memory.long_term import ExtractedMemory, _keep_extracted


def _episode(text: str) -> ExtractedMemory:
    return ExtractedMemory(kind="episode", content=text)


def _profile(text: str, facet: str = "other") -> ExtractedMemory:
    return ExtractedMemory(kind="profile", content=text, op="add", facet=facet)


@pytest.mark.parametrize(
    "text",
    [
        "用户正在确认中大力德（002896）的财务数据，已知该公司披露了2025Q3至2026Q3的财报。",
        "用户在查询中国银行的股东信息，已确认该行截至2026年第三季度的前十名普通股股东持股情况。",
        "用户正在生成绿的谐波2026Q3财报深度分析PDF研报，任务已提交至后台（job_id=32b54af91a4e），预计5-8分钟完成。",
        "用户已确认中科飞测证券代码为688361，用于后续分析通信及算力硬件板块领涨公司的财务数据。",
        "用户想了解A股最新板块走势，确认通信及算力硬件相关板块在11月11日领涨。",
        "用户正在对比中国银行和中国黄金集团的总资产，已确认中国银行总资产为411,084.49亿元。",
        "我下周要去深圳出差，报告里提到华南业务时可以提醒我。",
        "下周去杭州开会，顺便看看那边的客户。",
    ],
)
def test_ongoing_queries_and_jobs_are_dropped(text):
    assert not _keep_extracted(_episode(text))


@pytest.mark.parametrize(
    "text",
    [
        "我在做一个 A 股通信板块的研报系列，计划覆盖十家公司。",
        "正在学习 Node.js BFF，用 Koa 做数据聚合。",
        "我在整理一套财务报表模板，后续会持续使用。",
    ],
)
def test_real_episodes_are_kept(text):
    assert _keep_extracted(_episode(text))


def test_trip_is_not_locale():
    assert not _keep_extracted(_profile("深圳", facet="locale"))
    assert _keep_extracted(_profile("我住在深圳，工作在南山", facet="locale"))


def test_querying_a_company_does_not_make_it_occupation():
    assert not _keep_extracted(
        _profile("中大力德财务数据分析师（专注2026年第三季度财报对比）", facet="occupation")
    )
    assert _keep_extracted(_profile("我是中大力德的财务分析师", facet="occupation"))


def test_default_language_is_not_worth_storing():
    assert not _keep_extracted(_profile("中文", facet="language"))
    assert _keep_extracted(_profile("用英文回答", facet="language"))


def test_job_status_never_becomes_profile():
    assert not _keep_extracted(
        _profile("用户正在生成绿的谐波研报，job_id=32b54af91a4e", facet="other")
    )


def test_delete_intent_is_still_allowed():
    assert _keep_extracted(
        ExtractedMemory(kind="profile", content="深圳", op="delete", facet="locale")
    )


def test_transient_text_classifier_matches_filters():
    from ai_agent.memory.long_term import is_transient_text

    assert is_transient_text("用户正在确认中国银行的股东信息")
    assert is_transient_text("job_id=32b54af91a4e，预计5-8分钟完成")
    assert is_transient_text("下周去深圳出差")
    assert not is_transient_text("我在做一个 A 股通信板块的研报系列")
    assert not is_transient_text("财报数字以亿元为单位")


def test_transient_memory_is_demoted_in_recall(monkeypatch):
    """历史库里的噪声行不能靠 hit 热度霸榜：同等相关性下排在常青事实后面。"""
    from ai_agent.core.config import settings
    from ai_agent.memory.progressive import rerank_items
    from ai_agent.storage.db import MemoryItem

    def item(id_: str, text: str, hits: int) -> MemoryItem:
        return MemoryItem(
            id=id_,
            user_id="u1",
            text=text,
            created_at="2026-11-14T00:00:00+00:00",
            last_seen="2026-11-14T00:00:00+00:00",
            hits=hits,
        )

    noisy = item("noisy", "用户正在确认中大力德的财务数据", 55)
    evergreen = item("stable", "财报数字以亿元为单位", 1)

    monkeypatch.setattr(settings, "memory_transient_penalty", 0.3)
    ranked = rerank_items([noisy, evergreen], "财报数字")
    assert ranked[0].id == "stable"


def test_transient_penalty_zero_keeps_heat_ordering(monkeypatch):
    from ai_agent.core.config import settings
    from ai_agent.memory.progressive import rank_score
    from ai_agent.storage.db import MemoryItem

    def item(id_: str, text: str, hits: int) -> MemoryItem:
        return MemoryItem(
            id=id_,
            user_id="u1",
            text=text,
            created_at="2026-11-14T00:00:00+00:00",
            last_seen="2026-11-14T00:00:00+00:00",
            hits=hits,
        )

    noisy = item("noisy", "用户正在确认中大力德的财务数据", 55)
    monkeypatch.setattr(settings, "memory_transient_penalty", 0.0)
    assert rank_score(noisy, 1.0) == 0.0
    monkeypatch.setattr(settings, "memory_transient_penalty", 1.0)
    assert rank_score(noisy, 1.0) > 0.0
