"""轻量链路计数器：让静默降级在 /health 上看得见。"""

from ai_agent.llm import metrics


def test_counters_accumulate_and_hide_zeros():
    metrics.reset()
    metrics.incr(metrics.MEMORY_ZERO)
    metrics.incr(metrics.MEMORY_ZERO, 2)
    metrics.incr(metrics.SUMMARY_OK)
    snap = metrics.snapshot()
    assert snap == {"memory.extract.zero": 3, "summary.ok": 1}
    metrics.reset()
    assert metrics.snapshot() == {}


def test_incr_ignores_non_positive():
    metrics.reset()
    metrics.incr(metrics.TITLE_OK, 0)
    metrics.incr(metrics.TITLE_OK, -5)
    assert metrics.snapshot() == {}


def test_lightweight_health_reports_counters():
    """降级计数单独一个端点：/health 是存活探针，输出必须稳定可断言。"""
    from ai_agent.api.routers.health import lightweight_health

    metrics.reset()
    metrics.incr(metrics.MEMORY_ZERO)
    assert lightweight_health() == {"degraded": {"memory.extract.zero": 1}}
    metrics.reset()

    # 没有降级时不返回空字典以外的噪音
    assert lightweight_health() == {"degraded": {}}


def test_serialize_toggle_defaults_on():
    from ai_agent.core.config import settings

    assert settings.local_llm_serialize is True
