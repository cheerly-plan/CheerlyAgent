"""口径元数据：模板推定范围、金额归一为元。"""

from __future__ import annotations

from ai_agent.kb.ingest.tables import SCOPE_CONSOLIDATED, SCOPE_PARENT
from ai_agent.report.meta import infer_scope, normalize_amount, scope_label


def test_main_metrics_table_defaults_to_consolidated():
    scope, inferred = infer_scope(
        scope="",
        section="一、主要财务数据 · （一）主要会计数据和财务指标",
        column="年初至报告期末",
    )
    assert scope == SCOPE_CONSOLIDATED
    assert inferred is True
    assert "推定" in scope_label(scope, inferred=True)


def test_explicit_parent_not_overridden():
    scope, inferred = infer_scope(
        scope=SCOPE_PARENT,
        section="主要会计数据和财务指标",
        column="年初至报告期末",
    )
    assert scope == SCOPE_PARENT
    assert inferred is False


def test_column_name_alone_infers_consolidated():
    scope, inferred = infer_scope(scope="", section="", column="本报告期")
    assert scope == SCOPE_CONSOLIDATED
    assert inferred is True


def test_normalize_million_yuan_to_yuan():
    value, unit, raw, converted = normalize_amount("526,365", "百万元")
    assert converted is True
    assert unit == "元"
    assert raw == "百万元"
    assert value == "526,365,000,000"


def test_normalize_leaves_yuan_and_ratio():
    assert normalize_amount("20,297,371,376.50", "元") == (
        "20,297,371,376.50",
        "元",
        "元",
        False,
    )
    assert normalize_amount("66.22", "%")[3] is False
