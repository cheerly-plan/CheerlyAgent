"""研报指标注册表（配置化竖切）的 conformance 与回归用例。

竖切阶段的验收目标不是「配置能加载」，而是两件事：
1. 配置与内置常量逐字对齐，所以接入注册表后报告正文、校验、图表行为不变；
2. 改一份 YAML 的某个字段，抽取 / 校验 / 图表 / 结论 / 摘要映射同步生效，
   且未登记的指标、配置损坏时都能回退到原有常量。
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import yaml

from ai_agent.report import (
    compose,
    extract,
    industry,
    market_finance,
    registry,
    validate,
)


def _base_config() -> dict:
    """与 config/indicators.yaml 等价的一份可改配置。"""
    return {
        "version": 1,
        "indicators": {
            "毛利率": {
                "table_canonical": "毛利率",
                "table_labels": ["毛利率", "综合毛利率"],
                "narrative": {
                    "keyword": "毛利率",
                    "pattern": r"毛利率为?\s*([\d,]+(?:\.\d+)?)\s*%",
                    "unit": "%",
                },
                "abstract_column": "毛利率",
                "abstract_unit": "%",
                "flags": {
                    "money": False,
                    "chart": True,
                    "lead": True,
                    "prefer_market": True,
                    "ytd_ratio": True,
                },
                "lead_order": 4,
            }
        },
        "classification": {
            "name_hints": [
                ["银行", "bank"],
                ["置业", "real_estate"],
                ["地产", "real_estate"],
                ["软件", "software"],
                ["钢铁", "cyclical"],
            ],
            "segment_to_type": {
                "设计": "fabless",
                "制造": "foundry",
                "设备": "equipment",
                "封测": "packaging",
            },
        },
    }


@pytest.fixture(autouse=True)
def _clean_registry_cache():
    """每个用例前后清缓存，避免临时配置串到别的用例。"""
    registry.reload_registry()
    yield
    registry.reload_registry()


@pytest.fixture
def config_dir():
    """沙箱里 pytest 的 tmp_path 会撞到已存在的根目录，改用标准库临时目录。"""
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


@pytest.fixture
def write_config(config_dir, monkeypatch):
    """把注册表指向一份临时配置并清缓存；测试结束恢复默认配置。"""

    def _write(config: dict):
        path = config_dir / "indicators.yaml"
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        monkeypatch.setattr(registry, "_INDICATORS_YAML", path)
        registry.reload_registry()
        return path

    yield _write
    registry.reload_registry()


def test_registry_matches_legacy_constants():
    """配置与内置常量逐字一致 —— 这是「接入后行为不变」的直接证据。"""
    reg = registry.validate_registry()
    assert set(reg.indicators) == {"毛利率"}
    assert registry.table_entry("毛利率") == extract._INDICATORS["毛利率"]
    assert (
        registry.narrative_entry("毛利率")
        == extract._NARRATIVE_PATTERNS["毛利率"]
    )
    assert registry.abstract_entry("毛利率") == next(
        item for item in market_finance._ABSTRACT_MAP if item[1] == "毛利率"
    )
    assert registry.flag("毛利率", "money", default=True) is False
    assert registry.flag("毛利率", "chart", default=False) is True
    assert registry.flag("毛利率", "ytd_ratio", default=False) is True
    assert registry.lead_names() == ("毛利率",)
    assert registry.names_with("prefer_market") == (
        market_finance._SEMICONDUCTOR_PREFER_MARKET
    )
    # compose 的优先级 = 固定前四项 + 注册表派生后缀，与原常量一致
    assert (
        *compose._LEAD_GENERIC[:4],
        *registry.lead_names(fallback=compose._LEAD_GENERIC[4:]),
    ) == compose._LEAD_GENERIC
    # 分类规则确实来自 YAML，不是模块常量
    assert ("置业", "real_estate") in registry.name_hints()
    assert registry.segment_map()["设计"] == "fabless"


def test_one_yaml_change_propagates_to_all_readers(write_config):
    """核心 conformance：改一处配置，六个下游读取点同步变化。"""
    config = _base_config()
    config["indicators"]["毛利率"].update(
        {
            "table_labels": ["毛利比率"],
            "abstract_column": "毛利比例",
            "flags": {
                "money": True,
                "chart": False,
                "lead": False,
                "prefer_market": False,
                "ytd_ratio": False,
            },
        }
    )
    config["indicators"]["毛利率"]["narrative"]["keyword"] = "毛利"
    write_config(config)

    # extract：表格行别名与正文关键词
    assert extract._indicator_spec("毛利率") == ("毛利率", ("毛利比率",))
    assert registry.narrative_entry("毛利率")[0] == "毛利"
    # validate：金额标记
    assert registry.flag("毛利率", "money", default=False) is True
    # pdf：图表白名单
    assert registry.flag("毛利率", "chart", default=True) is False
    # compose：本期结论优先级
    assert registry.lead_names(fallback=("毛利率",)) == ()
    # market_finance：摘要映射与半导体覆盖集合
    assert registry.abstract_entry("毛利率")[0] == "毛利比例"
    assert registry.flag("毛利率", "ytd_ratio", default=True) is False
    assert registry.names_with(
        "prefer_market", fallback={"毛利率"}
    ) == frozenset()


def test_unregistered_indicators_fall_back_to_constants():
    """未登记的指标继续走原有常量，竖切期行为逐字不变。"""
    assert registry.spec_of("营业收入") is None
    assert extract._indicator_spec("营业收入") == extract._INDICATORS["营业收入"]
    assert registry.table_entry("营业收入") is None
    assert registry.narrative_entry("营业收入") is None
    assert registry.abstract_entry("营业收入") is None
    assert registry.flag("营业收入", "chart", default=True) is True
    assert registry.flag(
        "营业收入", "money", default="营业收入" in validate._MONEY_INDICATORS
    ) is True


def test_classification_rules_come_from_config(write_config):
    """分类规则外置：改 YAML 就能改分类，不需要动 industry.py。"""
    assert industry.classify_subject("某某置业股份有限公司") == industry.REAL_ESTATE
    assert industry.type_for_segment("设计") == industry.FABLESS

    config = _base_config()
    config["classification"]["name_hints"] = [["药业", "software"]]
    config["classification"]["segment_to_type"]["设计"] = "packaging"
    write_config(config)

    assert industry.classify_subject("某某药业股份有限公司") == industry.SOFTWARE
    assert industry.classify_subject("某某置业股份有限公司") == industry.GENERIC
    assert industry.type_for_segment("设计") == industry.PACKAGING


def test_reload_picks_up_changes_without_restart(write_config):
    path = write_config(_base_config())
    assert registry.lead_names() == ("毛利率",)
    config = _base_config()
    config["indicators"]["毛利率"]["flags"]["lead"] = False
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    assert registry.lead_names() == ("毛利率",)  # 缓存未清，仍读旧快照
    registry.reload_registry()
    assert registry.lead_names() == ()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update({"version": 2}),
        lambda c: c.update({"unexpected_top": 1}),
        lambda c: c["indicators"]["毛利率"].update({"value": 999999}),
        lambda c: c["indicators"]["毛利率"].update({"unknown_field": "x"}),
        lambda c: c["indicators"]["毛利率"]["flags"].update({"unknown_flag": True}),
        lambda c: c["indicators"]["毛利率"]["flags"].update({"chart": "yes"}),
        lambda c: c["indicators"]["毛利率"]["narrative"].update({"regex2": "x"}),
        lambda c: c["indicators"]["毛利率"]["narrative"].update({"pattern": "("}),
        lambda c: c["indicators"]["毛利率"]["narrative"].update({"pattern": "a" * 501}),
        lambda c: c["classification"].update({"unknown_section": []}),
        lambda c: c["classification"].update({"name_hints": [["药业"]]}),
        lambda c: c["classification"].update(
            {"name_hints": [["药业", "not_a_type"]]}
        ),
    ],
)
def test_validate_registry_rejects_bad_config(config_dir, mutate):
    """非法结构、未知字段、注入数字、坏正则一律 fail-fast，不静默忽略。"""
    config = copy.deepcopy(_base_config())
    mutate(config)
    path = config_dir / "bad.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        registry.validate_registry(path)


def test_config_without_indicators_keeps_builtin_classification(write_config):
    """只写 classification 段时，指标侧回退内置常量，分类仍按配置走。"""
    config = {"version": 1, "classification": _base_config()["classification"]}
    write_config(config)
    assert registry.spec_of("毛利率") is None
    assert registry.lead_names(fallback=("毛利率",)) == ("毛利率",)
    assert ("置业", "real_estate") in registry.name_hints()


def test_degraded_registry_keeps_legacy_behavior(config_dir, monkeypatch):
    """配置文件缺失时降级：所有读取点回退内置常量，服务不因配置坏掉而崩。"""
    monkeypatch.setattr(registry, "_INDICATORS_YAML", config_dir / "missing.yaml")
    registry.reload_registry()
    assert registry.spec_of("毛利率") is None
    assert extract._indicator_spec("毛利率") == extract._INDICATORS["毛利率"]
    assert registry.flag("毛利率", "chart", default=True) is True
    assert registry.lead_names(fallback=("毛利率",)) == ("毛利率",)
    assert registry.names_with("prefer_market", fallback={"毛利率"}) == frozenset(
        {"毛利率"}
    )
    assert industry.classify_subject("某某置业股份有限公司") == industry.REAL_ESTATE
    assert industry.type_for_segment("设计") == industry.FABLESS
