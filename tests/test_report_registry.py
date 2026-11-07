"""研报统一注册表（P0 全量收敛）的 conformance 与回归用例。

竖切期的脚手架是「配置 == 内置常量逐字相等」；全量收敛后常量已全部由配置
派生，验收目标随之改成：

1. shipped 配置严格可加载，且每张导出表与配置内容自洽（round-trip）；
2. 下游模块不再持有第二份名单——它们的常量就是注册表的启动快照；
3. 改一处 YAML，所有动态读取点同步生效；回退语义不丢未登记项；
4. 各语义组保持迁移前的成员差异（金额 10 / 8 / 7 / 2 项不能悄悄统一）；
5. 非法结构、注入数字、坏正则一律 fail-fast；主配置损坏回退只读镜像。
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import yaml

from ai_agent.report import (
    basis,
    compose,
    extract,
    features,
    industry,
    market_finance,
    parse,
    pdf,
    registry,
    supplemental,
    validate,
)

_SHIPPED = Path(registry.__file__).resolve().parent / "config" / "indicators.yaml"


def _shipped_config() -> dict:
    """直接读随包发布的配置，不再手抄一份（手抄必然漂移）。"""
    return yaml.safe_load(_SHIPPED.read_text(encoding="utf-8"))


def _base_config() -> dict:
    return copy.deepcopy(_shipped_config())


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
    monkeypatch.undo()
    registry.reload_registry()


def test_shipped_config_round_trips_through_registry():
    """配置里的每条元数据都能通过注册表原样读回来。"""
    cfg = _shipped_config()
    reg = registry.validate_registry()
    assert set(reg.indicators) == set(cfg["indicators"])
    assert list(registry.aliases_map()) == [tuple(pair) for pair in cfg["aliases"]]
    assert registry.indicator_keys() == tuple(
        (item["name"], tuple(item["aliases"]))
        for item in cfg["ordering"]["indicator_keys"]
    )

    tables = registry.table_specs()
    narratives = registry.narrative_patterns()
    year_ends = dict(
        (name, (pattern, unit)) for name, pattern, unit in registry.year_end_specs()
    )
    abstracts = {item[1]: item for item in registry.abstract_map()}
    for name, item in cfg["indicators"].items():
        if item.get("table_labels"):
            for key in item.get("table_keys", []):
                assert tables[key] == (item["table_canonical"], tuple(item["table_labels"]))
        if "narrative" in item:
            assert narratives[name] == (
                item["narrative"]["keyword"],
                item["narrative"]["pattern"],
                item["narrative"]["unit"],
            )
        if "year_end" in item:
            assert year_ends[name] == (
                item["year_end"]["pattern"],
                item["year_end"]["unit"],
            )
        if item.get("abstract_column"):
            assert abstracts[name] == (
                item["abstract_column"],
                name,
                item["abstract_unit"],
            )
        for flag, value in item.get("flags", {}).items():
            assert registry.flag(name, flag, default=not value) is value


def test_downstream_constants_are_registry_snapshots():
    """下游不再持有第二份名单：模块常量 == 注册表当前值。"""
    assert industry._ALIASES == registry.aliases_map()
    assert extract._INDICATORS == registry.table_specs()
    assert extract._NARRATIVE_PATTERNS == registry.narrative_patterns()
    assert validate._MONEY_INDICATORS == registry.names_with("money_validation")
    assert compose._LEAD_BANK == registry.ordering_list("lead_bank")
    assert compose._LEAD_GENERIC == registry.ordering_list("lead_generic")
    assert compose._LEAD_MAX == registry.ordering_int("lead_max")
    assert set(basis._POINT_IN_TIME) == set(registry.names_with("point_in_time"))
    assert features._EQUITY_COMPOSITION == registry.ordering_list(
        "equity_composition_order"
    )
    assert supplemental._YEAR_END_SPECS == registry.year_end_specs()
    assert market_finance._ABSTRACT_MAP == registry.abstract_map()
    assert parse._INDICATOR_KEYS == registry.indicator_keys()


def test_semantic_groups_keep_their_distinct_members():
    """迁移前各组口径不一致，P0 只搬位置、不统一口径——否则正文会悄悄变。"""
    validation = registry.names_with("money_validation")
    wide = registry.names_with("money_wide_table")
    cell = registry.names_with("money_cell_unit")
    abstract = registry.names_with("amount_abstract")
    assert (len(validation), len(wide), len(cell), len(abstract)) == (10, 8, 7, 2)
    assert "利润总额" in validation and "利润总额" not in wide
    assert "在手订单" in wide and "在手订单" not in cell
    assert "合同负债" in cell and "合同负债" in wide
    assert "合同负债" not in registry.names_with("chart_default")


def test_all_indicators_are_registered():
    """全量收敛后不再有「未登记指标」这一过渡态。"""
    for name in ("营业收入", "毛利率", "净息差", "合同负债", "不良贷款率"):
        assert registry.spec_of(name) is not None, name
    assert registry.table_entry("营业收入") is not None
    assert registry.flag("营业收入", "money", default=False) is True
    assert registry.flag("毛利率", "money", default=True) is False
    # 没显式声明的标记仍回退调用方 default，不会因「已登记」被当成 false。
    assert registry.flag("营业收入", "ytd_ratio", default=True) is True


def test_fallback_semantics_never_drop_unregistered_names():
    """竖切期的 bug：注册表非空就丢掉 fallback，登记第 2 个指标会静默缩圈。"""
    assert "未登记指标X" in registry.names_with(
        "money_validation", fallback=("未登记指标X",)
    )
    assert registry.lead_names(fallback=("未登记指标X",)) == (
        "毛利率",
        "未登记指标X",
    )
    assert registry.lead_names(fallback=("毛利率",)) == ("毛利率",)
    assert registry.names_with("prefer_market", fallback={"毛利率"}) == frozenset(
        {"毛利率"}
    )


def test_one_yaml_change_propagates_to_all_readers(write_config):
    """核心 conformance：改一处配置，各读取点同步变化。"""
    config = _base_config()
    gross = config["indicators"]["毛利率"]
    gross["table_labels"] = ["毛利比率"]
    gross["abstract_column"] = "毛利比例"
    gross["narrative"]["keyword"] = "毛利"
    gross["flags"] = {
        "money": True,
        "chart": False,
        "lead": False,
        "prefer_market": False,
        "ytd_ratio": False,
    }
    write_config(config)

    # extract：表格行别名 + 正文关键词（动态读取点）
    assert extract._indicator_spec("毛利率") == ("毛利率", ("毛利比率",))
    assert registry.table_specs()["毛利率"] == ("毛利率", ("毛利比率",))
    assert registry.narrative_entry("毛利率")[0] == "毛利"
    # validate / pdf / compose / market_finance
    assert registry.flag("毛利率", "money", default=False) is True
    assert registry.flag("毛利率", "chart", default=True) is False
    assert registry.lead_names(fallback=("毛利率",)) == ()
    assert registry.abstract_entry("毛利率")[0] == "毛利比例"
    assert registry.flag("毛利率", "ytd_ratio", default=True) is False
    assert registry.names_with("prefer_market", fallback={"毛利率"}) == frozenset()


def test_lead_priority_comes_from_config():
    """顺序敏感的结论优先级由配置给定，注册表不能把后缀吃掉。"""
    fixed = registry.ordering_list("lead_generic")
    assert compose._LEAD_GENERIC == fixed
    assert registry.lead_names(fallback=fixed[4:]) == fixed[4:]
    assert (*fixed[:4], *registry.lead_names(fallback=fixed[4:])) == fixed


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
        lambda c: c["indicators"]["毛利率"]["flags"].update({"Bad-Flag": True}),
        lambda c: c["indicators"]["毛利率"]["flags"].update({"chart": "yes"}),
        lambda c: c["indicators"]["毛利率"]["narrative"].update({"regex2": "x"}),
        lambda c: c["indicators"]["毛利率"]["narrative"].update({"pattern": "("}),
        lambda c: c["indicators"]["毛利率"]["narrative"].update({"pattern": "a" * 501}),
        lambda c: c["indicators"]["不良贷款率"]["year_end"].update({"pattern": "("}),
        lambda c: c["classification"].update({"unknown_section": []}),
        lambda c: c["classification"].update({"name_hints": [["药业"]]}),
        lambda c: c["classification"].update(
            {"name_hints": [["药业", "not_a_type"]]}
        ),
        lambda c: c["ordering"].update({"injected": 999}),
        lambda c: c["ordering"].update({"lead_max": "3"}),
        lambda c: c["aliases"].append(["简称"]),
    ],
)
def test_validate_registry_rejects_bad_config(config_dir, mutate):
    """非法结构、未知字段、注入数字、坏正则一律 fail-fast，不静默忽略。"""
    config = _base_config()
    mutate(config)
    path = config_dir / "bad.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        registry.validate_registry(path)


def test_config_without_indicators_keeps_classification(write_config):
    """只有 classification 段时，指标侧为空并由 fallback 兜住，服务不崩。"""
    config = {"version": 1, "classification": _base_config()["classification"]}
    write_config(config)
    assert registry.spec_of("毛利率") is None
    assert registry.lead_names(fallback=("毛利率",)) == ("毛利率",)
    assert ("置业", "real_estate") in registry.name_hints()


def test_degraded_registry_falls_back_to_snapshot(config_dir, monkeypatch):
    """主配置损坏时回退只读镜像：名单仍在，行为不退化，服务不崩。"""
    monkeypatch.setattr(registry, "_INDICATORS_YAML", config_dir / "missing.yaml")
    registry.reload_registry()
    assert registry.spec_of("毛利率") is not None
    assert registry.table_specs() == extract._INDICATORS
    assert registry.names_with("money_validation") == validate._MONEY_INDICATORS
    assert registry.flag("毛利率", "chart", default=False) is True
    assert industry.classify_subject("某某置业股份有限公司") == industry.REAL_ESTATE
    assert industry.type_for_segment("设计") == industry.FABLESS
    monkeypatch.undo()
    registry.reload_registry()


def test_snapshot_matches_shipped_config():
    """降级镜像与主配置同一次生成，不能各自漂移。"""
    from ai_agent.report.registry import _DEFAULTS_YAML, _load

    assert _load(_DEFAULTS_YAML).indicators.keys() == _load(_SHIPPED).indicators.keys()
    assert _load(_DEFAULTS_YAML).aliases == _load(_SHIPPED).aliases
