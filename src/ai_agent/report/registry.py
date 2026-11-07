"""研报统一注册表：指标元数据、别名、具名集合、有序名单、分类规则的唯一入口。

P0 全量收敛后的不变量：

* **单一数据源。** 所有指标名单、别名、抽取规则、结论优先级都在
  ``config/indicators.yaml``；下游模块不再各自维护一份常量。新增一个指标
  = 改这一份配置（必要时写一个 renderer），不必在 8+ 个文件里补名单。
* **配置只描述意图，不执行逻辑。** 这里只有别名、正则、成员资格、顺序和
  少量标量参数；抽数 / 同比环比 / 数字校验 / PDF 渲染仍是确定性 Python，
  配置不能注入正文数字或替换校验门禁。
* **回退语义统一。** ``flag`` 是「显式声明优先、未声明回退 default」；
  ``names_with`` / ``lead_names`` 同理：fallback 里**未登记**的项一定保留，
  已登记的按配置决定。竖切版「注册表非空就丢掉 fallback」会让登记第 2 个
  指标时静默缩小集合，这里已修掉。
* **读坏不拖垮服务，但也不静默。** 主配置损坏记 exception 日志并回退到
  ``indicators_defaults.yaml``——那是生成脚本与主配置同一次导出的只读镜像，
  两份不会各自漂移。``validate_registry()`` 是显式 fail-fast 入口，供
  CI / 启动 lint 调用。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ai_agent.core.logging import get_logger

logger = get_logger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent / "config"
_INDICATORS_YAML = _CONFIG_DIR / "indicators.yaml"
_DEFAULTS_YAML = _CONFIG_DIR / "indicators_defaults.yaml"

# 严格 schema：任何未声明字段一律 fail-fast，保证「配置只能描述意图」。
_ALLOWED_TOP_LEVEL = frozenset(
    {"version", "indicators", "ordering", "classification", "aliases"}
)
_ALLOWED_INDICATOR_FIELDS = frozenset(
    {
        "table_keys",
        "table_canonical",
        "table_labels",
        "narrative",
        "year_end",
        "abstract_column",
        "abstract_unit",
        "flags",
        "lead_order",
    }
)
_ALLOWED_NARRATIVE_FIELDS = frozenset({"keyword", "pattern", "unit"})
_ALLOWED_YEAR_END_FIELDS = frozenset({"pattern", "unit"})
_ALLOWED_CLASSIFICATION_FIELDS = frozenset({"name_hints", "segment_to_type"})
_FLAG_KEY_RE = re.compile(r"[a-z][a-z0-9_]*\Z")
_MAX_PATTERN_CHARS = 500

# ordering 是有序名单 / 标量参数的白名单。数字只能出现在已声明的参数里，
# 不能借配置往正文里塞数值。
_ORDERING_TYPES: dict[str, type | tuple[type, ...]] = {
    "year_end_order": list,
    "abstract_order": list,
    "equity_composition_order": list,
    "asset_drivers_order": list,
    "balance_move_names_order": list,
    "balance_year_end_rows": list,
    "balance_search_keywords": list,
    "year_end_reason_rows": list,
    "year_end_reason_keywords": list,
    "lead_bank": list,
    "lead_generic": list,
    "lead_operating": list,
    "lead_net_profit": list,
    "lead_max": int,
    "pdf_default_indicators": list,
    "market_default_indicators": list,
    "chart_defaults": list,
    "indicator_keys": list,
    "segment_table_core": int,
    "ocf": str,
    "rnd_ratio_indicator": str,
    "money_scale_units": list,
    "yoy_frame": list,
}

# 分类规则的降级快照：由 scripts/sync_indicators_snapshot.py 从配置自动生成，
# 不再是手工维护的第三份名单。保留最后一层 try/except，打包漏文件时也不至于
# 让整个 report 包无法导入。
try:
    from ai_agent.report.classification_defaults import (
        COMPANY_TYPES as _BUILTIN_COMPANY_TYPES,
        NAME_HINTS as _BUILTIN_NAME_HINTS,
        SEGMENT_TO_TYPE as _BUILTIN_SEGMENT_TO_TYPE,
    )
except ImportError:  # pragma: no cover - 仅打包缺生成物时触发
    _BUILTIN_COMPANY_TYPES = frozenset({"bank", "generic"})
    _BUILTIN_NAME_HINTS: tuple[tuple[str, str], ...] = (("银行", "bank"),)
    _BUILTIN_SEGMENT_TO_TYPE: tuple[tuple[str, str], ...] = ()

_INDUSTRIES_YAML = _CONFIG_DIR / "industries.yaml"


def _known_company_types() -> frozenset[str]:
    """公司类型白名单的唯一数据源是 industries.yaml 的 profile keys。

    刻意不读生成物常量：否则「新增行业 → 旧白名单拒绝新 classification」会变成
    先有鸡还是先有蛋。只有连 industries.yaml 都读不到时才回退内置快照。
    """
    try:
        raw = yaml.safe_load(_INDUSTRIES_YAML.read_text(encoding="utf-8")) or {}
        profiles = raw.get("profiles") or {}
        if isinstance(profiles, dict) and profiles:
            return frozenset(str(key) for key in profiles)
    except Exception:
        logger.exception("industries.yaml 读取失败，公司类型回退内置快照")
    return _BUILTIN_COMPANY_TYPES


_MIN_ORDERING: dict[str, Any] = {"lead_max": 3, "segment_table_core": 6}


@dataclass(frozen=True)
class IndicatorSpec:
    """一个指标的元数据。flags 里只存显式声明过的布尔值（含 false）。"""

    name: str
    table_canonical: str
    table_keys: tuple[str, ...]
    table_labels: tuple[str, ...]
    narrative: tuple[str, str, str] | None
    year_end: tuple[str, str] | None
    abstract_column: str
    abstract_unit: str
    flags: dict[str, bool]
    lead_order: int

    def has(self, flag: str) -> bool:
        return bool(self.flags.get(flag, False))

    def declares(self, flag: str) -> bool:
        return flag in self.flags


@dataclass(frozen=True)
class ClassificationRules:
    name_hints: tuple[tuple[str, str], ...]
    segment_to_type: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Registry:
    indicators: dict[str, IndicatorSpec] = field(default_factory=dict)
    aliases: tuple[tuple[str, str], ...] = ()
    ordering: dict[str, Any] = field(default_factory=lambda: dict(_MIN_ORDERING))
    classification: ClassificationRules = field(
        default_factory=lambda: ClassificationRules(
            _BUILTIN_NAME_HINTS, _BUILTIN_SEGMENT_TO_TYPE
        )
    )


_EMPTY = Registry()


def _reject_unknown(value: dict[str, Any], allowed: Iterable[str], where: str) -> None:
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"indicators.yaml {where} 含未声明字段 {sorted(unknown)}")


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"indicators.yaml {where} 必须是非空字符串")
    return value.strip()


def _optional_text(value: Any, where: str) -> str:
    return "" if value is None else _text(value, where)


def _labels(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_text(value, where),)
    if isinstance(value, (list, tuple)):
        return tuple(_text(item, where) for item in value)
    raise ValueError(f"indicators.yaml {where} 必须是字符串或列表")


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"indicators.yaml {where} 必须是 true/false")
    return value


def _pairs(value: Any, where: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError(f"indicators.yaml {where} 必须是列表")
    out: list[tuple[str, str]] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"indicators.yaml {where}[{index}] 必须是二元组")
        out.append(
            (
                _text(pair[0], f"{where}[{index}][0]"),
                _text(pair[1], f"{where}[{index}][1]"),
            )
        )
    return tuple(out)


def _pattern(value: Any, where: str) -> str:
    pattern = _text(value, where)
    if len(pattern) > _MAX_PATTERN_CHARS:
        raise ValueError(f"indicators.yaml {where} 超过 {_MAX_PATTERN_CHARS} 字符")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"indicators.yaml {where} 非法正则: {exc}") from exc
    return pattern


def _load(path: Path) -> Registry:
    """严格加载 + 校验；任何结构错误都抛 ValueError（不静默忽略）。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 顶层必须是映射")
    if raw.get("version") != 1:
        raise ValueError(f"{path}: version 必须为 1")
    _reject_unknown(raw, _ALLOWED_TOP_LEVEL, "顶层")

    indicators: dict[str, IndicatorSpec] = {}
    raw_indicators = raw.get("indicators") or {}
    if not isinstance(raw_indicators, dict):
        raise ValueError(f"{path}: indicators 必须是映射")
    for name, entry in raw_indicators.items():
        key = _text(name, "指标名")
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: {key} 必须是映射")
        _reject_unknown(entry, _ALLOWED_INDICATOR_FIELDS, key)

        flags_raw = entry.get("flags") or {}
        if not isinstance(flags_raw, dict):
            raise ValueError(f"{path}: {key}.flags 必须是映射")
        flags: dict[str, bool] = {}
        for flag_name, value in flags_raw.items():
            flag_key = _text(flag_name, f"{key}.flags 的 key")
            if not _FLAG_KEY_RE.fullmatch(flag_key):
                raise ValueError(f"{path}: {key}.flags 非法标记名 {flag_key!r}")
            flags[flag_key] = _bool(value, f"{key}.flags.{flag_key}")

        narrative: tuple[str, str, str] | None = None
        if entry.get("narrative") is not None:
            item = entry["narrative"]
            if not isinstance(item, dict):
                raise ValueError(f"{path}: {key}.narrative 必须是映射")
            _reject_unknown(item, _ALLOWED_NARRATIVE_FIELDS, f"{key}.narrative")
            narrative = (
                _text(item.get("keyword"), f"{key}.narrative.keyword"),
                _pattern(item.get("pattern"), f"{key}.narrative.pattern"),
                _text(item.get("unit"), f"{key}.narrative.unit"),
            )
        year_end: tuple[str, str] | None = None
        if entry.get("year_end") is not None:
            item = entry["year_end"]
            if not isinstance(item, dict):
                raise ValueError(f"{path}: {key}.year_end 必须是映射")
            _reject_unknown(item, _ALLOWED_YEAR_END_FIELDS, f"{key}.year_end")
            year_end = (
                _pattern(item.get("pattern"), f"{key}.year_end.pattern"),
                _text(item.get("unit"), f"{key}.year_end.unit"),
            )

        lead_order = entry.get("lead_order", 999)
        if not isinstance(lead_order, int) or isinstance(lead_order, bool):
            raise ValueError(f"{path}: {key}.lead_order 必须是整数")

        indicators[key] = IndicatorSpec(
            name=key,
            table_canonical=_text(
                entry.get("table_canonical") or key, f"{key}.table_canonical"
            ),
            table_keys=_labels(entry.get("table_keys"), f"{key}.table_keys"),
            table_labels=_labels(entry.get("table_labels"), f"{key}.table_labels"),
            narrative=narrative,
            year_end=year_end,
            abstract_column=_optional_text(
                entry.get("abstract_column"), f"{key}.abstract_column"
            ),
            abstract_unit=_optional_text(
                entry.get("abstract_unit"), f"{key}.abstract_unit"
            ),
            flags=flags,
            lead_order=lead_order,
        )

    aliases = _pairs(raw.get("aliases") or [], "aliases")

    ordering_raw = raw.get("ordering") or {}
    if not isinstance(ordering_raw, dict):
        raise ValueError(f"{path}: ordering 必须是映射")
    _reject_unknown(ordering_raw, _ORDERING_TYPES, "ordering")
    ordering: dict[str, Any] = {}
    for key, value in ordering_raw.items():
        expected = _ORDERING_TYPES[key]
        if not isinstance(value, expected) or (
            expected is int and isinstance(value, bool)
        ):
            raise ValueError(f"{path}: ordering.{key} 类型不对")
        if key == "indicator_keys":
            # 该项是 [{name, aliases}]，单独校验，不能套用字符串列表规则
            parsed: list[tuple[str, tuple[str, ...]]] = []
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    raise ValueError(f"{path}: indicator_keys[{index}] 必须是映射")
                _reject_unknown(item, {"name", "aliases"}, f"indicator_keys[{index}]")
                parsed.append(
                    (
                        _text(item.get("name"), f"indicator_keys[{index}].name"),
                        _labels(
                            item.get("aliases"), f"indicator_keys[{index}].aliases"
                        ),
                    )
                )
            ordering[key] = tuple(parsed)
        elif isinstance(value, list):
            ordering[key] = [_text(item, f"ordering.{key}") for item in value]
        else:
            ordering[key] = value

    class_raw = raw.get("classification") or {}
    if not isinstance(class_raw, dict):
        raise ValueError(f"{path}: classification 必须是映射")
    _reject_unknown(class_raw, _ALLOWED_CLASSIFICATION_FIELDS, "classification")
    hints = (
        _pairs(class_raw["name_hints"], "classification.name_hints")
        if "name_hints" in class_raw
        else _BUILTIN_NAME_HINTS
    )
    segments = _BUILTIN_SEGMENT_TO_TYPE
    if "segment_to_type" in class_raw:
        raw_segments = class_raw["segment_to_type"]
        if not isinstance(raw_segments, dict):
            raise ValueError(f"{path}: classification.segment_to_type 必须是映射")
        segments = tuple(
            (
                _text(k, "classification.segment_to_type"),
                _text(v, "classification.segment_to_type"),
            )
            for k, v in raw_segments.items()
        )
    known_types = _known_company_types()
    for hint_key, company_type in (*hints, *segments):
        if company_type not in known_types:
            raise ValueError(f"{path}: 未知公司类型 {company_type}（{hint_key}）")

    return Registry(
        indicators=indicators,
        aliases=aliases,
        ordering=ordering,
        classification=ClassificationRules(hints, segments),
    )


@lru_cache(maxsize=1)
def _registry() -> Registry:
    try:
        return _load(_INDICATORS_YAML)
    except Exception:
        logger.exception("研报主配置加载失败，回退只读镜像：%s", _INDICATORS_YAML)
    try:
        return _load(_DEFAULTS_YAML)
    except Exception:
        logger.exception("研报降级镜像也加载失败，使用最小内置注册表")
        return _EMPTY


def reload_registry() -> None:
    """改完配置后清缓存；单请求内仍绑定同一份不可变快照。"""
    _registry.cache_clear()


def validate_registry(path: Path | None = None) -> Registry:
    """显式 fail-fast 入口：配置坏就抛 ValueError，供启动 lint / CI 调用。"""
    return _load(path or _INDICATORS_YAML)


# ——— per-indicator 查询 ———


def all_specs() -> dict[str, IndicatorSpec]:
    return _registry().indicators


def spec_of(name: str) -> IndicatorSpec | None:
    return _registry().indicators.get((name or "").strip())


def flag(name: str, key: str, default: bool = False) -> bool:
    """显式声明优先；未声明 / 未登记一律回退调用方的 default。"""
    spec = spec_of(name)
    if spec is None or not spec.declares(key):
        return default
    return spec.flags[key]


def names_with(key: str, fallback: Iterable[str] = ()) -> frozenset[str]:
    """带该标记的已登记指标 ∪ fallback 中未登记的项（不会被静默丢弃）。"""
    reg = _registry()
    if not reg.indicators:
        return frozenset(fallback)
    derived = {
        name for name, spec in reg.indicators.items() if spec.has(key)
    }
    extra = {name for name in fallback if name not in reg.indicators}
    return frozenset(derived | extra)


def lead_names(fallback: Iterable[str] = ()) -> tuple[str, ...]:
    """带 lead 标记的指标按 lead_order 排序；fallback 中未登记的项补在后面。"""
    reg = _registry()
    if not reg.indicators:
        return tuple(fallback)
    picked = sorted(
        (spec for spec in reg.indicators.values() if spec.has("lead")),
        key=lambda spec: (spec.lead_order, spec.name),
    )
    names = tuple(spec.name for spec in picked)
    extra = tuple(
        name for name in fallback if name not in reg.indicators and name not in names
    )
    return (*names, *extra)


def table_entry(name: str) -> tuple[str, tuple[str, ...]] | None:
    spec = spec_of(name)
    if spec is None or not spec.table_labels:
        return None
    return spec.table_canonical, spec.table_labels


def narrative_entry(name: str) -> tuple[str, str, str] | None:
    spec = spec_of(name)
    return spec.narrative if spec is not None else None


def abstract_entry(name: str) -> tuple[str, str, str] | None:
    spec = spec_of(name)
    if spec is None or not spec.abstract_column:
        return None
    return spec.abstract_column, spec.name, spec.abstract_unit


# ——— 整表查询（替代下游的模块级常量）———


def table_specs() -> dict[str, tuple[str, tuple[str, ...]]]:
    """原 extract._INDICATORS：查询键 → (报表正式名, 表格行别名)。"""
    return {
        key: (spec.table_canonical, spec.table_labels)
        for spec in _registry().indicators.values()
        for key in spec.table_keys
    }


def narrative_patterns() -> dict[str, tuple[str, str, str]]:
    """原 extract._NARRATIVE_PATTERNS。"""
    return {
        name: spec.narrative
        for name, spec in _registry().indicators.items()
        if spec.narrative is not None
    }


def year_end_specs() -> tuple[tuple[str, str, str], ...]:
    """原 supplemental._YEAR_END_SPECS；顺序会影响「较上年末」正文，按配置保序。"""
    reg = _registry()
    order = reg.ordering.get("year_end_order") or [
        name for name, spec in reg.indicators.items() if spec.year_end is not None
    ]
    out: list[tuple[str, str, str]] = []
    for name in order:
        spec = reg.indicators.get(name)
        if spec is not None and spec.year_end is not None:
            out.append((name, spec.year_end[0], spec.year_end[1]))
    return tuple(out)


def abstract_map() -> tuple[tuple[str, str, str], ...]:
    """原 market_finance._ABSTRACT_MAP。"""
    reg = _registry()
    order = reg.ordering.get("abstract_order") or [
        name for name, spec in reg.indicators.items() if spec.abstract_column
    ]
    return tuple(
        (spec.abstract_column, name, spec.abstract_unit)
        for name in order
        if (spec := reg.indicators.get(name)) is not None and spec.abstract_column
    )


def aliases_map() -> tuple[tuple[str, str], ...]:
    """原 industry._ALIASES，保序（长词优先）。"""
    return _registry().aliases


def indicator_keys() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """原 parse._INDICATOR_KEYS。"""
    return _registry().ordering.get("indicator_keys", ())


def ordering_list(key: str) -> tuple[str, ...]:
    value = _registry().ordering.get(key, ())
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def ordering_int(key: str, default: int = 0) -> int:
    value = _registry().ordering.get(key, default)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def ordering_str(key: str, default: str = "") -> str:
    value = _registry().ordering.get(key, default)
    return value if isinstance(value, str) else default


def name_hints() -> tuple[tuple[str, str], ...]:
    return _registry().classification.name_hints


def segment_map() -> dict[str, str]:
    return dict(_registry().classification.segment_to_type)
