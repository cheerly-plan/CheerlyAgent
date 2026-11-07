"""研报指标注册表：把散落在各模块里的指标元数据收敛到一份配置。

竖切阶段的设计取舍：

* **只描述意图，不执行逻辑。** 配置里是别名、正则、展示标记、分类关键字；
  抽数 / 计算 / 校验 / 渲染仍走确定性 Python，配置不能注入数字或替换校验规则。
* **已登记优先，未登记回退。** 目前只登记「毛利率」，其余指标继续用各模块
  内的既有常量，行为逐字不变；这是渐进迁移的过渡态，不是最终形态。
* **读坏不拖垮服务，但也不静默。** 加载失败记 exception 日志并降级为空注册表，
  下游自动回退内置常量；``validate_registry()`` 是显式 fail-fast 入口，供
  CI / 启动 lint 调用，P3 会在启动时强制执行。
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

_INDICATORS_YAML = Path(__file__).resolve().parent / "config" / "indicators.yaml"

FLAGS = ("money", "chart", "lead", "prefer_market", "ytd_ratio")

# 严格 schema：配置里出现任何未声明字段（尤其是数字、校验规则）一律 fail-fast，
# 保证「配置只能描述意图」不是一句口号。P3 可换成 pydantic 严格模型。
_ALLOWED_TOP_LEVEL = frozenset({"version", "indicators", "classification"})
_ALLOWED_INDICATOR_FIELDS = frozenset(
    {
        "table_canonical",
        "table_labels",
        "narrative",
        "abstract_column",
        "abstract_unit",
        "flags",
        "lead_order",
    }
)
_ALLOWED_NARRATIVE_FIELDS = frozenset({"keyword", "pattern", "unit"})
_ALLOWED_CLASSIFICATION_FIELDS = frozenset({"name_hints", "segment_to_type"})
_MAX_PATTERN_CHARS = 500

# 分类规则的内置兜底：配置缺失或损坏时仍保持原有分类行为。
# 值与 industry.py 的类型常量对齐（这里不 import industry，避免循环依赖）。
_KNOWN_COMPANY_TYPES = frozenset(
    {
        "bank",
        "generic",
        "semiconductor",
        "fabless",
        "foundry",
        "equipment",
        "packaging",
        "manufacturing",
        "software",
        "real_estate",
        "cyclical",
    }
)
_BUILTIN_NAME_HINTS: tuple[tuple[str, str], ...] = (
    ("银行", "bank"),
    ("置业", "real_estate"),
    ("地产", "real_estate"),
    ("房地产", "real_estate"),
    ("软件", "software"),
    ("互联", "software"),
    ("云", "software"),
    ("钢铁", "cyclical"),
    ("化工", "cyclical"),
    ("煤炭", "cyclical"),
    ("有色", "cyclical"),
)
_BUILTIN_SEGMENT_TO_TYPE: tuple[tuple[str, str], ...] = (
    ("设计", "fabless"),
    ("制造", "foundry"),
    ("设备", "equipment"),
    ("封测", "packaging"),
)


@dataclass(frozen=True)
class IndicatorSpec:
    """一个指标的元数据。字段与 indicators.yaml 一一对应。"""

    name: str
    table_canonical: str
    table_labels: tuple[str, ...]
    narrative: tuple[str, str, str] | None
    abstract_column: str
    abstract_unit: str
    flags: frozenset[str]
    lead_order: int

    def has(self, flag: str) -> bool:
        return flag in self.flags


@dataclass(frozen=True)
class ClassificationRules:
    name_hints: tuple[tuple[str, str], ...]
    segment_to_type: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Registry:
    indicators: dict[str, IndicatorSpec] = field(default_factory=dict)
    classification: ClassificationRules = field(
        default_factory=lambda: ClassificationRules(
            _BUILTIN_NAME_HINTS, _BUILTIN_SEGMENT_TO_TYPE
        )
    )


_EMPTY = Registry()


def _reject_unknown(value: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"indicators.yaml {where} 含未声明字段 {sorted(unknown)}")


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"indicators.yaml {where} 必须是非空字符串")
    return value.strip()


def _optional_text(value: Any, where: str) -> str:
    if value is None:
        return ""
    return _text(value, where)


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
        key = _text(pair[0], f"{where}[{index}][0]")
        kind = _text(pair[1], f"{where}[{index}][1]")
        out.append((key, kind))
    return tuple(out)


def _load(path: Path) -> Registry:
    """严格加载 + 校验；任何结构或正则错误都抛 ValueError（不静默忽略）。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 顶层必须是映射")
    if raw.get("version") != 1:
        raise ValueError(f"{path}: version 必须为 1")
    _reject_unknown(raw, _ALLOWED_TOP_LEVEL, "顶层")

    indicators_raw = raw.get("indicators") or {}
    if not isinstance(indicators_raw, dict):
        raise ValueError(f"{path}: indicators 必须是映射")
    indicators: dict[str, IndicatorSpec] = {}
    for name, entry in indicators_raw.items():
        key = _text(name, "指标名")
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: {key} 必须是映射")
        _reject_unknown(entry, _ALLOWED_INDICATOR_FIELDS, key)
        flags_raw = entry.get("flags") or {}
        if not isinstance(flags_raw, dict):
            raise ValueError(f"{path}: {key}.flags 必须是映射")
        unknown = set(flags_raw) - set(FLAGS)
        if unknown:
            raise ValueError(f"{path}: {key}.flags 含未知标记 {sorted(unknown)}")
        flags = frozenset(
            flag
            for flag, value in flags_raw.items()
            if _bool(value, f"{key}.flags.{flag}")
        )
        narrative: tuple[str, str, str] | None = None
        narrative_raw = entry.get("narrative")
        if narrative_raw is not None:
            if not isinstance(narrative_raw, dict):
                raise ValueError(f"{path}: {key}.narrative 必须是映射")
            _reject_unknown(narrative_raw, _ALLOWED_NARRATIVE_FIELDS, f"{key}.narrative")
            pattern = _text(narrative_raw.get("pattern"), f"{key}.narrative.pattern")
            if len(pattern) > _MAX_PATTERN_CHARS:
                raise ValueError(f"{path}: {key}.narrative.pattern 超过 {_MAX_PATTERN_CHARS} 字符")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"{path}: {key}.narrative.pattern 非法正则: {exc}") from exc
            narrative = (
                _text(narrative_raw.get("keyword"), f"{key}.narrative.keyword"),
                pattern,
                _text(narrative_raw.get("unit"), f"{key}.narrative.unit"),
            )
        lead_order = entry.get("lead_order", 999)
        if not isinstance(lead_order, int) or isinstance(lead_order, bool):
            raise ValueError(f"{path}: {key}.lead_order 必须是整数")
        indicators[key] = IndicatorSpec(
            name=key,
            table_canonical=_text(
                entry.get("table_canonical") or key, f"{key}.table_canonical"
            ),
            table_labels=_labels(entry.get("table_labels"), f"{key}.table_labels"),
            narrative=narrative,
            abstract_column=_optional_text(
                entry.get("abstract_column"), f"{key}.abstract_column"
            ),
            abstract_unit=_optional_text(
                entry.get("abstract_unit"), f"{key}.abstract_unit"
            ),
            flags=flags,
            lead_order=lead_order,
        )

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
                _text(key, "classification.segment_to_type"),
                _text(value, "classification.segment_to_type"),
            )
            for key, value in raw_segments.items()
        )
    for key, company_type in (*hints, *segments):
        if company_type not in _KNOWN_COMPANY_TYPES:
            raise ValueError(f"{path}: 未知公司类型 {company_type}（{key}）")
    return Registry(
        indicators=indicators,
        classification=ClassificationRules(hints, segments),
    )


@lru_cache(maxsize=1)
def _registry() -> Registry:
    try:
        return _load(_INDICATORS_YAML)
    except Exception:
        logger.exception("研报指标注册表加载失败，回退内置常量：%s", _INDICATORS_YAML)
        return _EMPTY


def reload_registry() -> None:
    """改完配置后清缓存；单请求内仍绑定同一份不可变快照。"""
    _registry.cache_clear()


def validate_registry(path: Path | None = None) -> Registry:
    """显式 fail-fast 入口：配置坏就抛 ValueError，供启动 lint / CI 调用。"""
    return _load(path or _INDICATORS_YAML)


def spec_of(name: str) -> IndicatorSpec | None:
    return _registry().indicators.get((name or "").strip())


def flag(name: str, key: str, default: bool = False) -> bool:
    """已登记指标取配置里的布尔标记；未登记指标回退调用方的原有判断。"""
    spec = spec_of(name)
    return spec.has(key) if spec is not None else default


def names_with(key: str, fallback: Iterable[str] = ()) -> frozenset[str]:
    """返回带该标记的全部已登记指标；配置降级时回退调用方常量。"""
    reg = _registry()
    if not reg.indicators:
        return frozenset(fallback)
    return frozenset(
        name for name, spec in reg.indicators.items() if spec.has(key)
    )


def table_entry(name: str) -> tuple[str, tuple[str, ...]] | None:
    """extract._INDICATORS 形态：(报表正式名, 表格行别名)。"""
    spec = spec_of(name)
    if spec is None or not spec.table_labels:
        return None
    return spec.table_canonical, spec.table_labels


def narrative_entry(name: str) -> tuple[str, str, str] | None:
    """extract._NARRATIVE_PATTERNS 形态：(检索关键词, 正则, 单位)。"""
    spec = spec_of(name)
    return spec.narrative if spec is not None else None


def abstract_entry(name: str) -> tuple[str, str, str] | None:
    """market_finance._ABSTRACT_MAP 形态：(摘要列名, 研报指标名, 单位)。"""
    spec = spec_of(name)
    if spec is None or not spec.abstract_column:
        return None
    return spec.abstract_column, spec.name, spec.abstract_unit


def lead_names(fallback: Iterable[str] = ()) -> tuple[str, ...]:
    """带 lead 标记的指标，按 lead_order 排序；配置降级时回退调用方常量。"""
    reg = _registry()
    if not reg.indicators:
        return tuple(fallback)
    picked = sorted(
        (spec for spec in reg.indicators.values() if spec.has("lead")),
        key=lambda spec: (spec.lead_order, spec.name),
    )
    return tuple(spec.name for spec in picked)


def name_hints() -> tuple[tuple[str, str], ...]:
    return _registry().classification.name_hints


def segment_map() -> dict[str, str]:
    return dict(_registry().classification.segment_to_type)
