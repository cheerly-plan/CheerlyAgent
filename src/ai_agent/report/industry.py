"""按公司类型路由研报指标集。

铁律：先分类，再取模板；模板外的指标不抽、也不写「没抽到」。
银行看息差／资产质量；Fabless 看毛利率／研发／存货；设备看合同负债／存货／在手订单；
Foundry 看稼动率／CAPEX——一套制造业模板通吃，必然出跨行业污染与假缺失。

粗粒度 ``industry_of`` 仍只服务 skill 路由（bank vs generic）；
细粒度 ``company_type_of`` 决定抽哪些指标、排除哪些污染项。
指标名一律用报表原文术语，才能和原文对得上。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ai_agent.report import registry

# —— skill 路由用的粗行业（与 SKILL.md industries / indicator_set 对齐）——
BANK = "bank"
GENERIC = "generic"
SEMICONDUCTOR = "semiconductor"

# —— 指标路由用的细类型 ——
FABLESS = "fabless"
FOUNDRY = "foundry"
EQUIPMENT = "equipment"
PACKAGING = "packaging"
MANUFACTURING = "manufacturing"
SOFTWARE = "software"
REAL_ESTATE = "real_estate"
CYCLICAL = "cyclical"

# 赛道名 → 细类型、公司名关键字 → 细类型，已迁到 config/indicators.yaml 的
# classification 段（registry.segment_map() / registry.name_hints()）；配置不可用时
# 由 registry 的内置兜底保持原有行为，这里不再重复维护一份。

_INDUSTRIES_YAML = Path(__file__).resolve().parent / "config" / "industries.yaml"


@dataclass(frozen=True)
class IndicatorProfile:
    """一类公司该看／不该看的核心指标（5–8 个，不是三张表全抄）。"""

    indicators: tuple[str, ...]
    exclude: tuple[str, ...] = ()


def _as_str_tuple(raw: Any) -> tuple[str, ...]:
    if not raw:
        return ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw)


def _resolve_exclude(entry: dict[str, Any], sets: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for ref in entry.get("exclude_refs") or []:
        names.extend(_as_str_tuple(sets.get(str(ref))))
    names.extend(_as_str_tuple(entry.get("exclude")))
    # 保序去重，行为与迁出前字面量一致
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


def _load_profiles(path: Path) -> dict[str, IndicatorProfile]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sets = raw.get("sets") or {}
    profiles_raw = raw.get("profiles") or {}
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ValueError(f"industries.yaml 缺少 profiles: {path}")
    profiles: dict[str, IndicatorProfile] = {}
    for key, entry in profiles_raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"industries.yaml profile 非法: {key}")
        include = _as_str_tuple(entry.get("include"))
        if not include:
            raise ValueError(f"industries.yaml profile 缺少 include: {key}")
        profiles[str(key)] = IndicatorProfile(
            indicators=include,
            exclude=_resolve_exclude(entry, sets if isinstance(sets, dict) else {}),
        )
    if GENERIC not in profiles:
        raise ValueError("industries.yaml 必须包含 generic 兜底 profile")
    return profiles


@lru_cache(maxsize=1)
def _profiles() -> dict[str, IndicatorProfile]:
    """首次调用时加载 industries.yaml；测试可 `_profiles.cache_clear()`。"""
    return _load_profiles(_INDUSTRIES_YAML)


def _profile_map() -> dict[str, IndicatorProfile]:
    return _profiles()

# 需求文本里可能出现的简称 → 报表正式指标名。长词放前面避免被短词吃掉。
_ALIASES: tuple[tuple[str, str], ...] = registry.aliases_map()

def industry_of(subjects: list[str]) -> str:
    """粗行业：主体名含「银行」→ bank；否则 generic（供 skill 路由）。"""
    if any("银行" in (name or "") for name in subjects):
        return BANK
    return GENERIC


def company_type_of(subjects: list[str]) -> str:
    """细类型：决定指标模板。多主体时取第一个可判定的类型；判不出 → generic。"""
    if not subjects:
        return GENERIC
    types = [classify_subject(name) for name in subjects]
    known = [item for item in types if item != GENERIC]
    if not known:
        return GENERIC
    # 多主体且类型不一致时，保守退回 generic，避免把银行指标套到制造业。
    if len(set(known)) > 1:
        return GENERIC
    return known[0]


def classify_subject(subject: str) -> str:
    """单家公司分类：半导体花名册 → 名称关键字 → generic。不调用模型。"""
    text = (subject or "").strip()
    if not text:
        return GENERIC
    if "银行" in text:
        return BANK
    roster = _semiconductor_roster()
    for key, company_type in roster.items():
        if key == text or key in text or text in key:
            return company_type
    for token, company_type in registry.name_hints():
        if token in text:
            return company_type
    return GENERIC


def type_for_segment(segment: str) -> str:
    """赛道名 → 细类型；未知赛道回退 semiconductor 共用集。"""
    return registry.segment_map().get((segment or "").strip(), SEMICONDUCTOR)


def default_indicators(industry_or_type: str) -> list[str]:
    """indicator_set / 公司类型 → 默认指标列表。"""
    key = _normalize_key(industry_or_type)
    profiles = _profile_map()
    profile = profiles.get(key) or profiles[GENERIC]
    return list(profile.indicators)


def excluded_indicators(industry_or_type: str) -> frozenset[str]:
    """该类公司不该出现的指标（抽数／缺口／观察句一律跳过）。"""
    key = _normalize_key(industry_or_type)
    profiles = _profile_map()
    profile = profiles.get(key) or profiles[GENERIC]
    return frozenset(profile.exclude)


def is_excluded(industry_or_type: str, indicator: str) -> bool:
    name = (indicator or "").strip()
    if not name:
        return False
    banned = excluded_indicators(industry_or_type)
    if name in banned:
        return True
    # 简称／别名也要拦住（PPOP、拨备覆盖率…）
    canonical = normalize_indicator(name)
    return canonical in banned and canonical != name


def profile_of(industry_or_type: str) -> IndicatorProfile:
    key = _normalize_key(industry_or_type)
    profiles = _profile_map()
    return profiles.get(key) or profiles[GENERIC]


def normalize_indicator(text: str) -> str:
    """把需求里的简称归一到报表正式指标名；不认识就原样返回。"""
    cleaned = (text or "").strip()
    for alias, canonical in _ALIASES:
        if alias in cleaned:
            return canonical
    return cleaned


def indicator_aliases() -> tuple[tuple[str, str], ...]:
    return _ALIASES


def _normalize_key(raw: str) -> str:
    text = (raw or "").strip().lower()
    aliases = {
        "bank": BANK,
        "银行": BANK,
        "generic": GENERIC,
        "manufacturing": MANUFACTURING,
        "一般制造": MANUFACTURING,
        "制造业": MANUFACTURING,
        "semiconductor": SEMICONDUCTOR,
        "半导体": SEMICONDUCTOR,
        "fabless": FABLESS,
        "设计": FABLESS,
        "foundry": FOUNDRY,
        "制造": FOUNDRY,
        "equipment": EQUIPMENT,
        "设备": EQUIPMENT,
        "packaging": PACKAGING,
        "封测": PACKAGING,
        "software": SOFTWARE,
        "软件": SOFTWARE,
        "real_estate": REAL_ESTATE,
        "地产": REAL_ESTATE,
        "cyclical": CYCLICAL,
        "周期": CYCLICAL,
    }
    return aliases.get(text) or aliases.get(raw.strip()) or (
        raw.strip() if raw.strip() in _profile_map() else GENERIC
    )


def _semiconductor_roster() -> dict[str, str]:
    """从半导体行业 skill 的 companies 花名册推导公司→类型；失败则空。"""
    try:
        from ai_agent.report.skills import get_skill
    except Exception:
        return {}
    skill = get_skill("semiconductor_industry_report")
    if skill is None or not skill.companies:
        return {}
    out: dict[str, str] = {}
    for name, _code, segment in skill.companies:
        company_type = registry.segment_map().get((segment or "").strip())
        if company_type:
            out[name] = company_type
    return out
