"""按公司类型路由研报指标集。

铁律：先分类，再取模板；模板外的指标不抽、也不写「没抽到」。
银行看息差／资产质量；Fabless 看毛利率／研发／存货；设备看合同负债／存货／在手订单；
Foundry 看稼动率／CAPEX——一套制造业模板通吃，必然出跨行业污染与假缺失。

粗粒度 ``industry_of`` 仍只服务 skill 路由（bank vs generic）；
细粒度 ``company_type_of`` 决定抽哪些指标、排除哪些污染项。
指标名一律用报表原文术语，才能和原文对得上。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ai_agent.core.logging import get_logger
from ai_agent.report import registry

logger = get_logger(__name__)

# 下面这些常量只为兼容既有调用与测试保留，是纯字符串别名，**不是注册表**。
# 新增行业不需要在这里补常量：公司类型白名单、行业名别名一律来自
# config/industries.yaml（profiles / aliases 段）。
BANK = "bank"
GENERIC = "generic"
SEMICONDUCTOR = "semiconductor"
ROBOTICS = "robotics"
INDUSTRY_SHARED = "industry_shared"

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

_ALLOWED_NOTE_FIELDS = frozenset({"next_period", "margin_gloss", "rnd_gloss"})
# 行业文案只能写定性句，不能借配置往正文塞未经 facts 校验的数字。
_NUMBER_RE = re.compile(r"[0-9０-９]|%|％")
_PROFILE_KEY_RE = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class IndustryNotes:
    """行业级定性文案；全部可选，缺省时由渲染层使用通用兜底句。"""

    next_period: tuple[str, ...] = ()
    margin_gloss: str = ""
    rnd_gloss: str = ""


@dataclass(frozen=True)
class IndicatorProfile:
    """一类公司该看／不该看的核心指标（5–8 个，不是三张表全抄）。"""

    indicators: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    notes: IndustryNotes = IndustryNotes()


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


def _parse_notes(entry: dict[str, Any], where: str) -> IndustryNotes:
    """行业定性文案：严格 schema + 数字护栏，配置不能注入未经核对的数值。"""
    raw = entry.get("notes") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{where}.notes 必须是映射")
    unknown = set(raw) - _ALLOWED_NOTE_FIELDS
    if unknown:
        raise ValueError(f"{where}.notes 含未声明字段 {sorted(unknown)}")
    next_period = _as_str_tuple(raw.get("next_period"))
    glosses: dict[str, str] = {}
    for key in ("margin_gloss", "rnd_gloss"):
        value = raw.get(key) or ""
        if not isinstance(value, str):
            raise ValueError(f"{where}.notes.{key} 必须是字符串")
        glosses[key] = value.strip()
    for text in (*next_period, *glosses.values()):
        if not text.strip():
            continue  # 三个字段都是可选；留空表示用渲染层的通用兜底句
        if _NUMBER_RE.search(text):
            raise ValueError(f"{where}.notes 不得含数字或百分比：{text!r}")
    clean_next_period = tuple(item.strip() for item in next_period if item.strip())
    return IndustryNotes(
        clean_next_period, glosses["margin_gloss"], glosses["rnd_gloss"]
    )


_ALLOWED_TOP_LEVEL = frozenset({"aliases", "sets", "profiles"})


def _load_profiles(path: Path) -> dict[str, IndicatorProfile]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"industries.yaml 顶层必须是映射：{path}")
    unknown_top = set(raw) - _ALLOWED_TOP_LEVEL
    if unknown_top:
        raise ValueError(f"industries.yaml 顶层含未声明字段 {sorted(unknown_top)}")
    sets = raw.get("sets") or {}
    profiles_raw = raw.get("profiles") or {}
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ValueError(f"industries.yaml 缺少 profiles: {path}")
    profiles: dict[str, IndicatorProfile] = {}
    for key, entry in profiles_raw.items():
        profile_key = str(key)
        if not _PROFILE_KEY_RE.fullmatch(profile_key):
            raise ValueError(f"industries.yaml profile key 非法：{profile_key!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"industries.yaml profile 非法: {profile_key}")
        include = _as_str_tuple(entry.get("include"))
        if not include:
            raise ValueError(f"industries.yaml profile 缺少 include: {profile_key}")
        profiles[profile_key] = IndicatorProfile(
            indicators=include,
            exclude=_resolve_exclude(entry, sets if isinstance(sets, dict) else {}),
            notes=_parse_notes(entry, f"industries.yaml profile {profile_key}"),
        )
    for required in (GENERIC, INDUSTRY_SHARED):
        if required not in profiles:
            raise ValueError(f"industries.yaml 必须包含 {required} 兜底 profile")
    return profiles


@lru_cache(maxsize=1)
def _profiles() -> dict[str, IndicatorProfile]:
    """首次调用时加载 industries.yaml；测试可 `_profiles.cache_clear()`。"""
    return _load_profiles(_INDUSTRIES_YAML)


@lru_cache(maxsize=1)
def _aliases() -> dict[str, str]:
    """行业名 → profile key；主源 industries.yaml，读不到才回退自动生成快照。"""
    try:
        raw = yaml.safe_load(_INDUSTRIES_YAML.read_text(encoding="utf-8")) or {}
        aliases = raw.get("aliases") or {}
        if isinstance(aliases, dict) and aliases:
            return {str(k).strip(): str(v).strip() for k, v in aliases.items()}
    except Exception:
        logger.exception("industries.yaml 读取失败，行业别名回退内置快照")
    try:
        from ai_agent.report.classification_defaults import TYPE_ALIASES

        return dict(TYPE_ALIASES)
    except ImportError:
        return {}


def known_company_types() -> frozenset[str]:
    """公司类型白名单：与 registry 共用 industries.yaml 的 profile keys。"""
    try:
        return frozenset(_profiles())
    except Exception:
        logger.exception("industries.yaml 读取失败，公司类型回退内置快照")
        try:
            from ai_agent.report.classification_defaults import COMPANY_TYPES

            return COMPANY_TYPES
        except ImportError:
            return frozenset({BANK, GENERIC})


def validate_industries(path: Path | None = None) -> dict[str, IndicatorProfile]:
    """显式校验：schema + aliases 指向有效 profile + 定性文案不含数字。"""
    target = path or _INDUSTRIES_YAML
    profiles = _load_profiles(target)
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    aliases = raw.get("aliases") or {}
    if not isinstance(aliases, dict):
        raise ValueError("industries.yaml aliases 必须是映射")
    unknown = sorted({str(v).strip() for v in aliases.values()} - set(profiles))
    if unknown:
        raise ValueError(f"industries.yaml aliases 指向未知 profile: {unknown}")
    return profiles


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
    """单家公司分类：行业 skill 花名册 → 名称关键字 → generic。不调用模型。"""
    text = (subject or "").strip()
    if not text:
        return GENERIC
    if "银行" in text:
        return BANK
    roster = _skill_company_roster()
    for key, company_type in roster.items():
        if key == text or key in text or text in key:
            return company_type
    for token, company_type in registry.name_hints():
        if token in text:
            return company_type
    return GENERIC


def type_for_segment(segment: str) -> str:
    """赛道名 → 细类型；未登记赛道回退 generic。

    刻意不回退 semiconductor：那会让漏配 segment_to_type 的新行业静默套上
    半导体指标模板，产出跨行业污染且零报错。漏配由 lint 显式拦截。
    """
    return registry.segment_map().get((segment or "").strip(), GENERIC)


def default_indicators(industry_or_type: str) -> list[str]:
    """indicator_set / 公司类型 → 默认指标列表。"""
    key = _normalize_key(industry_or_type)
    profiles = _profiles()
    profile = profiles.get(key) or profiles[GENERIC]
    return list(profile.indicators)


def excluded_indicators(industry_or_type: str) -> frozenset[str]:
    """该类公司不该出现的指标（抽数／缺口／观察句一律跳过）。"""
    key = _normalize_key(industry_or_type)
    profiles = _profiles()
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
    profiles = _profiles()
    return profiles.get(key) or profiles[GENERIC]


def notes_for_plan(plan: Any) -> IndustryNotes:
    """从 skill 的 indicator_set（或报告行业名）取行业定性文案。

    取不到时返回空 notes，由渲染层使用通用兜底句；这里不出现任何 skill_id 分支。
    """
    from ai_agent.report.skills import get_skill

    skill_id = (getattr(plan, "skill_id", "") or "").strip()
    skill = get_skill(skill_id) if skill_id else None
    key = (getattr(skill, "indicator_set", "") or "").strip() or (
        getattr(plan, "industry", "") or ""
    ).strip()
    return profile_of(key).notes


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
    """行业名／公司类型 → profile key；别名表来自 industries.yaml。

    新增行业只要在该文件补一行 alias，不必再往 Python 字典里加常量。
    """
    text = (raw or "").strip().lower()
    stripped = (raw or "").strip()
    aliases = _aliases()
    return (
        aliases.get(text)
        or aliases.get(stripped)
        or (stripped if stripped in _profiles() else GENERIC)
    )


def _skill_company_roster() -> dict[str, str]:
    """合并各行业 skill 的 companies 花名册 → 公司／又名 → 细类型；失败则空。"""
    try:
        from ai_agent.report.skills import SKILLS
    except Exception:
        return {}
    out: dict[str, str] = {}
    for skill in SKILLS:
        if skill.kind != "industry" or not skill.companies:
            continue
        for name, _code, segment, aka in skill.companies:
            company_type = registry.segment_map().get((segment or "").strip())
            if not company_type:
                continue
            out[name] = company_type
            if aka and aka != name:
                out[aka] = company_type
    return out


# 兼容旧测试／调用名。
_semiconductor_roster = _skill_company_roster
