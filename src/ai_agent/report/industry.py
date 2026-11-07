"""按公司类型路由研报指标集。

铁律：先分类，再取模板；模板外的指标不抽、也不写「没抽到」。
银行看息差／资产质量；Fabless 看毛利率／研发／存货；设备看合同负债／存货／在手订单；
Foundry 看稼动率／CAPEX——一套制造业模板通吃，必然出跨行业污染与假缺失。

粗粒度 ``industry_of`` 只服务 skill 路由：按 industries.yaml 的行业别名匹配主体名，
返回 profile key（bank / insurance / …），匹配不到才 generic。
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

# 行业级「赛道名 → 细类型」在 config/industries.yaml 的 profile.segments；
# 公司名关键字 → 细类型仍在 config/indicators.yaml 的 classification.name_hints。
# 全局 classification.segment_to_type 只作「没有行业上下文时」的兜底，声明了
# segments 的行业不会读它；配置不可用时由只读降级快照保持分类能力。

_INDUSTRIES_YAML = Path(__file__).resolve().parent / "config" / "industries.yaml"

_ALLOWED_NOTE_FIELDS = frozenset(
    {"next_period", "structure_gloss", "observation"}
)
# 行业文案只能写定性句，不能借配置往正文塞未经 facts 校验的数字。
_NUMBER_RE = re.compile(r"[0-9０-９]|%|％")
_PROFILE_KEY_RE = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class IndustryNotes:
    """行业级定性文案；全部可选，缺省时由渲染层使用通用兜底句。"""

    next_period: tuple[str, ...] = ()
    # 赛道均值句的定性 gloss：按结构指标名索引（毛利率／净息差…）。
    structure_gloss: tuple[tuple[str, str], ...] = ()
    observation: str = ""

    def gloss_for(self, indicator: str) -> str:
        key = (indicator or "").strip()
        for name, text in self.structure_gloss:
            if name == key:
                return text
        return ""


@dataclass(frozen=True)
class IndicatorProfile:
    """一类公司该看／不该看的核心指标（5–8 个，不是三张表全抄）。"""

    indicators: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    notes: IndustryNotes = IndustryNotes()
    # 该行业的赛道名 → 公司类型；声明后即为权威，不再回退全局 segment_to_type。
    # 用 tuple 保序且可哈希，避免 frozen dataclass 里塞可变 dict。
    segments: tuple[tuple[str, str], ...] = ()
    # 「本期结论」的指标出句顺序；声明后即接管该行业的结论口径，
    # 渲染层不再按 skill_id 写分支（原来是 bank / generic 两套写死）。
    lead: tuple[str, ...] = ()
    # 出现这些指标就说明是本品类公司（如银行的净息差）：用于报告类型被误路由时的兜底。
    lead_hint: tuple[str, ...] = ()
    # 赛道经营解读／全行业梯队表用的比率清单；有赛道 ≠ 一律写毛利率。
    # 声明后 analyze / compose 按名单写梯队，银行可配净息差／不良等。
    structure: tuple[str, ...] = ()


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


def _parse_structure_gloss(raw: Any, where: str) -> tuple[tuple[str, str], ...]:
    """structure_gloss: {指标名: 定性句} → 保序 tuple；值不得含数字。"""
    if raw is None or raw == "":
        return ()
    if not isinstance(raw, dict):
        raise ValueError(f"{where}.notes.structure_gloss 必须是映射")
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, value in raw.items():
        name = str(key).strip() if key is not None else ""
        if not name:
            raise ValueError(f"{where}.notes.structure_gloss 不得含空指标名")
        if not isinstance(value, str):
            raise ValueError(f"{where}.notes.structure_gloss[{name!r}] 必须是字符串")
        text = value.strip()
        if not text:
            continue
        if _NUMBER_RE.search(text):
            raise ValueError(
                f"{where}.notes.structure_gloss 不得含数字或百分比：{text!r}"
            )
        if name in seen:
            continue
        seen.add(name)
        pairs.append((name, text))
    return tuple(pairs)


def _parse_notes(entry: dict[str, Any], where: str) -> IndustryNotes:
    """行业定性文案：严格 schema + 数字护栏，配置不能注入未经核对的数值。"""
    raw = entry.get("notes") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{where}.notes 必须是映射")
    unknown = set(raw) - _ALLOWED_NOTE_FIELDS
    if unknown:
        raise ValueError(f"{where}.notes 含未声明字段 {sorted(unknown)}")
    next_period = _as_str_tuple(raw.get("next_period"))
    observation_raw = raw.get("observation") or ""
    if not isinstance(observation_raw, str):
        raise ValueError(f"{where}.notes.observation 必须是字符串")
    observation = observation_raw.strip()
    structure_gloss = _parse_structure_gloss(raw.get("structure_gloss"), where)
    for text in (*next_period, observation):
        if not text.strip():
            continue  # 字段都是可选；留空表示用渲染层的通用兜底句
        if _NUMBER_RE.search(text):
            raise ValueError(f"{where}.notes 不得含数字或百分比：{text!r}")
    clean_next_period = tuple(item.strip() for item in next_period if item.strip())
    return IndustryNotes(
        clean_next_period,
        structure_gloss,
        observation,
    )


def _parse_name_list(
    entry: dict[str, Any], field: str, where: str
) -> tuple[str, ...]:
    """profile 里的指标名名单（lead / lead_hint）：保序去重，拒绝空串。

    「名字是否真的登记过」属于跨文件校验，放在 lint 里做；这里只保证形状合法，
    避免 industries.yaml 读不出来时把整条分类链路带崩。
    """
    raw = entry.get(field)
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{where}.{field} 必须是列表")
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        name = str(item).strip() if isinstance(item, str) else ""
        if not name:
            raise ValueError(f"{where}.{field} 不得含空指标名：{item!r}")
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


def _parse_segments(entry: dict[str, Any], where: str) -> tuple[tuple[str, str], ...]:
    """行业级赛道语义：赛道名 → 公司类型（profile key）。

    严格 schema：值必须是合法的 profile key 形态；「值确实存在」由
    _load_profiles 在全部 profile 构建完后统一校验，避免解析顺序造成误判。
    """
    raw = entry.get("segments") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{where}.segments 必须是映射")
    pairs: list[tuple[str, str]] = []
    for key, value in raw.items():
        segment = str(key).strip()
        company_type = str(value).strip()
        if not segment or not _PROFILE_KEY_RE.fullmatch(company_type):
            raise ValueError(
                f"{where}.segments 非法：{key!r} → {value!r}（值须为 profile key）"
            )
        pairs.append((segment, company_type))
    return tuple(pairs)


_ALLOWED_TOP_LEVEL = frozenset({"aliases", "sets", "profiles"})
# profile 里允许的字段；未声明的字段一律报错，避免「写了但没人读」的假配置。
_ALLOWED_PROFILE_FIELDS = frozenset(
    {
        "include",
        "exclude",
        "exclude_refs",
        "notes",
        "segments",
        "lead",
        "lead_hint",
        "structure",
    }
)


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
        unknown = set(entry) - _ALLOWED_PROFILE_FIELDS
        if unknown:
            raise ValueError(
                f"industries.yaml profile {profile_key} 含未声明字段 {sorted(unknown)}"
            )
        include = _as_str_tuple(entry.get("include"))
        if not include:
            raise ValueError(f"industries.yaml profile 缺少 include: {profile_key}")
        where = f"industries.yaml profile {profile_key}"
        profiles[profile_key] = IndicatorProfile(
            indicators=include,
            exclude=_resolve_exclude(entry, sets if isinstance(sets, dict) else {}),
            notes=_parse_notes(entry, where),
            segments=_parse_segments(entry, where),
            lead=_parse_name_list(entry, "lead", where),
            lead_hint=_parse_name_list(entry, "lead_hint", where),
            structure=_parse_name_list(entry, "structure", where),
        )
    for required in (GENERIC, INDUSTRY_SHARED):
        if required not in profiles:
            raise ValueError(f"industries.yaml 必须包含 {required} 兜底 profile")
    # 赛道类型必须指向本文件里真实存在的 profile：新增行业写错类型名时
    # fail-fast，而不是在渲染期静默落到 generic 指标集。
    for key, profile in profiles.items():
        for segment, company_type in profile.segments:
            if company_type not in profiles:
                raise ValueError(
                    f"industries.yaml profile {key} 的赛道 {segment!r} "
                    f"指向未知公司类型 {company_type!r}"
                )
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

def _industry_tokens() -> tuple[tuple[str, str], ...]:
    """主体名里可能出现的行业词 → profile key，按长度降序（最长优先）。

    只取中文别名（英文 profile key 不做子串匹配，避免在英文公司名里瞎猜）；
    「制造业」这类长词排在「制造」前，避免短词先命中。名单来自
    industries.yaml 的 aliases，不在 Python 里写死「银行」。
    """
    tokens = [
        (token.strip(), key)
        for token, key in _aliases().items()
        if token.strip() and not token.strip().isascii()
    ]
    return tuple(sorted(tokens, key=lambda item: len(item[0]), reverse=True))


def industry_of(subjects: list[str]) -> str:
    """粗行业（供 skill 路由）：主体名命中哪个行业词，就返回它的 profile key。

    例：「某某银行股份有限公司」→ bank；「某某保险」→ insurance（配置了
    ``保险: insurance`` 别名后）。都没命中 → generic。
    新增一个公司类行业只要在 industries.yaml 补一行中文别名，不用改这里。
    """
    tokens = _industry_tokens()
    for name in subjects:
        text = (name or "").strip()
        if not text:
            continue
        for token, key in tokens:
            if token in text:
                return key
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
    """单家公司分类：行业 skill 花名册 → 名称关键字 → 行业别名 → generic。

    名称关键字来自 indicators.yaml 的 classification.name_hints，
    行业别名来自 industries.yaml 的 aliases；两者都是配置，本函数没有
    写死的行业名（原先「主体名含银行即判 bank」的分支已由 ``银行: bank``
    别名接管）。
    """
    text = (subject or "").strip()
    if not text:
        return GENERIC
    roster = _skill_company_roster()
    for key, company_type in roster.items():
        if key == text or key in text or text in key:
            return company_type
    for token, company_type in registry.name_hints():
        if token in text:
            return company_type
    # 只在配了专属指标档案（非 generic）时按行业别名兜底，避免「软件」这类
    # 泛词把公司错判进某个细类型后套错模板。
    for token, profile_key in _industry_tokens():
        if token in text and profile_key != GENERIC:
            return profile_key
    return GENERIC


def _fallback_industry_segments(key: str) -> tuple[tuple[str, str], ...] | None:
    """industries.yaml 读不到时，从只读降级快照取该行业的赛道语义。"""
    try:
        from ai_agent.report.classification_defaults import INDUSTRY_SEGMENTS

        return tuple(INDUSTRY_SEGMENTS.get(key, {}).items())
    except ImportError:  # pragma: no cover - 仅打包缺生成物时触发
        return None


def segment_map_for(industry_key: str | None = None) -> dict[str, str]:
    """取某行业的赛道 → 公司类型映射。

    声明了 ``segments`` 的行业是**权威源，不合并全局字典**：不同行业的同名赛道
    （例如都叫「设备」）可以指向不同指标模板，且漏配赛道时会拿到 generic，
    绝不会静默套上别的行业模板。未声明 segments 的行业才回退全局兜底。
    """
    key = _normalize_key(industry_key or "")
    profile: IndicatorProfile | None = None
    try:
        profile = _profiles().get(key)
    except Exception:
        logger.exception("industries.yaml 读取失败，赛道语义回退内置快照")
        pairs = _fallback_industry_segments(key)
        if pairs is not None:
            return dict(pairs)
    if profile is not None and profile.segments:
        return dict(profile.segments)
    return dict(registry.segment_map())


def type_for_segment(segment: str, industry_key: str | None = None) -> str:
    """赛道名 → 细类型；未登记赛道回退 generic。

    ``industry_key`` 传入时按该行业自己的 segments 解析（渲染层应始终传）。
    刻意不回退 semiconductor：那会让漏配赛道的新行业静默套上半导体模板，
    产出跨行业污染且零报错。漏配由 lint 显式拦截。
    """
    return segment_map_for(industry_key).get((segment or "").strip(), GENERIC)


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


def profile_key_for_plan(plan: Any) -> str:
    """报告 → 行业档案 key：skill.indicator_set → industry → company_type。

    所有需要「按行业取配置」的调用都走这里，避免各处重复拼 skill_id 判断。
    返回值已经过别名归一（如「银行」→ bank）。
    """
    from ai_agent.report.skills import get_skill

    skill_id = (getattr(plan, "skill_id", "") or "").strip()
    skill = get_skill(skill_id) if skill_id else None
    raw = (
        (getattr(skill, "indicator_set", "") or "").strip()
        or (getattr(plan, "industry", "") or "").strip()
        or (getattr(plan, "company_type", "") or "").strip()
    )
    return _normalize_key(raw) if raw else ""


def notes_for_plan(plan: Any) -> IndustryNotes:
    """从行业档案取定性文案；取不到时返回空 notes，由渲染层用通用兜底句。"""
    return profile_of(profile_key_for_plan(plan)).notes


def structure_for_plan(plan: Any) -> tuple[str, ...]:
    """赛道解读／梯队表用的比率清单；未声明则空（不默认毛利率）。"""
    key = profile_key_for_plan(plan)
    if not key:
        return ()
    return profile_of(key).structure


def _fallback_industry_lead(key: str, field: str) -> tuple[str, ...] | None:
    """industries.yaml 读不到时，从只读降级快照取该行业的结论口径。"""
    try:
        from ai_agent.report import classification_defaults

        table = getattr(classification_defaults, "INDUSTRY_LEAD", {})
        entry = (table or {}).get(key) or {}
        return tuple(entry.get(field, ())) if isinstance(entry, dict) else ()
    except ImportError:  # pragma: no cover - 仅打包缺生成物时触发
        return None


def lead_priority(profile_key: str) -> tuple[str, ...]:
    """该行业的「本期结论」出句顺序；未声明时返回空元组（走通用口径）。

    结论口径跟着行业档案走，而不是跟着 skill_id 或「是不是银行」走：
    新增一个公司口径只要在 profile 里写 ``lead:``，渲染层不用改。
    """
    key = _normalize_key(profile_key or "")
    try:
        profile = _profiles().get(key)
    except Exception:
        logger.exception("industries.yaml 读取失败，结论口径回退内置快照")
        return _fallback_industry_lead(key, "lead") or ()
    return profile.lead if profile is not None else ()


def lead_hint_profiles() -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """[(hint 指标名, 对应结论顺序), …]：按档案声明顺序，用于报告类型被误路由时兜底。

    例：银行档案声明 ``lead_hint: [净息差, …]``，即使请求被路由到通用公司模板，
    只要正文里出现净息差，结论仍按银行口径出句。
    """
    try:
        profiles = _profiles()
    except Exception:
        logger.exception("industries.yaml 读取失败，结论口径线索不可用")
        return ()
    return tuple(
        (profile.lead_hint, profile.lead)
        for profile in profiles.values()
        if profile.lead_hint and profile.lead
    )


def normalize_indicator(text: str) -> str:
    """把需求里的简称归一到报表正式指标名；不认识就原样返回。"""
    cleaned = (text or "").strip()
    for alias, canonical in _ALIASES:
        if alias in cleaned:
            return canonical
    return cleaned


def indicator_aliases() -> tuple[tuple[str, str], ...]:
    return _ALIASES


def canonical_industry(raw: str) -> str:
    """行业名 → profile key；不认识返回空串。

    与 ``_normalize_key`` 的区别：不把未知行业悄悄当成 generic。路由层需要
    「这个行业到底登记过没有」的判断，用这个函数，别用会兜底的那个。
    """
    text = (raw or "").strip()
    if not text:
        return ""
    aliases = _aliases()
    found = aliases.get(text.lower()) or aliases.get(text)
    if found:
        return found
    try:
        return text if text in _profiles() else ""
    except Exception:
        logger.exception("industries.yaml 读取失败，行业名按原值返回")
        return text


def _normalize_key(raw: str) -> str:
    """行业名／公司类型 → profile key；别名表来自 industries.yaml。

    新增行业只要在该文件补一行 alias，不必再往 Python 字典里加常量。
    未知值落 generic——需要「到底登记过没有」的判断请用 canonical_industry。
    """
    text = (raw or "").strip().lower()
    stripped = (raw or "").strip()
    aliases = _aliases()
    found = aliases.get(text) or aliases.get(stripped)
    if found:
        return found
    try:
        return stripped if stripped in _profiles() else GENERIC
    except Exception:
        # 配置文件损坏时保留原 key，让降级快照仍有机会命中；不让分类链路整体抛错。
        logger.exception("industries.yaml 读取失败，行业名归一保留原值")
        return stripped or GENERIC


def _skill_company_roster() -> dict[str, str]:
    """合并各行业 skill 的 companies 花名册 → 公司／又名 → 细类型；失败则空。

    每个 skill 用自己 indicator_set 的赛道映射：两个行业可以都有「设备」赛道，
    各自指向不同的指标模板，不会被合并成一份全局字典互相覆盖。
    """
    try:
        from ai_agent.report.skills import SKILLS
    except Exception:
        return {}
    out: dict[str, str] = {}
    for skill in SKILLS:
        if skill.kind != "industry" or not skill.companies:
            continue
        segments = segment_map_for(skill.indicator_set)
        for name, _code, segment, aka in skill.companies:
            company_type = segments.get((segment or "").strip())
            if not company_type:
                continue
            out[name] = company_type
            if aka and aka != name:
                out[aka] = company_type
    return out


# 兼容旧测试／调用名。
_semiconductor_roster = _skill_company_roster
