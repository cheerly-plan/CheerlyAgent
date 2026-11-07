"""研报章节注册表：把「标题 → 写哪一节」从 if-else 变成可查表分派。

边界（与评审结论一致，勿放宽）：

* **配置只选规则，不执行逻辑。** sections.yaml 里只有标题别名和一句兜底
  文案；数字、正文、抽数、校验、PDF 全部仍由 compose.py 的确定性 Python
  决定。配置不能注入任何正文内容或绕过防幻觉门禁。
* **SKILL.md 优先。** 大纲小节可在正文首行写 ``section: <id>`` 直接绑定
  renderer，优先级高于标题别名。新增一个行业 / 报告类型，要么复用已有
  renderer（只加 SKILL.md），要么写一个 renderer 并注册一次；不必再往
  compose.py 里补 if，也不必在十几个文件里补名单。
* **读坏不拖垮服务，但也不静默。** 配置损坏记 exception 日志并降级到
  ``sections_defaults.py``——由 ``sections.yaml`` 自动生成的静态快照，不是
  第二份手工名单。``validate_sections()`` 是显式 fail-fast 入口，
  ``validate_bindings()`` 检查配置里的 id 是否都有 renderer，供 CI / 启动
  lint 调用；启动 lint 还会校验生成物与主配置同步。
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ai_agent.core.logging import get_logger
from ai_agent.report.compare import SeriesCompare
from ai_agent.report.extract import IndicatorFact
from ai_agent.report.features import Features
from ai_agent.report.parse import ReportPlan
from ai_agent.report.supplemental import SupplementalResult

try:
    from ai_agent.report.sections_defaults import ALIASES as _BUILTIN_ALIASES
    from ai_agent.report.sections_defaults import NOTE as _BUILTIN_NOTE
except ImportError:
    # 最后一道兜底：生成物被打包漏掉时别让整个 report 包无法导入。
    # 此时所有章节只能写占位句，启动 lint 的 sections_snapshot_sync 会报错。
    get_logger(__name__).exception("章节降级快照缺失，章节分派将全部写占位句")
    _BUILTIN_ALIASES = ()
    _BUILTIN_NOTE = "这一节没有对应的数据，未写。"

logger = get_logger(__name__)

_SECTIONS_YAML = Path(__file__).resolve().parent / "config" / "sections.yaml"
_SECTION_ID_RE = re.compile(r"[a-z][a-z0-9_]*\Z")
_MAX_NOTE_CHARS = 200

# 配置缺失/损坏时的兜底来自 sections_defaults.py——那是 sections.yaml 的
# 自动生成快照，不再手工维护第二份名单。纯 Python 字面量，降级路径不依赖
# YAML 解析。改章节别名只改 sections.yaml + 跑一次同步脚本，lint 会拦漂移。


@dataclass(frozen=True)
class SectionContext:
    """一节正文能看到的全部输入；renderer 只读，不改状态。"""

    title: str
    plan: ReportPlan
    facts: list[IndicatorFact]
    current: list[IndicatorFact]
    series: list[SeriesCompare]
    gaps: list[str]
    views: SupplementalResult
    state: Features
    analysis: list[str] | None = None


SectionRenderer = Callable[[SectionContext], list[str]]


@dataclass(frozen=True)
class SectionsConfig:
    """标题 → id 的反查表 + 占位句；不可变，单请求内绑定同一份快照。"""

    by_title: dict[str, str] = field(default_factory=dict)
    ids: frozenset[str] = frozenset()
    fallback_note: str = _BUILTIN_NOTE


_RENDERERS: dict[str, SectionRenderer] = {}


def _ensure_renderers() -> None:
    """renderer 声明在 compose.py 顶层；直接单独 import 本模块时按需补加载。

    compose 已导入时这是 no-op；compose 正在导入时本函数不会被调用
    （注册只写字典，不查表），不会造成循环 import。
    """
    if not _RENDERERS:
        importlib.import_module("ai_agent.report.compose")


def register(section_id: str) -> Callable[[SectionRenderer], SectionRenderer]:
    """注册一个 renderer。id 全局唯一，重复注册直接 fail-fast。"""

    def decorator(fn: SectionRenderer) -> SectionRenderer:
        if not _SECTION_ID_RE.fullmatch(section_id):
            raise ValueError(f"非法 section_id: {section_id!r}")
        if section_id in _RENDERERS:
            raise ValueError(f"section_id 重复注册: {section_id}")
        _RENDERERS[section_id] = fn
        return fn

    return decorator


def renderer_ids() -> frozenset[str]:
    _ensure_renderers()
    return frozenset(_RENDERERS)


def _invert(pairs: tuple[tuple[str, tuple[str, ...]], ...], note: str) -> SectionsConfig:
    by_title: dict[str, str] = {}
    ids: set[str] = set()
    for section_id, titles in pairs:
        if not _SECTION_ID_RE.fullmatch(section_id):
            raise ValueError(f"sections.yaml 非法 section_id: {section_id!r}")
        ids.add(section_id)
        for title in titles:
            key = title.strip()
            if not key:
                raise ValueError(f"sections.yaml {section_id} 含空标题")
            if key in by_title and by_title[key] != section_id:
                raise ValueError(
                    f"sections.yaml 标题 {key!r} 同时映射到 "
                    f"{by_title[key]} 与 {section_id}"
                )
            by_title[key] = section_id
    return SectionsConfig(by_title, frozenset(ids), note)


def _require_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"sections.yaml {where} 必须是非空字符串")
    return value.strip()


def _load(path: Path) -> SectionsConfig:
    """严格加载；任何未声明字段、非法结构、含数字/模板的占位句都抛 ValueError。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 顶层必须是映射")
    if raw.get("version") != 1:
        raise ValueError(f"{path}: version 必须为 1")
    unknown = set(raw) - {"version", "aliases", "fallback"}
    if unknown:
        raise ValueError(f"{path}: 顶层含未声明字段 {sorted(unknown)}")

    aliases_raw = raw.get("aliases") or {}
    if not isinstance(aliases_raw, dict):
        raise ValueError(f"{path}: aliases 必须是映射")
    pairs: list[tuple[str, tuple[str, ...]]] = []
    for section_id, titles in aliases_raw.items():
        key = _require_text(section_id, "aliases 的 key")
        if not _SECTION_ID_RE.fullmatch(key):
            raise ValueError(f"{path}: 非法 section_id {key!r}")
        if isinstance(titles, str):
            items = (titles,)
        elif isinstance(titles, list):
            items = tuple(titles)
        else:
            raise ValueError(f"{path}: {key} 必须是字符串或字符串列表")
        pairs.append((key, tuple(_require_text(t, key) for t in items)))

    fallback_raw = raw.get("fallback") or {}
    if not isinstance(fallback_raw, dict):
        raise ValueError(f"{path}: fallback 必须是映射")
    unknown = set(fallback_raw) - {"note"}
    if unknown:
        raise ValueError(f"{path}: fallback 含未声明字段 {sorted(unknown)}")
    note = _require_text(fallback_raw.get("note"), "fallback.note")
    if len(note) > _MAX_NOTE_CHARS or "\n" in note:
        raise ValueError(f"{path}: fallback.note 必须是单行且不超过 {_MAX_NOTE_CHARS} 字符")
    # 占位句只是文案，不能夹带数字或模板占位——配置不许注入正文数字。
    if any(ch.isdigit() for ch in note) or "{" in note or "}" in note:
        raise ValueError(f"{path}: fallback.note 不得含数字或模板占位")
    return _invert(tuple(pairs), note)


@lru_cache(maxsize=1)
def _config() -> SectionsConfig:
    try:
        return _load(_SECTIONS_YAML)
    except Exception:
        logger.exception("研报章节配置加载失败，回退内置别名表：%s", _SECTIONS_YAML)
        return _invert(_BUILTIN_ALIASES, _BUILTIN_NOTE)


def reload_sections() -> None:
    """改完配置后清缓存；单请求内仍绑定同一份不可变快照。"""
    _config.cache_clear()


def validate_sections(path: Path | None = None) -> SectionsConfig:
    """显式 fail-fast 入口：配置坏就抛 ValueError，供启动 lint / CI 调用。"""
    return _load(path or _SECTIONS_YAML)


def validate_bindings(path: Path | None = None) -> None:
    """配置里引用的 id 必须都有 renderer；拼错 id 不让它悄悄降级成占位句。"""
    config = _load(path or _SECTIONS_YAML)
    _ensure_renderers()
    missing = sorted(config.ids - renderer_ids())
    if missing:
        raise ValueError(f"sections.yaml 引用了未注册的 renderer: {missing}")


def resolve(title: str, section_id: str = "") -> tuple[str, SectionRenderer] | None:
    """按 SKILL.md 声明的 id 优先，其次按标题别名查；未绑定返回 None。"""
    _ensure_renderers()
    key = (section_id or "").strip() or _config().by_title.get((title or "").strip(), "")
    if not key:
        return None
    renderer = _RENDERERS.get(key)
    if renderer is None:
        logger.warning("章节 %r 绑定的 renderer %r 未注册，写占位句", title, key)
        return None
    return key, renderer


def fallback_lines() -> list[str]:
    return [_config().fallback_note]
