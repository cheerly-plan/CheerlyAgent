"""文档类型配置：抽取 prompt 的「通用底座 + 可插拔配置块」。

为什么拆两层：
- 只服招股书的硬特化 prompt 抽得最干净，但换成年报、法律意见书、公告就废了；
- 纯通用 prompt 换文档都能跑，但在招股书上会飘出大量行业概念与零散供应商，
  谓词也不稳定（同一段关系这次叫「供应」下次叫「采购自」）。

折中：基础 SYSTEM 固定下来（输出契约、禁止项、片段序号规则），把**文档相关**
的部分（实体范围、谓词偏好、别名归一、补充约束）在运行时按 doc_type 注入。
抽取器只有一份，换文件是换配置，不是维护 N 份 system prompt。

配置来源：
1. 内置 ``kb/skills/<doc_type>/SKILL.md``（与报告 skill 同构：加类型 = 加目录）；
2. 可选 ``data/kb_doc_profiles.json`` 覆盖或追加——业务方临时改类型可不发版。

skill 只决定「抽什么」；清洗、对齐、落库、名称门禁仍是确定性代码。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
# CONFIG_SLOT 与图谱底座已迁到 ai_agent.prompts.kb —— 它们是纯提示词，
# 不该和文档类型配置的解析逻辑混在一起。这里 import 进来保持原有导入路径可用。
from ai_agent.prompts.builder import render
from ai_agent.prompts.kb import (
    CONFIG_LABEL_ALIASES,
    CONFIG_LABEL_DOC_TYPE,
    CONFIG_LABEL_NOTES,
    CONFIG_LABEL_PREDICATES,
    CONFIG_LABEL_SCOPE,
    CONFIG_LABEL_SCOPE_ANY,
    CONFIG_LEVEL_FULL,
    CONFIG_LEVEL_LIGHT,
    CONFIG_SLOT,
    GRAPH_BASE_SYSTEM,
)

logger = get_logger(__name__)

# 兼容别名：tests/test_doc_profiles.py 直接 import _BASE_SYSTEM 做契约断言
# （底座不许出现行业词、关键约束一条不能少），保留旧名让它们零改动。
_BASE_SYSTEM = GRAPH_BASE_SYSTEM

# 抽取粒度：同一份文档配置下再分轻重。
# 轻量不是「少抽一半片段」，而是**同一批片段里只留主干**——投研快速看图时
# 不需要每个零散供应商都变成一个点。
LEVELS = {"full": "完整", "light": "轻量"}
DEFAULT_LEVEL = "full"

_SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<meta>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)
_SECTION_RE = re.compile(
    r"^##\s+(?P<title>\S.*?)\s*\n(?P<body>.*?)(?=^##\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_BULLET_RE = re.compile(r"^[-*]\s+(.+)$", re.MULTILINE)
_ALIAS_RE = re.compile(r"^(.+?)\s*→\s*(.+)$")


@dataclass(frozen=True)
class DocProfile:
    """一套文档类型的抽取配置。字段全部对应 prompt 里的一段可读文本。"""

    key: str
    label: str
    doc_name: str
    # 重点实体范围：给模型一个「该看什么」的锚，不是硬白名单
    entity_scope: tuple[str, ...] = ()
    # 推荐谓词：优先选用，但允许合理新增——锁死词表会让陌生文档的关系全被压成
    # 少数几个泛词，反而丢了信息
    predicates: tuple[str, ...] = ()
    # 别名归一：原文写法 → 标准名。只在能确定映射时写，别写「发行人 → 某公司」
    # 这种需要靠文档本身才能补全的规则
    aliases: tuple[tuple[str, str], ...] = ()
    # 补充约束：这类文档特有的坑（法律意见书的程序性套话、年报的代称客户）
    notes: tuple[str, ...] = ()

    @property
    def is_general(self) -> bool:
        return not self.entity_scope and not self.predicates and not self.aliases

    def brief(self) -> dict:
        """给前端下拉用的摘要：不暴露完整 prompt，只给能帮人做选择的字段。"""
        return {
            "key": self.key,
            "label": self.label,
            "doc_name": self.doc_name,
            "entity_scope": list(self.entity_scope),
            "predicates": list(self.predicates),
            "note_count": len(self.notes),
            "is_general": self.is_general,
        }


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError("SKILL.md 必须以 --- YAML frontmatter --- 开头")
    meta: dict[str, str] = {}
    for raw in match.group("meta").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"frontmatter 行缺少冒号：{raw!r}")
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, match.group("body")


def _section_bullets(body: str, title: str) -> tuple[str, ...]:
    for match in _SECTION_RE.finditer(body):
        if match.group("title").strip() != title:
            continue
        return tuple(
            item.strip()
            for item in _BULLET_RE.findall(match.group("body"))
            if item.strip()
        )
    return ()


def _section_aliases(body: str) -> tuple[tuple[str, str], ...]:
    aliases: list[tuple[str, str]] = []
    for item in _section_bullets(body, "别名"):
        matched = _ALIAS_RE.match(item)
        if matched is None:
            logger.warning("kb skill 别名行无法解析，已跳过：%s", item)
            continue
        src, dst = matched.group(1).strip(), matched.group(2).strip()
        if src and dst:
            aliases.append((src, dst))
    return tuple(aliases)


def _load_skill_md(path: Path) -> DocProfile:
    meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    key = meta.get("name", "").strip()
    label = meta.get("label", "").strip()
    doc_name = meta.get("doc_name", "").strip()
    if not key or not label or not doc_name:
        raise ValueError(f"{path}: frontmatter 需要 name/label/doc_name")
    if path.parent.name != key:
        raise ValueError(f"{path}: 目录名 {path.parent.name!r} 须与 name {key!r} 一致")
    return DocProfile(
        key=key,
        label=label,
        doc_name=doc_name,
        entity_scope=_section_bullets(body, "实体范围"),
        predicates=_section_bullets(body, "推荐谓词"),
        aliases=_section_aliases(body),
        notes=_section_bullets(body, "补充约束"),
    )


def _skill_order(path: Path) -> tuple[int, str]:
    """按 frontmatter 的 order 排序；缺省靠后，同 order 按目录名。"""
    try:
        meta, _body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        order_raw = meta.get("order", "500").strip() or "500"
        order = int(order_raw)
    except Exception:
        order = 500
    return order, path.parent.name


@lru_cache(maxsize=1)
def _builtin_profiles() -> tuple[DocProfile, ...]:
    """扫描 kb/skills/*/SKILL.md。读坏单份只记日志跳过，不能拖垮建图。"""
    if not _SKILLS_ROOT.is_dir():
        logger.error("kb skills 目录不存在：%s", _SKILLS_ROOT)
        return ()
    loaded: list[DocProfile] = []
    for skill_md in sorted(_SKILLS_ROOT.glob("*/SKILL.md"), key=_skill_order):
        try:
            loaded.append(_load_skill_md(skill_md))
        except Exception:
            logger.exception("kb skill 读取失败，已跳过：%s", skill_md)
    if not any(p.key == "general" for p in loaded):
        logger.error("kb skills 缺少 general 兜底，建图未知类型将无配置可回落")
    return tuple(loaded)


def _from_json(raw: dict) -> DocProfile | None:
    """把 JSON 里的一条配置转成 DocProfile；缺关键字段就跳过（不让坏配置拖垮建图）。"""
    key = str(raw.get("key") or "").strip()
    if not key:
        logger.warning("kb 文档配置缺少 key，已跳过：%s", raw)
        return None

    def _tuple(name: str) -> tuple[str, ...]:
        value = raw.get(name) or []
        if isinstance(value, str):
            value = [value]
        return tuple(str(v).strip() for v in value if str(v).strip())

    aliases_raw = raw.get("aliases") or {}
    if isinstance(aliases_raw, dict):
        aliases = tuple(
            (str(k).strip(), str(v).strip())
            for k, v in aliases_raw.items()
            if str(k).strip() and str(v).strip()
        )
    else:  # 也接受 [["简称", "全称"], ...]
        aliases = tuple(
            (str(pair[0]).strip(), str(pair[1]).strip())
            for pair in aliases_raw
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        )

    return DocProfile(
        key=key,
        label=str(raw.get("label") or key),
        doc_name=str(raw.get("doc_name") or raw.get("label") or key),
        entity_scope=_tuple("entity_scope"),
        predicates=_tuple("predicates"),
        aliases=aliases,
        notes=_tuple("notes"),
    )


@lru_cache(maxsize=1)
def _overrides() -> tuple[DocProfile, ...]:
    """从 JSON 读覆盖配置。读坏了只记一行日志并回退内置，绝不抛给建图流程。"""
    path = Path(settings.kb_doc_profiles_path)
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("profiles") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError("profiles 必须是数组")
        parsed = [_from_json(item) for item in items if isinstance(item, dict)]
        return tuple(p for p in parsed if p is not None)
    except Exception:
        logger.exception("kb 文档配置读取失败，回退内置配置：%s", path)
        return ()


def _general_fallback() -> DocProfile:
    for profile in _builtin_profiles():
        if profile.key == "general":
            return profile
    # skills 目录损坏时的最后兜底：空范围 + 空谓词，仍能跑通底座
    return DocProfile(key="general", label="通用（其他文档）", doc_name="通用商业文档")


def list_profiles() -> list[DocProfile]:
    """内置 skill + JSON 覆盖，同 key 覆盖、新 key 追加、general 永远在最后。"""
    merged: dict[str, DocProfile] = {p.key: p for p in _builtin_profiles()}
    for item in _overrides():
        merged[item.key] = item
    ordered = [p for key, p in merged.items() if key != "general"]
    ordered.append(merged.get("general", _general_fallback()))
    return ordered


def resolve_profile(key: str | None) -> DocProfile:
    """按 key 取配置；未知/空 key 一律回落到通用模式（换陌生文档也能跑）。"""
    wanted = (key or "").strip()
    if not wanted:
        return _general_fallback()
    for profile in list_profiles():
        if profile.key == wanted:
            return profile
    logger.warning("kb 未知文档类型 %s，回落到通用配置", wanted)
    return _general_fallback()


def normalize_level(level: str | None) -> str:
    value = (level or "").strip().lower()
    return value if value in LEVELS else DEFAULT_LEVEL


def render_config(profile: DocProfile, level: str = DEFAULT_LEVEL) -> str:
    """把配置渲染成 prompt 里的一段文本。

    写成自然语言而不是 JSON：模型对「文档类型：IPO 招股意向书 / 重点实体范围：…」
    这种配置块的服从度，明显高于嵌套 JSON（后者会被它当成待抽取的内容）。
    """
    lines: list[str] = [render(CONFIG_LABEL_DOC_TYPE, NAME=profile.doc_name)]

    if profile.entity_scope:
        lines.append(CONFIG_LABEL_SCOPE + "、".join(profile.entity_scope))
    else:
        lines.append(CONFIG_LABEL_SCOPE_ANY)

    if profile.predicates:
        lines.append(CONFIG_LABEL_PREDICATES + "、".join(profile.predicates))

    if profile.aliases:
        lines.append(
            CONFIG_LABEL_ALIASES
            + "；".join(f"{src} → {dst}" for src, dst in profile.aliases)
        )

    if profile.notes:
        lines.append(CONFIG_LABEL_NOTES)
        lines.extend(f"- {note}" for note in profile.notes)

    if normalize_level(level) == "light":
        lines.append(CONFIG_LEVEL_LIGHT)
    else:
        lines.append(CONFIG_LEVEL_FULL)

    return "\n".join(lines)


def build_system_prompt(doc_type: str | None, level: str = DEFAULT_LEVEL) -> str:
    """基础 SYSTEM + 配置块 → 最终 system prompt。"""
    profile = resolve_profile(doc_type)
    return _BASE_SYSTEM.replace(CONFIG_SLOT, render_config(profile, level))


def reload_profiles() -> None:
    """运维改完 skill / JSON 后调一次即可生效（不必重启服务）。"""
    _builtin_profiles.cache_clear()
    _overrides.cache_clear()
