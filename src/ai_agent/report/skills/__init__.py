"""报告类型 skill 层：管「做什么报告」，不管「数字怎么算」。

一个 skill = ``skills/<skill_id>/SKILL.md``（YAML frontmatter + 大纲章节）。
新增报告类型 = 加一个目录，不改 Python、不复制管线；parse 按 kind + 行业
规则路由，extract / compare / validate / pdf 积木全部复用。

刻意守住的边界（与评审结论一致）：

* skill 只决定大纲、指标集、写哪些节——「写什么」。
* 抽数、同比环比、数字校验、PDF 落盘路径仍是确定性代码——「数字怎么来、
  怎么验」。这些积木不接受 skill 改写，防幻觉门禁不因报告类型放宽。
* 路由本身是规则匹配（kind + 主体行业），不调用 LLM，可回归测试。

大纲条目是 (标题, 要点) 二元组，要点里可写 ``{indicators}`` 占位，
由 parse 按实际选中的指标集填充——skill 保持纯数据，不引用 ReportPlan，
避免和 parse 模块互相 import。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ai_agent.report.industry import default_indicators

_SKILLS_ROOT = Path(__file__).resolve().parent
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<meta>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)
_SECTION_RE = re.compile(
    r"^###\s+(?P<title>\S.*?)\s*\n(?P<body>.*?)(?=^###\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class ReportSkill:
    """一种报告类型。从 SKILL.md 加载后由 parse 按需求路由。"""

    skill_id: str
    # company / industry：与 ReportPlan.kind 同一套取值
    kind: str
    name: str
    # 主体行业（industry.BANK 等）精确匹配；空元组 = 该 kind 下的兜底模板
    industries: tuple[str, ...]
    description: str
    # 默认指标集。银行模板直接展开；通用模板复用 industry.generic 集。
    indicators: tuple[str, ...]
    sections: tuple[tuple[str, str], ...]
    # 行业宇宙：((公司名, 股票代码, 赛道), ...)；赛道可空；空元组 = 仍只从知识库推断主体
    companies: tuple[tuple[str, str, str], ...] = ()

    def build_outline(self, indicators: list[str]) -> list[tuple[str, str]]:
        """把 {indicators} 占位填成实际选中的指标名列表。"""
        joined = "、".join(indicators) if indicators else "已点名的指标"
        return [(title, brief.replace("{indicators}", joined)) for title, brief in self.sections]

    def segment_of(self, subject: str) -> str:
        """按公司名（或包含关系）查赛道；未标注返回空串。"""
        token = (subject or "").strip()
        if not token:
            return ""
        for name, _code, segment in self.companies:
            if name == token or name in token or token in name:
                return segment
        return ""

    def companies_by_segment(self) -> list[tuple[str, list[str]]]:
        """按赛道稳定分组：[(赛道, [公司名…]), …]；无赛道的放「未分组」。"""
        order: list[str] = []
        buckets: dict[str, list[str]] = {}
        for name, _code, segment in self.companies:
            key = segment or "未分组"
            if key not in buckets:
                order.append(key)
                buckets[key] = []
            buckets[key].append(name)
        return [(key, buckets[key]) for key in order]


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


def _parse_industries(raw: str) -> tuple[str, ...]:
    text = raw.strip().strip("[]")
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _parse_companies(raw: str) -> tuple[tuple[str, str, str], ...]:
    """解析 ``公司名|代码`` 或 ``公司名|代码|赛道``；代码须为 6 位数字。"""
    text = (raw or "").strip().strip("[]")
    if not text:
        return ()
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        pieces = [p.strip() for p in item.split("|")]
        if len(pieces) not in {2, 3}:
            raise ValueError(f"companies 项须为「公司名|代码」或「公司名|代码|赛道」：{item!r}")
        name, code = pieces[0], pieces[1]
        segment = pieces[2] if len(pieces) == 3 else ""
        if not name or not re.fullmatch(r"\d{6}", code):
            raise ValueError(f"companies 项非法：{item!r}")
        if code in seen:
            continue
        seen.add(code)
        out.append((name, code, segment))
    return tuple(out)


def _parse_sections(body: str) -> tuple[tuple[str, str], ...]:
    marker = "## 大纲"
    start = body.find(marker)
    if start < 0:
        raise ValueError("SKILL.md 正文必须包含「## 大纲」章节")
    outline_block = body[start + len(marker) :]
    sections: list[tuple[str, str]] = []
    for match in _SECTION_RE.finditer(outline_block):
        title = match.group("title").strip()
        brief = " ".join(match.group("body").strip().split())
        if not title or not brief:
            raise ValueError(f"大纲小节不完整：title={title!r}")
        sections.append((title, brief))
    if not sections:
        raise ValueError("「## 大纲」下至少要有一个 ### 小节")
    return tuple(sections)


def _load_skill(path: Path) -> ReportSkill:
    meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    skill_id = meta.get("name", "").strip()
    kind = meta.get("kind", "").strip()
    title = meta.get("title", "").strip()
    description = meta.get("description", "").strip()
    indicator_set = meta.get("indicator_set", "generic").strip() or "generic"
    if not skill_id or not kind or not title or not description:
        raise ValueError(f"{path}: frontmatter 需要 name/kind/title/description")
    if path.parent.name != skill_id:
        raise ValueError(f"{path}: 目录名 {path.parent.name!r} 须与 name {skill_id!r} 一致")
    return ReportSkill(
        skill_id=skill_id,
        kind=kind,
        name=title,
        industries=_parse_industries(meta.get("industries", "")),
        description=description,
        indicators=tuple(default_indicators(indicator_set)),
        sections=_parse_sections(body),
        companies=_parse_companies(meta.get("companies", "")),
    )


@lru_cache(maxsize=1)
def _load_skills() -> tuple[ReportSkill, ...]:
    """扫描本包下 */SKILL.md，按目录名排序以保证稳定顺序。"""
    loaded: list[ReportSkill] = []
    for skill_md in sorted(_SKILLS_ROOT.glob("*/SKILL.md")):
        loaded.append(_load_skill(skill_md))
    if not loaded:
        raise LookupError(f"未找到任何 SKILL.md：{_SKILLS_ROOT}")
    return tuple(loaded)


def reload_skills() -> tuple[ReportSkill, ...]:
    """测试 / 热加载：清缓存后重新扫描目录，并同步模块级 SKILLS。"""
    global SKILLS
    _load_skills.cache_clear()
    SKILLS = _load_skills()
    return SKILLS


# 模块导入即加载；对外仍叫 SKILLS，与既有 import 兼容。
SKILLS: tuple[ReportSkill, ...] = _load_skills()


def resolve_skill(kind: str, industry: str = "") -> ReportSkill:
    """先找行业专属模板，找不到再退到该 kind 的通用模板。

    kind 不认识时按通用公司模板兜底：调用方（parse）对 no_request 场景
    提前返回，不会走到这里依赖兜底语义。
    """
    skills = _load_skills()
    if industry:
        for skill in skills:
            if skill.kind == kind and industry in skill.industries:
                return skill
    for skill in skills:
        if skill.kind == kind and not skill.industries:
            return skill
    for skill in skills:
        if skill.kind == "company" and not skill.industries:
            return skill
    raise LookupError("报告类型注册表为空：至少要注册一个通用公司模板")


def get_skill(skill_id: str) -> ReportSkill | None:
    for skill in _load_skills():
        if skill.skill_id == skill_id:
            return skill
    return None


def skill_ids() -> list[str]:
    """给 API / 提示词 / 报错信息用的可选类型清单。"""
    return [skill.skill_id for skill in _load_skills()]


def skill_catalog() -> str:
    """report_type 可选值清单（skill_id（名称） 顿号分隔），供提示词渲染。

    消费方是 tools 层与 agents 层（组装工具描述、系统提示词时注入）；
    prompts 层刻意不 import 这里——文案层不反向依赖业务域。
    """
    return "、".join(f"{skill.skill_id}（{skill.name}）" for skill in _load_skills())
