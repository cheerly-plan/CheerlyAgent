"""文档类型配置：抽取 prompt 的「通用底座 + 可插拔配置块」。

为什么拆两层：
- 只服招股书的硬特化 prompt 抽得最干净，但换成年报、法律意见书、公告就废了；
- 纯通用 prompt 换文档都能跑，但在招股书上会飘出大量行业概念与零散供应商，
  谓词也不稳定（同一段关系这次叫「供应」下次叫「采购自」）。

折中：基础 SYSTEM 固定下来（输出契约、禁止项、片段序号规则），把**文档相关**
的部分（实体范围、谓词偏好、别名归一、补充约束）在运行时按 doc_type 注入。
抽取器只有一份，换文件是换配置，不是维护 N 份 system prompt。

配置来源：内置几套常用类型（见 _BUILTIN），再用 data/kb_doc_profiles.json
覆盖或追加——业务方要加「债券募集说明书」这类新类型，改 JSON 就行，
不用改代码、不用发版。
"""

from __future__ import annotations

import json
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


# ——— 内置配置：覆盖资本市场最常见的几类文档 ———

# 年报配置单独提出来：季度报告与它几乎同构（同一家上市公司的定期报告，
# 主体/子公司/财务指标/分季度分行业数据都一样），直接复用它的实体范围、
# 谓词与别名，避免复制一份后调年报规则时要同步改两份。
_ANNUAL_REPORT = DocProfile(
    key="annual_report",
    label="上市公司年报",
    doc_name="上市公司年度报告",
    entity_scope=(
        "上市公司及其子公司",
        "参股主体与联营企业",
        "董事、监事与高管",
        "业务板块",
        "产品与服务",
        "主要客户与供应商",
        "财务指标",
        "重大投资项目",
        "对外担保与诉讼",
    ),
    predicates=(
        "控股",
        "参股",
        "子公司",
        "销售给",
        "采购自",
        "投资",
        "营收来自",
        "属于",
        "计提",
        "担保",
        "任职",
        "披露",
    ),
    notes=(
        "客户/供应商若原文只写「第一名客户」「前五大供应商」，就用原文称呼，"
        "不要编造具体公司名。",
        "分季度、分行业、分地区的数据归到同一个指标实体，不要拆成多个名字相近的点。",
        "「本公司」「公司」指上市公司本身，统一用其全称。",
    ),
)

# 季度报告（含一季报/半年报/中报/三季报）：抽取内容复用年报配置，只改展示名
# 并补一条季报特有的约束——季报披露的科目与篇幅通常少于年报，未披露的内容
# 自然抽不到，不需要额外白名单。
_QUARTERLY_REPORT = DocProfile(
    key="quarterly_report",
    label="上市公司季报",
    doc_name="上市公司季度报告（含半年报/中报）",
    entity_scope=_ANNUAL_REPORT.entity_scope,
    predicates=_ANNUAL_REPORT.predicates,
    aliases=_ANNUAL_REPORT.aliases,
    notes=(
        "季度报告的信息完整度低于年报，未披露的科目、章节不要脑补抽取。",
        *_ANNUAL_REPORT.notes,
    ),
)

_BUILTIN: tuple[DocProfile, ...] = (
    DocProfile(
        key="prospectus",
        label="招股意向书",
        doc_name="IPO 招股意向书 / 招股说明书",
        entity_scope=(
            "发行人及其子公司",
            "控股股东与实际控制人",
            "董事、监事、高管与核心技术人员",
            "主要产品与型号",
            "原材料与生产设备",
            "募投项目",
            "资质许可与专利",
            "监管与审批单位",
            "财务指标",
            "重大风险事项",
            "中介机构",
        ),
        predicates=(
            "控股",
            "全资子公司",
            "参股",
            "聘请",
            "生产",
            "采购",
            "供应",
            "销售",
            "募投包含",
            "拥有资质",
            "持有专利",
            "投资",
            "审批",
            "属于",
            "担任",
            "依赖",
            "应用于",
            "同比增长",
        ),
        notes=(
            "concept 只用于行业赛道（如「民爆行业」「风电运营」），不要用于普通业务抽象词。",
            "财务指标抽指标名（「营业收入」「装机容量」），不要抽成「营业收入 120 亿元」"
            "或「62,200万千瓦」这种带数值的整句；具体数字和年份写进说明。"
            "不要把「流动资产」「其他流动资产」这类科目名单独当实体。",
            "母公司与子公司是不同实体，不要把子公司并进发行人。",
        ),
    ),
    _ANNUAL_REPORT,
    _QUARTERLY_REPORT,
    DocProfile(
        key="legal_opinion",
        label="法律意见书",
        doc_name="法律意见书",
        entity_scope=(
            "委托人及发行主体",
            "自然人（股东、董监高、实际控制人）",
            "标的公司与标的资产",
            "合同与协议",
            "权利与义务",
            "证照、批复与许可",
            "诉讼与仲裁",
            "监管机构",
            "签字律师与律师事务所",
        ),
        predicates=(
            "持有",
            "签署",
            "涉及",
            "取得",
            "有效",
            "出资",
            "担保",
            "诉讼对象",
            "批准",
            "任职",
        ),
        notes=(
            "「经核查」「本所律师认为」「截至本意见书出具日」是程序性表述，"
            "不要抽成实体或关系。",
            "文号、日期、条款序号是限定语，不是实体。",
            "同一主体在正文与附件里写法可能不同（简称/全称），统一到全称。",
        ),
    ),
    DocProfile(
        key="due_diligence",
        label="尽调报告",
        doc_name="尽职调查报告",
        entity_scope=(
            "标的企业",
            "股东与实际控制人",
            "关联方",
            "核心人员",
            "资质与知识产权",
            "主要资产",
            "重大合同",
            "客户与供应商",
            "财务与税务事项",
            "诉讼、处罚与合规风险",
        ),
        predicates=(
            "持有",
            "控制",
            "关联",
            "签署",
            "供应",
            "采购",
            "抵押",
            "质押",
            "涉诉",
            "处罚",
            "承诺",
            "依赖",
            "出资",
        ),
        notes=(
            "问题清单里的「存在……风险」要连到具体主体或资产，不要只留一个抽象风险词。",
            "访谈记录、资料清单里的一问一答不要当关系抽，只取陈述性事实。",
        ),
    ),
    DocProfile(
        key="announcement",
        label="公司公告",
        doc_name="上市公司公告",
        entity_scope=(
            "上市公司",
            "交易对方",
            "标的资产或标的公司",
            "董事、监事与高管",
            "股东",
            "担保与借款",
            "监管机构",
            "中介机构",
        ),
        predicates=(
            "收购",
            "出售",
            "增资",
            "减持",
            "质押",
            "担保",
            "聘任",
            "辞职",
            "审议通过",
            "披露",
            "停牌",
        ),
        notes=(
            "「本公司」「公司」指上市公司本身，统一用其全称。",
            "金额与时点只在构成关系时顺带写进说明，不单独成实体。",
        ),
    ),
    DocProfile(
        key="general",
        label="通用（其他文档）",
        doc_name="通用商业文档",
        notes=(
            "不限定实体白名单：按基础规则抽原文中的专有名词与关键业务概念。",
            "谓词简洁即可，不强制套用词表；不做别名映射。",
        ),
    ),
)

_GENERAL = next(p for p in _BUILTIN if p.key == "general")


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


def list_profiles() -> list[DocProfile]:
    """内置 + 覆盖，同 key 覆盖、新 key 追加、general 永远在最后。"""
    merged: dict[str, DocProfile] = {p.key: p for p in _BUILTIN}
    for item in _overrides():
        merged[item.key] = item
    ordered = [p for key, p in merged.items() if key != "general"]
    ordered.append(merged.get("general", _GENERAL))
    return ordered


def resolve_profile(key: str | None) -> DocProfile:
    """按 key 取配置；未知/空 key 一律回落到通用模式（换陌生文档也能跑）。"""
    wanted = (key or "").strip()
    if not wanted:
        return _GENERAL
    for profile in list_profiles():
        if profile.key == wanted:
            return profile
    logger.warning("kb 未知文档类型 %s，回落到通用配置", wanted)
    return _GENERAL


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
    """运维改完 JSON 后调一次即可生效（不必重启服务）。"""
    _overrides.cache_clear()
