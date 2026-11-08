"""外部数据源适配层（Phase 3）。

**这一层解决什么问题**：10 章里第 2/3/4/6/7/9 章需要近 5 年年报附注、
审计报告与关键审计事项、交易所问询函及回复、行业原始统计。现有
``report/ensure_filings.py`` + ``kb/cninfo.py`` 只拉本期/基期的季报与半年报，
所以这些章节在当前数据条件下只能诚实地输出缺口。

**这一层不解决什么问题（重要）**：本模块不解析 PDF 附注、不抓取第三方站点、
不内置任何外部 API 端点。它提供的是：

1. ``SourceProvider`` 协议 —— 可插拔、可注入、可单测；
2. ``KnowledgeBaseProvider`` —— 盘点知识库里已入库的近 5 年源文件，
   把「缺哪类文件」从笼统断言变成**具体到章节与文档类型的缺口清单**；
3. ``LocalFactsProvider`` —— 已抽取好的结构化事实（JSON/JSONL）注入通道，
   这是真正能把数字喂进 ``SourceLedger`` 的路径；
4. ``IndustryStatProvider`` —— 行业统计只留 ``fetcher`` 注入点，
   未注入时明确声明缺口，绝不编造数据或接口。

事实准入统一复用 ``SourceLedger.add_external``：L4/L5 二手数据未回溯到
L1-L3 源头一律拒绝入表并记缺口。本层不做二次放行。

典型用法::

    bundle = collect_sources("兆易创新", "2026Q3", years=5,
                             facts_dir=Path("data/deep_facts"))
    deep = run_deep_analysis("兆易创新", "2026Q3", facts,
                             external_sources=bundle.ledger_kwargs())
    deep.ledger_gaps.extend(bundle.gaps)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from ai_agent.report.deep.schema import DataGap

# 文档类型 → 默认消费章节（10 章编号）。用于覆盖度盘点与缺口归属。
DOC_CHAPTERS: dict[str, tuple[int, ...]] = {
    "annual": (3, 4, 6, 7, 9),
    "audit": (2, 3),
    "inquiry": (2, 5, 9),
    "periodic": (3, 4),
    "industry_stat": (6, 9),
    "other": (),
}
# 公司模式必需的三类源文件；行业模式额外需要行业统计。
COMPANY_REQUIRED = ("annual", "audit", "inquiry")
INDUSTRY_REQUIRED = ("industry_stat",)
DEFAULT_YEARS = 5
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_FACTS = 500
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_VALID_LEVELS = frozenset({"L1", "L2", "L3", "L4", "L5", "L6"})

# 分类关键词按优先级排列：问询函回复里也会出现「审计报告」字样，
# 先判函件类才不会把回复归到审计类。
_DOC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("inquiry", re.compile(r"问询函|监管函|关注函|问询回复|回复函|反馈意见")),
    ("audit", re.compile(r"审计报告|关键审计事项|审计意见|KAM|会计师", re.I)),
    ("industry_stat", re.compile(r"行业协会|统计局|产能统计|价格指数|行业统计|协会发布")),
    ("annual", re.compile(r"年度报告|年报|年度审计")),
    ("periodic", re.compile(r"季度报告|季报|半年度|半年报|中期报告|临时公告")),
)
_REQUIRED_GAP: dict[str, tuple[list[int], str, str]] = {
    "annual": (
        [3, 4, 6, 7, 9],
        "杜邦拆解、分部收入、上下游与护城河缺少近 5 年纵向数据",
        "入库近 5 年年度报告全文，并把附注数字抽成结构化事实清单",
    ),
    "audit": (
        [2, 3],
        "第 2 章财报可信度无法评级（缺审计意见、KAM、审计机构变更）",
        "入库近 5 年审计报告与关键审计事项原文",
    ),
    "inquiry": (
        [2, 5, 9],
        "无法交叉验证监管问询与管理层回应的一致性",
        "入库交易所问询函及公司回复原文",
    ),
    "industry_stat": (
        [6, 9],
        "第 6 章供给结构、CR5 与对照组净效果无法量化",
        "接入合规授权的行业原始统计（L3），L4 二手数据须回溯到 L1-L3",
    ),
}


def classify_doc(filename: str, doc_type: str = "", description: str = "") -> str:
    """按文件名 / 库内 doc_type / 描述识别文档类型，识别不了返回 ``other``。"""
    blob = " ".join(str(item or "") for item in (filename, doc_type, description))
    for kind, pattern in _DOC_PATTERNS:
        if pattern.search(blob):
            return kind
    return "other"


def year_of(text: str) -> int:
    """从 period 或文件名里取第一个四位年份；取不到返回 0。"""
    match = _YEAR_RE.search(str(text or ""))
    return int(match.group(0)) if match else 0


def window_years(period: str, years: int, *, fallback_now: bool = True) -> tuple[int, int, bool]:
    """返回 (起始年, 截止年, 是否用了当前年份兜底)。"""
    base = year_of(period)
    assumed = False
    if not base and fallback_now:
        base = datetime.now().year
        assumed = True
    span = max(int(years), 1)
    return base - span + 1, base, assumed


@dataclass(frozen=True)
class ExternalFact:
    """一条外部结构化事实。字段与 ``SourceLedger.add_external`` 一一对应。

    ``value`` 必须是源文里出现过的原文数字/表述，不允许在这里做二次计算——
    台账的数字白名单会用它做校验，模型心算的数一律丢弃。
    """

    item: str
    value: str
    source_file: str
    doc_type: str = "annual"
    level: str = "L1"
    location: str = ""
    quote: str = ""
    backtrace_to: str = ""
    subject: str = ""
    period: str = ""
    acquisition: str = ""
    chapters: tuple[int, ...] = ()

    def ledger_kwargs(self) -> dict[str, Any]:
        """转成 ``add_external(**kwargs)`` 可直接消费的字典。"""
        return {
            "item": self.item,
            "value": self.value,
            "source_file": self.source_file,
            "location": self.location,
            "level": self.level,
            "acquisition": self.acquisition or f"外部数据源/{self.doc_type}",
            "quote": self.quote or f"{self.item}={self.value}",
            "backtrace_to": self.backtrace_to,
            "subject": self.subject,
            "period": self.period,
            "chapters": list(self.chapters or DOC_CHAPTERS.get(self.doc_type, ())),
        }

    @classmethod
    def from_row(cls, row: dict, *, default_doc_type: str = "annual") -> ExternalFact | None:
        """把外部清单的一行转成事实；必填字段缺失返回 None（由调用方记缺口）。"""
        if not isinstance(row, dict):
            return None
        item = str(row.get("item") or "").strip()
        value = str(row.get("value") or "").strip()
        source_file = str(row.get("source_file") or row.get("filename") or "").strip()
        if not item or not value or not source_file:
            return None
        level = str(row.get("level") or "L1").upper()
        if level not in _VALID_LEVELS:
            level = "L4"  # 未知等级按最保守的二手数据处理，交给台账回溯门禁
        chapters = row.get("chapters") or ()
        if isinstance(chapters, (list, tuple)):
            chapter_tuple = tuple(int(n) for n in chapters if str(n).isdigit())
        else:
            chapter_tuple = ()
        return cls(
            item=item[:120],
            value=value[:120],
            source_file=source_file[:200],
            doc_type=str(row.get("doc_type") or default_doc_type),
            level=level,
            location=str(row.get("location") or "")[:120],
            quote=str(row.get("quote") or "")[:400],
            backtrace_to=str(row.get("backtrace_to") or "")[:200],
            subject=str(row.get("subject") or "")[:80],
            period=str(row.get("period") or "")[:40],
            acquisition=str(row.get("acquisition") or "")[:120],
            chapters=chapter_tuple,
        )


@dataclass(frozen=True)
class SourceDocument:
    """知识库里一份可用的源头文件（有文件 ≠ 已抽出可溯源事实）。"""

    filename: str
    doc_type: str
    subject: str = ""
    period: str = ""
    file_id: str = ""
    level: str = "L1"
    chapters: tuple[int, ...] = ()


@dataclass
class SourceBundle:
    """一次数据采集的结果：事实 + 源文件盘点 + 缺口 + 执行说明。"""

    facts: list[ExternalFact] = field(default_factory=list)
    documents: list[SourceDocument] = field(default_factory=list)
    gaps: list[DataGap] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def ledger_kwargs(self) -> list[dict[str, Any]]:
        return [fact.ledger_kwargs() for fact in self.facts]

    def coverage(self) -> dict[int, list[str]]:
        """章节 → 可用源文件名。供执行报告说明「哪几章有料」。"""
        result: dict[int, list[str]] = {}
        for document in self.documents:
            for number in document.chapters or DOC_CHAPTERS.get(document.doc_type, ()):
                result.setdefault(number, []).append(document.filename)
        for fact in self.facts:
            for number in fact.chapters or DOC_CHAPTERS.get(fact.doc_type, ()):
                result.setdefault(number, []).append(fact.source_file)
        return {key: list(dict.fromkeys(value)) for key, value in result.items()}

    def missing_doc_types(self, mode: str = "company") -> list[str]:
        required = COMPANY_REQUIRED if mode != "industry" else INDUSTRY_REQUIRED
        present = {document.doc_type for document in self.documents}
        present |= {fact.doc_type for fact in self.facts}
        return [kind for kind in required if kind not in present]

    def requirement_gaps(self, mode: str = "company") -> list[DataGap]:
        """把「缺哪类文件」变成带章节归属和处理办法的具体缺口。"""
        return [
            DataGap(
                item=f"近 5 年{kind}原文未入库",
                chapters=list(_REQUIRED_GAP[kind][0]),
                impact=_REQUIRED_GAP[kind][1],
                handling=_REQUIRED_GAP[kind][2],
            )
            for kind in self.missing_doc_types(mode)
            if kind in _REQUIRED_GAP
        ]


class SourceProvider(Protocol):
    """外部数据源协议。实现方自行决定从哪里取数，但必须如实声明缺口。"""

    def collect(self, target: str, period: str, *, years: int = DEFAULT_YEARS) -> SourceBundle:
        ...


def _default_list_files() -> list[dict]:
    """延迟导入 store，避免 import 本模块就触发数据库连接。"""
    from ai_agent.storage.store import store

    return list(store.list_kb_files())


class KnowledgeBaseProvider:
    """盘点知识库里近 N 年的源头文件（只读，不下载、不解析 PDF）。

    只做「源文件覆盖度」：库里有年报 PDF 不等于已经抽出附注数字，因此本
    provider **不产生 ExternalFact**，只产生 SourceDocument 与缺口。真正让
    数字进台账的是 ``LocalFactsProvider``。
    """

    def __init__(
        self,
        list_files: Callable[[], Sequence[dict]] | None = None,
        *,
        require_status: str = "ready",
    ) -> None:
        if list_files is not None and not callable(list_files):
            # 快速失败：误传 list[dict] 会在 collect 里被当成「读取失败」记成缺口，
            # 静默掩盖调用错误，比直接抛异常更难排查。
            raise TypeError("list_files 必须是无参 callable，返回 list[dict]")
        self._list_files = list_files or _default_list_files
        self.require_status = require_status

    def collect(self, target: str, period: str, *, years: int = DEFAULT_YEARS) -> SourceBundle:
        bundle = SourceBundle()
        start, end, assumed = window_years(period, years)
        if assumed:
            bundle.notes.append(f"需求未写明年份，近 {years} 年窗口按当前年份 {end} 推算")
        try:
            rows = self._list_files()
        except Exception as exc:
            bundle.notes.append(f"知识库文件列表读取失败，覆盖度盘点跳过：{type(exc).__name__}")
            bundle.gaps.append(
                DataGap(
                    item="近 5 年源头文件盘点",
                    chapters=[2, 3, 4, 6, 7, 9],
                    impact="无法判断哪些章节有 L1 原始文件支撑",
                    handling="确认知识库可读后重新运行；本轮不假设任何源文件存在",
                )
            )
            return bundle

        token = (target or "").strip()
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if self.require_status and str(row.get("status") or "") != self.require_status:
                continue
            filename = str(row.get("filename") or "")
            subject = str(row.get("subject") or "")
            aliases = " ".join(str(item) for item in (row.get("subject_aliases") or []))
            blob = f"{subject} {aliases} {filename}"
            if token and token not in blob:
                continue
            row_period = str(row.get("period") or "")
            year = year_of(row_period) or year_of(filename)
            if year and not (start <= year <= end):
                continue
            kind = classify_doc(filename, str(row.get("doc_type") or ""),
                                str(row.get("description") or ""))
            bundle.documents.append(
                SourceDocument(
                    filename=filename,
                    doc_type=kind,
                    subject=subject or token,
                    period=row_period or (str(year) if year else ""),
                    file_id=str(row.get("file_id") or ""),
                    # 知识库里的公告原文属发行人/交易所原始披露；
                    # 行业统计若来自第三方文件则按 L3（需回溯）。
                    level="L3" if kind == "industry_stat" else "L1",
                    chapters=DOC_CHAPTERS.get(kind, ()),
                )
            )
        bundle.notes.append(
            f"知识库盘点：近 {years} 年（{start}-{end}）命中 {len(bundle.documents)} 份源文件；"
            "源文件仅代表可回溯出处，未抽取的结构化事实不计入台账"
        )
        bundle.gaps.extend(bundle.requirement_gaps("company"))
        if not bundle.documents:
            bundle.gaps.append(
                DataGap(
                    item=f"{token or '标的'}近 {years} 年源头文件",
                    chapters=[2, 3, 4, 6, 7, 9],
                    impact="除本期财报底稿外无纵向/佐证材料，深度分析只能停留在骨架",
                    handling="先入库年报/审计报告/问询函，再重跑深度分析",
                )
            )
        return bundle


class LocalFactsProvider:
    """读取本地结构化事实清单（``.json`` / ``.jsonl``），是数字进台账的通道。

    清单每行/每项的字段见 ``ExternalFact.from_row``。只扫描目录顶层，
    单文件 ≤2MB、总事实 ≤500 条，避免一个误放的大文件拖垮整份报告。
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def collect(self, target: str, period: str, *, years: int = DEFAULT_YEARS) -> SourceBundle:
        bundle = SourceBundle()
        start, end, assumed = window_years(period, years)
        if assumed:
            bundle.notes.append(f"事实清单未指定基准年，按当前年份 {end} 做近 {years} 年筛选")
        if not self.directory.is_dir():
            bundle.notes.append(f"事实清单目录不存在，跳过：{self.directory}")
            bundle.gaps.append(
                DataGap(
                    item="外部结构化事实清单",
                    chapters=[2, 3, 4, 6, 7, 9],
                    impact="年报附注/审计/问询函的数字未进入台账，相关章节无法量化",
                    handling="把已核对原文的条目按 schema 落到该目录后重跑",
                )
            )
            return bundle

        paths = sorted(
            path for path in self.directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        )
        dropped = 0
        for path in paths:
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    bundle.notes.append(f"事实清单超过 2MB 已跳过：{path.name}")
                    continue
                rows = _read_rows(path)
            except Exception as exc:
                bundle.notes.append(f"事实清单解析失败 {path.name}：{type(exc).__name__}")
                continue
            for row in rows:
                fact = ExternalFact.from_row(row, default_doc_type=classify_doc(path.name))
                if fact is None:
                    dropped += 1
                    continue
                if fact.subject and target and target not in fact.subject:
                    continue
                fact_year = year_of(fact.period) or year_of(fact.source_file)
                if fact_year and not (start <= fact_year <= end):
                    continue
                if len(bundle.facts) >= _MAX_FACTS:
                    bundle.notes.append(f"事实条数达到上限 {_MAX_FACTS}，其余已跳过")
                    break
                bundle.facts.append(fact)
        if dropped:
            bundle.notes.append(f"事实清单里 {dropped} 条缺 item/value/source_file，已丢弃")
        bundle.notes.append(
            f"本地事实清单：{len(paths)} 个文件、{len(bundle.facts)} 条入账候选"
            "（L4/L5 仍须过台账回溯门禁）"
        )
        return bundle


def _read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line) for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    payload = json.loads(text)
    if isinstance(payload, dict):
        rows = payload.get("facts", payload.get("items", []))
    else:
        rows = payload
    return [row for row in rows if isinstance(row, dict)]


class IndustryStatProvider:
    """行业统计注入点：**本层不内置任何第三方端点**。

    ``fetcher(target, period, years) -> Sequence[dict]``，返回行按
    ``ExternalFact.from_row`` 的字段规范；每条必须自带 ``source_file``
    与 ``level``，L4/L5 还必须给 ``backtrace_to``，否则台账会拒绝入表。
    未注入 fetcher 时只产出一条明确缺口，不假装数据存在。
    """

    def __init__(
        self,
        fetcher: Callable[[str, str, int], Sequence[dict]] | None = None,
    ) -> None:
        self.fetcher = fetcher

    def collect(self, target: str, period: str, *, years: int = DEFAULT_YEARS) -> SourceBundle:
        bundle = SourceBundle()
        if self.fetcher is None:
            bundle.notes.append(
                "行业统计数据源未接入：本层不内置第三方 API，未注入 fetcher 即声明缺口"
            )
            bundle.gaps.append(
                DataGap(
                    item="行业产能/份额/价格战等格局数据（L3 原始统计）",
                    chapters=[6, 9],
                    impact="第 6 章供给结构与对照组净效果无法量化",
                    handling="接入合规授权的行业协会/统计局原始发布；L4 二手数据须回溯 L1-L3",
                )
            )
            return bundle
        try:
            rows = self.fetcher(target, period, years)
        except Exception as exc:
            bundle.notes.append(f"行业统计 fetcher 失败：{type(exc).__name__}: {exc}")
            bundle.gaps.append(
                DataGap(
                    item="行业统计本次拉取失败",
                    chapters=[6, 9],
                    impact="本轮无行业格局数据，不得用上期值或常识填补",
                    handling="检查数据源授权与网络后重跑",
                )
            )
            return bundle
        dropped = 0
        for row in rows or []:
            fact = ExternalFact.from_row(row, default_doc_type="industry_stat")
            if fact is None:
                dropped += 1
                continue
            bundle.facts.append(fact)
            bundle.documents.append(
                SourceDocument(
                    filename=fact.source_file,
                    doc_type="industry_stat",
                    subject=fact.subject or target,
                    period=fact.period,
                    level=fact.level,
                    chapters=DOC_CHAPTERS["industry_stat"],
                )
            )
        if dropped:
            bundle.notes.append(f"行业统计 {dropped} 条缺必填字段，已丢弃")
        bundle.notes.append(f"行业统计 fetcher 返回 {len(bundle.facts)} 条候选")
        return bundle


def collect_sources(
    target: str,
    period: str,
    *,
    years: int = DEFAULT_YEARS,
    mode: str = "company",
    facts_dir: Path | str | None = None,
    list_files: Callable[[], Sequence[dict]] | None = None,
    stat_fetcher: Callable[[str, str, int], Sequence[dict]] | None = None,
    providers: Sequence[SourceProvider] | None = None,
) -> SourceBundle:
    """组合采集入口。默认三类 provider 全跑；也可直接注入自定义 provider 列表。

    ``mode="industry"`` 时只声明行业统计缺口，不强求单个公司的年报/审计文件。
    """
    active: list[SourceProvider] = list(providers) if providers is not None else [
        KnowledgeBaseProvider(list_files),
        LocalFactsProvider(facts_dir) if facts_dir is not None else _EmptyProvider(),
        IndustryStatProvider(stat_fetcher),
    ]
    merged = SourceBundle()
    for provider in active:
        try:
            bundle = provider.collect(target, period, years=years)
        except Exception as exc:
            merged.notes.append(f"{type(provider).__name__} 采集失败：{type(exc).__name__}: {exc}")
            continue
        merged.facts.extend(bundle.facts)
        merged.documents.extend(bundle.documents)
        merged.notes.extend(bundle.notes)
        # 各 provider 自行声明的通用缺口（如「源文件盘点失败」）保留；
        # 「必需文档类型」类缺口统一在最后按合并结果重算，避免重复报缺。
        merged.gaps.extend(
            gap for gap in bundle.gaps if not str(gap.item).endswith("原文未入库")
        )
    merged.gaps.extend(merged.requirement_gaps(mode))
    coverage = merged.coverage()
    if coverage:
        merged.notes.append(
            "章节源文件覆盖：" + "；".join(
                f"第{n}章 {len(names)} 份" for n, names in sorted(coverage.items())
            )
        )
    return merged


class _EmptyProvider:
    """facts_dir 未提供时的空实现，避免组合入口里写 None 分支。"""

    def collect(self, target: str, period: str, *, years: int = DEFAULT_YEARS) -> SourceBundle:
        return SourceBundle(notes=["未提供本地事实清单目录，结构化事实注入通道未启用"])
