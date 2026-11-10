"""数据溯源台账（L1-L6 源头优先级 + Tier 分级 + 缺口清单）。

把 report/extract.py 的 ``IndicatorFact`` / ``Evidence`` 适配成深度分析层自己的
``SourceRecord``：不修改事实层 dataclass，靠 duck typing 读字段，测试可传 stub。

落实 data-sourcing-protocol.md 的三条硬规则：

* 每个量化数字可回答「最初出现在哪个文件」→ ``SourceRecord`` 带文件/页码/原文；
* 禁止间接引用 → L4/L5 的二手数据若无 ``backtrace_to``，不进数据来源表，只记缺口；
* 缺口显式声明 → ``SourceLedger.gaps`` 贯穿全流程，禁止臆测填补。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai_agent.report.deep.schema import DataGap, DataSourceRow

# L1-L3 是可直接采信的源头；L4 需交叉验证；L5 仅线索；L6 不采信。
LEVEL_TIER = {
    "L1": "Tier 1",
    "L2": "Tier 1",
    "L3": "Tier 1",
    "L4": "Tier 2",
    "L5": "Tier 3",
    "L6": "Tier 4",
}
# 跨模块共用的数字模式（分析层做正文数字白名单校验时要 import）。
NUMBER_RE = re.compile(r"-?\d[\d,，]*(?:[ \t\u00a0]\d{3})*(?:\.\d+)?%?")


def normalize_number(raw: str) -> str:
    """数字白名单归一：去千分位/百分号，按数值比较，避免格式差异误杀。"""
    text = raw.replace(",", "").replace("，", "").replace("%", "").strip().lstrip("+")
    text = re.sub(r"[ \t\u00a0]", "", text)
    try:
        value = float(text)
    except ValueError:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def numbers_in(text: str) -> set[str]:
    return {
        key
        for match in NUMBER_RE.finditer(text or "")
        if (key := normalize_number(match.group(0)))
    }


# 金额单位换算白名单：只把「台账里真实存在的金额」按固定倍率换成另一种单位写法。
# 事实层原文以「元」为主，模型却习惯写「亿元」；此前这些等价写法全被
# numbers_allowed 判成台账外数字遮蔽掉，评审意见也指出结论出现成片的数据空白。
# 这里不做任何计算派生（同比、差值、占比一律不放行），只做单位换算。
_AMOUNT_UNITS = {"元": 1.0, "万元": 1e4, "亿元": 1e8}
_DERIVED_PRECISIONS = (0, 1, 2, 3, 4, 6)
_VALUE_WITH_UNIT = re.compile(r"^\s*(-?\d[\d,，]*(?:\.\d+)?)\s*(亿元|万元|元|%)?\s*$")


def derived_number_forms(value: str, unit: str) -> set[str]:
    """台账金额数值的等价单位写法（元 / 万元 / 亿元），按常见书写精度归一。

    只有 ``value`` 是单个数字、且单位属于金额单位时才派生；百分比、比率、
    「倍」「人」等一律跳过。负金额同时派生绝对值写法（「亏损 2.05 亿元」）。
    返回值是归一化后的数字键，可直接并入白名单。非台账来源的数字、模型
    心算出来的同比/差值都不会因为本函数被放行。
    """
    match = _VALUE_WITH_UNIT.match(str(value or ""))
    if not match:
        return set()
    unit_text = (str(unit or "").strip() or match.group(2) or "").strip()
    scale = _AMOUNT_UNITS.get(unit_text)
    if scale is None:
        return set()
    try:
        base_yuan = float(match.group(1).replace(",", "").replace("，", "")) * scale
    except ValueError:
        return set()
    forms: set[str] = set()
    for target_scale in _AMOUNT_UNITS.values():
        converted = base_yuan / target_scale
        # 负金额常被写成「亏损 2.05 亿元」「净流出 3.46 亿元」：方向由语义词
        # 承担，数值取绝对值。这也是台账数值的等价写法，不应被当成编造数字。
        magnitudes = (converted, abs(converted)) if converted < 0 else (converted,)
        for magnitude in magnitudes:
            for precision in _DERIVED_PRECISIONS:
                key = normalize_number(f"{magnitude:.{precision}f}")
                if key:
                    forms.add(key)
    return forms


@dataclass
class SourceRecord:
    """一条可溯源事实。``e0`` 这类短 id 只在本层内使用，不覆盖事实层的引用锚点。"""

    source_id: str
    item: str
    value: str
    unit: str
    subject: str
    period: str
    filename: str
    file_id: str
    page: int | None
    section: str
    quote: str
    level: str = "L1"
    acquisition: str = "知识库定期报告原文"
    backtrace_to: str = ""
    # 外部结构化事实清单登记时置 True，并记下它服务的章节；仅用于证据包
    # 超限时的挑选，不改变台账本身与数字白名单。
    external: bool = False
    chapters: tuple[int, ...] = ()

    @property
    def tier(self) -> str:
        return LEVEL_TIER.get(self.level, "Tier 5")

    @property
    def location(self) -> str:
        page = f"第{self.page}页" if self.page is not None else "页码未标注"
        section = self.section or "章节未标注"
        return f"{page} / {section}"

    def to_row(self) -> DataSourceRow:
        return DataSourceRow(
            item=f"{self.item} {self.value}{self.unit}".strip(),
            source_file=self.filename or self.file_id or "（未标注源头文件）",
            location=self.location,
            tier=self.tier,
            acquisition=self.acquisition,
        )

    def render(self) -> str:
        quote = " ".join((self.quote or "").split())[:200]
        return (
            f"- {self.source_id}｜{self.subject} {self.period}｜"
            f"{self.item}={self.value}{self.unit}｜{self.tier}/{self.level}｜"
            f"{self.filename}，{self.location}｜原文：{quote}"
        )


@dataclass
class SourceLedger:
    records: list[SourceRecord] = field(default_factory=list)
    gaps: list[DataGap] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @classmethod
    def from_facts(
        cls,
        facts: list,
        *,
        level: str = "L1",
        acquisition: str = "知识库定期报告原文",
    ) -> SourceLedger:
        """适配 ``IndicatorFact``：只读属性，不 import 具体类。"""
        ledger = cls()
        for index, fact in enumerate(facts):
            evidence = getattr(fact, "evidence", None)
            page = getattr(evidence, "page", None)
            ledger.records.append(
                SourceRecord(
                    source_id=f"e{index}",
                    item=str(getattr(fact, "name", "") or "指标"),
                    value=str(getattr(fact, "value", "") or ""),
                    unit=str(getattr(fact, "unit", "") or ""),
                    subject=str(getattr(fact, "subject", "") or ""),
                    period=str(getattr(fact, "period", "") or ""),
                    filename=str(getattr(evidence, "filename", "") or ""),
                    file_id=str(getattr(evidence, "file_id", "") or ""),
                    page=int(page) if isinstance(page, int) else None,
                    section=str(getattr(evidence, "section", "") or ""),
                    quote=str(getattr(evidence, "quote", "") or ""),
                    level=level,
                    acquisition=acquisition,
                )
            )
        if not ledger.records:
            ledger.gaps.append(
                DataGap(
                    item="本期财报事实",
                    impact="无法开展任何章节验证",
                    handling="先补齐定期报告原文入库，再运行深度分析",
                )
            )
        return ledger

    def add_external(
        self,
        *,
        item: str,
        value: str,
        source_file: str,
        location: str = "",
        level: str = "L3",
        acquisition: str = "外部原始统计",
        quote: str = "",
        backtrace_to: str = "",
        subject: str = "",
        period: str = "",
        chapters: list[int] | None = None,
    ) -> bool:
        """登记外部数据。L4/L5 二手数据必须回溯到 L1-L3 源头，否则拒绝入表。

        返回是否入表；被拒的记入 ``rejected`` 与缺口清单，绝不静默丢弃。
        """
        if level in {"L4", "L5", "L6"} and not backtrace_to:
            self.rejected.append(f"{item}（{level} 未回溯到原始出处）")
            self.gaps.append(
                DataGap(
                    item=item,
                    chapters=chapters or [],
                    impact="二手数据未经引用链回溯，不得支撑结论",
                    handling="找到 L1-L3 源头文件后再入表",
                )
            )
            return False
        page_match = re.search(r"第(\d+)页", location or "")
        self.records.append(
            SourceRecord(
                source_id=f"e{len(self.records)}",
                item=item,
                value=value,
                unit="",
                subject=subject,
                period=period,
                filename=source_file,
                file_id="",
                page=int(page_match.group(1)) if page_match else None,
                section=location,
                quote=quote or f"{item}={value}",
                level=level,
                acquisition=acquisition,
                backtrace_to=backtrace_to,
                external=True,
                chapters=tuple(chapters or ()),
            )
        )
        return True

    @property
    def ids(self) -> set[str]:
        return {record.source_id for record in self.records}

    @property
    def allowed_numbers(self) -> set[str]:
        """正文数字白名单：台账出现过的值 + 台账金额的确定性单位换算写法。

        派生只来自 ``record.value`` 与金额单位，不从原文片段（页码、报告期、
        表格里的其它数）派生，也不含任何由模型计算出来的比率或差额。
        """
        allowed: set[str] = set()
        for record in self.records:
            allowed |= numbers_in(record.value)
            allowed |= numbers_in(record.quote)
            allowed |= derived_number_forms(record.value, record.unit)
        return allowed

    @property
    def unit_conversions(self) -> list[str]:
        """可审计的换算条目：哪些台账金额被允许写成万元 / 亿元。"""
        return [
            f"{record.source_id} {record.item}：{record.value}{record.unit}"
            for record in self.records
            if derived_number_forms(record.value, record.unit)
        ]

    def source_rows(self, source_ids: list[str] | None = None) -> list[DataSourceRow]:
        wanted = set(source_ids) if source_ids is not None else None
        return [
            record.to_row()
            for record in self.records
            if wanted is None or record.source_id in wanted
        ]

    def render_for(self, chapter_numbers: list[int] | None = None, *, limit: int = 40) -> str:
        """按章渲染证据包。

        记录数没超限（当前默认数据条件）时逐字沿用「按台账顺序全量展开」，
        行为与加外部事实通道之前完全一致。超限时才做挑选：优先「清单里
        声明服务本章」的外部事实，其次是其它外部事实，最后才是本期事实层。
        否则外部事实统一排在事实层之后，一旦清单变大就会被 ``limit`` 整段
        截掉，通道接了等于没接。挑完仍按原顺序输出，``eN`` 编号可对照。
        """
        note = ""
        if chapter_numbers:
            note = f"（供第 {','.join(str(n) for n in chapter_numbers)} 章使用）"
        if not self.records:
            return "（台账为空：只能写数据缺口，不得写任何数字或结论）"
        records = self.records
        if len(records) > limit:
            wanted = set(chapter_numbers or ())

            def priority(index: int) -> tuple[int, int]:
                record = records[index]
                if wanted and wanted.intersection(record.chapters):
                    rank = 2
                elif record.external:
                    rank = 1
                else:
                    rank = 0
                return rank, -index  # 同优先级下靠前的记录优先

            picked = sorted(
                range(len(records)), key=priority, reverse=True
            )[:limit]
            shown = [records[index] for index in sorted(picked)]
        else:
            shown = list(records)
        lines = [f"已授权事实台账{note}："] if note else "已授权事实台账："
        lines.extend(record.render() for record in shown)
        if len(records) > len(shown):
            lines.append(f"…另有 {len(records) - len(shown)} 条未展开")
        return "\n".join(lines)

    def gap_rows(self) -> list[DataGap]:
        return list(self.gaps)
