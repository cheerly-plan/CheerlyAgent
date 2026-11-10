"""切片体检：解析器的自检规则。

解析出来的 Markdown 表格是「数字事实」的唯一载体，但 PDF 排版痕迹会以两种
方式污染它：

* **静默错数**：数字被换行/单元格切开成「40312292 2」（真值 403122922），
  或者一张表少了表头、列名变成「列2」，值就归到了错误的列。这类错误不会被
  任何下游校验发现，却会让研报里的金额差 10 倍。
* **垃圾切片**：页脚页码「6 / 12」单独成片。

本模块只做**只读判定**，不修数据：`scripts/kb.py doctor` 用它给文件打健康分，
解析流程用它写一条告警日志。修不修由人决定（重解析 / 重新上传）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 单元格里「长数字 空格 1~3 位数字」：千分位或小数点被排版甩开
_SPLIT_NUMBER = re.compile(r"^\s*-?[\d,]{4,}\s+\d{1,3}\s*$")
# 整片就是一个页码：6 / 12
_PAGE_MARK = re.compile(r"^\s*\d+\s*[/／]\s*\d+\s*$")
# 占位列头：| 列2 | 列3 |
_PLACEHOLDER = re.compile(r"\|\s*列\d+\s*\|")
_NUM = re.compile(r"-?[\d,]+(?:\.\d+)?")
_ROW_LABEL = re.compile(r"^\|\s*([^|]+?)\s*\|")


@dataclass
class FileIssues:
    """一份文件的体检结果。"""

    chunks: int = 0
    page_marks: int = 0
    placeholder_tables: int = 0
    split_numbers: list[str] = field(default_factory=list)
    identity_failures: list[str] = field(default_factory=list)
    tiny_chunks: int = 0

    @property
    def ok(self) -> bool:
        return not (
            self.page_marks
            or self.placeholder_tables
            or self.split_numbers
            or self.identity_failures
        )

    def summary(self) -> str:
        bits = []
        if self.page_marks:
            bits.append(f"页码垃圾片 {self.page_marks}")
        if self.split_numbers:
            bits.append(f"数字被劈开 {len(self.split_numbers)}")
        if self.placeholder_tables:
            bits.append(f"占位列头表 {self.placeholder_tables}")
        if self.identity_failures:
            bits.append(f"恒等式不成立 {len(self.identity_failures)}")
        return "、".join(bits) if bits else "正常"


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _to_float(text: str) -> float | None:
    raw = text.replace(",", "").strip()
    if not raw or not _NUM.fullmatch(raw):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _table_rows(text: str) -> list[list[str]]:
    return [_cells(line) for line in text.splitlines() if line.startswith("|")]


def _balance_rows(text: str) -> dict[str, list[list[str]]]:
    """抽出一份切片里的「资产总计 / 负债和所有者权益总计」行（按列宽分组）。"""
    found: dict[str, list[list[str]]] = {}
    for row in _table_rows(text):
        if not row:
            continue
        label = row[0]
        key = ""
        if label.startswith("资产总计"):
            key = "assets"
        elif label.startswith("负债和所有者权益") and "总计" in label:
            key = "liabilities"
        if key:
            found.setdefault(key, []).append(row[1:])
    return found


def check_balance_identity(texts: list[str]) -> list[str]:
    """资产负债表勾稽：资产总计 == 负债和所有者权益总计。

    在**整份文件**范围里比对：合并资产负债表常被切成多片，两张总额行未必落在
    同一片里。为了不把母公司报表和合并报表混在一起比，按「列数 + 是否带附注列」
    分组，且只在该组里两种行**各恰好一条**时才判定——出现两条说明文件里既有
    合并表又有母公司表，凑不齐对应关系，宁可不判。
    """
    groups: dict[tuple[int, bool], dict[str, list[list[str]]]] = {}
    for text in texts:
        for key, rows in _balance_rows(text).items():
            for row in rows:
                group_key = (len(row), any(cell for cell in row))
                groups.setdefault(group_key, {}).setdefault(key, []).append(row)
    problems: list[str] = []
    for (_width, _filled), bucket in groups.items():
        assets = bucket.get("assets") or []
        liabilities = bucket.get("liabilities") or []
        if len(assets) != 1 or len(liabilities) != 1:
            continue
        for index, (left, right) in enumerate(zip(assets[0], liabilities[0])):
            a, b = _to_float(left), _to_float(right)
            if a is None or b is None:
                continue
            if abs(a - b) > 0.5:  # 四舍五入到分的误差容限
                problems.append(
                    f"第 {index + 1} 个数列：资产总计 {a:,.2f} ≠ "
                    f"负债和所有者权益总计 {b:,.2f}"
                )
    return problems


def check_chunks(texts: list[str]) -> FileIssues:
    """对一份文件的全部切片做体检。"""
    issues = FileIssues(chunks=len(texts))
    for text in texts:
        stripped = text.strip()
        if _PAGE_MARK.match(stripped):
            issues.page_marks += 1
        if len(stripped) < 60:
            issues.tiny_chunks += 1
        if _PLACEHOLDER.search(text):
            issues.placeholder_tables += 1
        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            for cell in _cells(line):
                if _SPLIT_NUMBER.match(cell):
                    issues.split_numbers.append(cell)
    issues.identity_failures = check_balance_identity(texts)
    return issues