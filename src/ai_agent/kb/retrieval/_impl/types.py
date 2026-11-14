"""命中结构与引用位置：Hit 数据类、人读位置文案。

由原 retrieval.py 拆分而来，逻辑未改动。
"""

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass, field

from ai_agent.kb.ingest.parsing import FIGURE
from ai_agent.kb.ingest.parsing import TABLE


# 测试用 monkeypatch.setattr(retrieval, NAME, ...) 打补丁的知识点：
# 直接 NAME(...) 走的是本模块 globals，绕过模块属性，补丁会失效；
# 统一经 _public() 取「对外模块」的当前属性再调用，补丁才生效。
def _public():
    import ai_agent.kb.retrieval.retrieval as public

    return public


@dataclass
class Hit:
    """一条命中：命中片本体 + 前后邻接片段（拼上下文用）。"""

    file_id: str
    filename: str
    seq: int
    page: int | None
    kind: str
    text: str
    # 向量余弦距离；关键词独有命中没有走过向量路，保持 None，不能填 0。
    distance: float | None
    # 跨页切片的末页（同节内聚合会一片跨两页）；单页片等于 page
    page_end: int | None = None
    # 所属一级节，如「第十节 管理层讨论与分析」
    section: str | None = None
    # 报表口径：consolidated / parent / None（非报表或表里没印表题）
    scope: str | None = None
    # 文件报告期，如 2025Q3。正文表格常常不写年份，回答时靠它区分两年。
    period: str = ""
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)

    @property
    def where(self) -> str:
        """引用位置：图表标图像识别，表格按文件类型区分页码/表号。"""
        return format_where(
            kind=self.kind,
            page=self.page,
            page_end=self.page_end,
            seq=self.seq,
            filename=self.filename,
        )

    @property
    def label(self) -> str:
        """完整出处：位置 + 章节（有章节时）。"""
        return f"{self.where} · {self.section}" if self.section else self.where

def format_where(
    *,
    kind: str,
    page: int | None,
    page_end: int | None = None,
    seq: int = 0,
    filename: str = "",
    mime: str = "",
) -> str:
    """人读的引用位置：PDF 表格落在页上，只有电子表才叫「表N」。"""
    if kind == FIGURE:
        return f"第{page}页图表" if page is not None else "图表"
    if kind == TABLE:
        if page is None:
            return "表格"
        if is_sheet_file(filename, mime):
            return f"表{page}"
        if page_end is not None and page_end != page:
            return f"第{page}–{page_end}页"
        return f"第{page}页"
    if page is None:
        return f"第{seq + 1}段"
    if page_end is not None and page_end != page:
        return f"第{page}–{page_end}页"
    return f"第{page}页"

def _where(row: dict) -> str:
    return format_where(
        kind=row.get("kind") or "text",
        page=row.get("page"),
        page_end=row.get("page_end"),
        seq=int(row.get("seq") or 0),
        filename=row.get("filename") or "",
        mime=row.get("mime") or "",
    )

def is_sheet_file(filename: str = "", mime: str = "") -> bool:
    suffix = Path(filename or "").suffix.lower()
    ext = (mime or "").lower()
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    return suffix in _SHEET_EXTS or ext in _SHEET_EXTS

# Excel 的 page 是工作表序号；PDF/Word 的 page 是页码。展示名不能混用「表N」。
_SHEET_EXTS = {".xlsx", ".xlsm", ".xls", ".csv"}

# 查询里的功能词/疑问词，几乎每条查询都有、对召回没有区分度
_STOPWORDS = {
    "怎么", "如何", "什么", "哪些", "多少", "是否", "请问", "查询",
    "查找", "检索", "相关", "关于", "介绍", "说明", "可以", "需要",
    "应该", "如果", "我们", "他们", "以及", "还有", "一下", "这些",
    "文档", "文件", "知识库", "资料",
}
