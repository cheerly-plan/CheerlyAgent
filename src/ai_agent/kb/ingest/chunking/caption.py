"""表题与报表口径：识别表题，并把表题向下带给续片。

由原 chunking.py 拆分而来，逻辑未改动。
"""

import re

from ai_agent.kb.ingest.parsing import Segment, TABLE


def _scope_from_caption(text: str) -> str | None:
    """从表题推报表口径：``母公司资产负债表`` → parent、``合并利润表`` → consolidated。

    只看**第一行**。整段里出现「归属于母公司所有者的净利润」是科目名，不代表
    这张表是母公司口径——按全文匹配会把合并报表误标成 parent。
    """
    head = next((ln.strip() for ln in (text or "").splitlines() if ln.strip()), "")
    if head.startswith("母公司"):
        return "parent"
    if head.startswith("合并"):
        return "consolidated"
    return None

def _carry_md_table_captions(segments: list[Segment]) -> list[Segment]:
    """给 Markdown 表格补回表题——表题常常留在上一页的正文段里。

    实测《中国能源建设 2026Q3》：p13 的正文段以「母公司资产负债表」收尾，
    表格本体从 p14 才开始印列名。结果是 p14 的切片只剩

        | 项目 | 2026年9月30日 | 2025年12月31日 |
        | 货币资金 | 2,696,401 | 770,790 |

    **表名整片丢失**，向量里没有任何「资产负债表」的信号，问「母公司货币
    资金」时排到母公司现金流量表后面（后者正文里带「货币资金」字样）。

    HTML 表的同类问题由 ``_carry_html_page_headers`` 处理；这里只补 Markdown
    表，且只在四件事同时成立时补：上一个正文段以表题收尾、这一段的表开头
    就是列名、章节没变、表题还没被别处覆盖。
    """
    last_caption = ""
    last_section: str | None = None
    out: list[Segment] = []
    for seg in segments:
        if seg.kind != TABLE:
            text = seg.text or ""
            if seg.section != last_section:
                last_caption = ""
                last_section = seg.section
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            caption = _caption_at_tail(lines)
            if caption:
                last_caption = caption
            out.append(seg)
            continue

        body = seg.text or ""
        if seg.section != last_section:
            last_caption = ""
            last_section = seg.section
        lines = [ln for ln in body.splitlines() if ln.strip()]
        has_caption = bool(lines) and "|" not in lines[0] and not lines[0].lstrip().startswith("<")
        if has_caption:
            last_caption = lines[0].strip()
            out.append(seg)
            continue
        is_md_table = bool(lines) and "|" in lines[0]
        if last_caption and is_md_table:
            seg = Segment(
                text=f"{last_caption}\n{body}",
                page=seg.page,
                kind=seg.kind,
                section=seg.section,
                scope=seg.scope,
                page_end=seg.page_end,
            )
        out.append(seg)
    return out

def _caption_at_tail(lines: list[str]) -> str:
    """正文段末尾的表题（可带一行单位/币种）。找不到返回空串。"""
    if not lines:
        return ""
    for index in range(len(lines) - 1, max(-1, len(lines) - 8), -1):
        line = lines[index]
        if not _CAPTION_LINE.match(line):
            continue
        extra = ""
        if index + 1 < len(lines) and _UNIT_LINE.match(lines[index + 1]):
            extra = f"\n{lines[index + 1]}"
        return f"{line}{extra}"
    return ""

# 表题行：合并资产负债表 / 母公司利润表 / 现金流量表（续）…
_CAPTION_LINE = re.compile(r"^(?:合并|母公司)?[\u4e00-\u9fa5]{1,14}表(?:（续）|\(续\))?$")

_UNIT_LINE = re.compile(r"^(?:单位|币种)[:：]")
