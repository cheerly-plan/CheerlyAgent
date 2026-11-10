"""文档切片。

两级策略：先按段落聚合到目标长度；单段超长时按句子边界（中英文
标点）二次切分。相邻切片保留 overlap 字符，避免关键信息恰好落在
边界上被截断。纯字符数切分（中文 1 字符 ≈ 1 token 量级），不上
语义切分——收益配不上复杂度。

按内容类型分派：
- text：上面的常规策略
- table：一个行项目一片，每片重复表头、表题和单位。「其中：」、加减行和
  合计行跟所属科目走，不单独成片。正文里的 HTML `<table>` 先整张抽出，
  kind 记成 table。续页没印列名时补上一页的列名，表题同样往下带。
  向量和重排读的是单元格文字，不读标签。给模型的检索和抽取把 HTML 表
  换成 Markdown 表，库里仍留 HTML，前端合并单元格不用重切。
- figure：图像描述本身很短，直接沿用常规策略（一般不触发切分）

两级栅栏，谁窄听谁的：
- **节栅栏**：同节的连续正文段先合成一条流再聚合，一片不跨章节。
  章节是语义的自然边界（"第十节 管理层讨论与分析"里不会混进"第五节
  风险因素"的句子），跨节拼接只会让引用张冠李戴；
- **页边界**：无章节信息的文档（扫描件、MD/TXT、Excel）仍按段独立切，
  不做跨页合并——页码是溯源的最低要求。

页码溯源：解析层产出 Segment（带 page / section / kind），切片后每片
携带自己的页码区间与类型（PDF 页码、Excel sheet 序号、或 None）。
"""

import html
import re
from dataclasses import dataclass, replace
from pathlib import Path

from ai_agent.kb.ingest.parsing import FIGURE, TABLE, TEXT, Segment

# Excel 的 page 是 sheet 序号；PDF/Word 的 page 是页码。文档里的「表格16」
# 是作者自己编的表号，解析层不会去数，引用只能落到页。
_SHEET_EXTS = {".xlsx", ".xlsm", ".xls", ".csv"}

# 段落边界：空行或单个换行都算；PDF 抽出来的文本换行很密，宽松处理
_PARA_SPLIT = re.compile(r"\n+")
# 视觉模型对「值得送审但实际没图」的页面常常照格式编一句结论，这些描述
# 不含任何可检索信息，进库只会给图谱喂假实体（"无图表"被当成机构名）。
# 只认「整句都在否认有图」的短描述，避免误伤真描述里带了"无"字的句子。
_NO_CHART = re.compile(
    r"^(图表类型[:：]\s*无|无图表|无图片|没有图表|无任何图表|图中无任何图表|无法总结)",
)
_NO_CHART_MAX = 120
# 目录页被过滤后常只剩「目 录」两字，单成一片是纯噪声
_TOC_RESIDUE = re.compile(r"^目\s*录$|^contents$", re.IGNORECASE)
# OCR 常把「续上表：」「(2) 应付项目」拆成独立正文段；并进下一张表，少出废片。
_TABLE_LEAD_MAX = 40
_TABLE_LEAD_HINT = re.compile(
    r"续上表|如下表?|见下表|单位\s*[:：]|编制单位|币种\s*[:：]"
)
# 「□适用 ☑不适用 / 报告期不存在…」是独立结论段，不能并进后面的表。
_STANDALONE_ANSWER = re.compile(
    r"[□☑■√]\s*适用|[□☑■√]\s*不适用|报告期不存在|未发生"
)
_TABLE_LEAD_HEADING = re.compile(
    r"^(?:#{1,6}\s*)?"
    r"(?:"
    r"\(?\d{1,2}\)?[、.．]"
    r"|[（(][\d一二三四五六七八九十]+[）)]"
    r"|[一二三四五六七八九十]+[、.]"
    r")"
)
# 同页过小相邻表合并：单表纯文字不超过此值才考虑合并，合并后总长仍受 max_chars 约束。
_SMALL_TABLE_PLAIN = 300
# 句子边界：中英文常见终止符后的位置（保留标点在句尾）
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")
# Markdown 表格的分隔行：|---|---|
_MD_SEPARATOR = re.compile(r"^\|[\s:\-|]+\|$")
# 整页 OCR 的表是一整段 HTML。按 500 字硬切会切断 </td>，所以整表抽出，
# 再按行项目切开。这个数是重排读正文的窗口，不是一行组能装多少字。
TABLE_GROUP_CHARS = 800
_HTML_TABLE = re.compile(r"<table\b[^>]*>[\s\S]*?</table>", re.IGNORECASE)
_HTML_ROW = re.compile(r"<tr\b[^>]*>[\s\S]*?</tr>", re.IGNORECASE)
_HTML_CELL = re.compile(r"<t[dh]\b([^>]*)>([\s\S]*?)</t[dh]>", re.IGNORECASE)
_HTML_COLSPAN = re.compile(r"""colspan=["'](\d+)["']""", re.IGNORECASE)
_HTML_ROWSPAN = re.compile(r"""rowspan=["'](\d+)["']""", re.IGNORECASE)
# 嵌套子行：紧跟上一行，不能单独切到下一片
_TABLE_SUB = re.compile(r"^(其中[:：]?|[（(]其中[）)]|[—–\-•·]\s*|[加减][:：])")
# 金融工具三个计量类别是同一组，即使没写「其中：」也不能拆
_TABLE_IFRS_SUB = re.compile(r"^以(公允价值|摊余成本)计量")
# 汇总行必须跟本片明细走，禁止单独成片
_TABLE_TOTAL = re.compile(r"(合计|小计|总计)$")
_NUM_ONLY_CELL = re.compile(r"^[\d.,%+\-]+$")
# 「利润总额 | 不适用」「研发占比 | 减少5.75个百分点」是数据，不是列名。
# 当成列名会从这里把表切开，起始行上的 rowspan 还留在上一片，
# 后面的行被当成新表头，最后一行还会因为「没有科目」被丢掉。
_NOT_HEADER_CELL = re.compile(
    r"^不适用$|^(?:增加|减少)[\d.,]+个百分点$"
)
# 第一列里不是行项目的字：表头、单位、纯数字、分组标题（「流动资产：」）
_INDICATOR_NUM = re.compile(r"^[\d.,，%％+\-－—()（）/\s]+$")
_INDICATOR_HEADER = re.compile(
    r"^(项目|科目|附注|注释|行次|序号|金额|本期金额|上期金额|期末余额|期初余额)$"
    r"|^(单位|币种)[:：]"
)
_INDICATOR_SECTION = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9]{1,12}[：:]$")
_INDICATOR_TOTAL = re.compile(r"^(合计|小计|总计)$")
_INDICATOR_LIMIT = 40


@dataclass
class Chunk:
    text: str
    page: int | None = None  # PDF 页码 / Excel sheet 序号；无页概念的文档为 None
    kind: str = TEXT  # text / table / figure
    page_end: int | None = None  # 同节内聚合导致的跨页切片：末页页码
    section: str | None = None  # 所属一级节，如「第十节 管理层讨论与分析」
    # 报表口径：consolidated / parent / None（非报表或无表题）
    scope: str | None = None
    # 入库后的切片序号。从库里读出来建图时必须带上：建图会先过滤掉低信息
    # 切片，过滤后的列表下标和库里的 seq 不再相等，用下标当 seq 写 mention
    # 会让"原文证据"整体错位（实测只有 5% 指对了片子）
    seq: int | None = None


def chunk_segments(
    segments: list[Segment],
    *,
    max_chars: int = 500,
    overlap: int = 50,
    within_page: bool = False,
    table_group_mode: str = "row",
) -> list[Chunk]:
    """按节成流、按段兜底：同节连续正文合并后聚合，表与无章节段独立处理。

    `within_page=True` 时同节内也按页断（一片只落一页，溯源最精确，但
    跨页的长论述会被切断）；默认 False，允许同节内跨页聚合。
    """
    chunks: list[Chunk] = []
    flow: list[Segment] = []
    plain: list[Segment] = []

    def flush_flow() -> None:
        if flow:
            chunks.extend(
                _chunk_flow(flow, max_chars=max_chars, overlap=overlap)
            )
            flow.clear()

    def flush_plain() -> None:
        if plain:
            chunks.extend(
                _chunk_plain_pages(plain, max_chars=max_chars, overlap=overlap)
            )
            plain.clear()

    for seg in _carry_html_page_headers(
        _coalesce_segments(_drop_noise(segments), max_chars=max_chars)
    ):
        if seg.kind == TABLE:
            # 表必须整表成片（超长时重复表头），不能揉进正文流
            flush_plain()
            flush_flow()
            chunks.extend(
                Chunk(
                    text=piece,
                    page=seg.page,
                    kind=seg.kind,
                    page_end=seg.page_end if seg.page_end is not None else seg.page,
                    section=seg.section,
                    scope=seg.scope,
                )
                for piece in _chunk_table(
                    seg.text, max_chars=max_chars, group_mode=table_group_mode
                )
            )
        elif seg.section is None:
            # 无章节信息：页与页的正文不合并。紧挨着的 HTML 表在
            # _chunk_plain_pages 里按列数接成一张，并记上 page_end。
            flush_flow()
            plain.append(seg)
        else:
            # 两条栅栏：换节必断；开了单页栅栏时换页也断。
            # 断点处不保留重叠——重叠只发生在同一片流内部（超长段落二次
            # 切分时），所以重叠永远不会跨节、跨页（when within_page）。
            flush_plain()
            if flow and (
                flow[0].section != seg.section
                or (within_page and flow[-1].page != seg.page)
            ):
                flush_flow()
            flow.append(seg)
    flush_plain()
    flush_flow()
    return _expand_long_html_tables(
        chunks, max_chars=max_chars, group_mode=table_group_mode
    )


def _drop_noise(segments: list[Segment]) -> list[Segment]:
    """剔除解析层漏下来的空信息片段（视觉模型的「无图表」、目录残留）。

    放在切片入口而不是解析层：解析层要负责「这一页有没有内容」的判断，
    切片层负责「这段值不值得进库」，后者是全文统一的最后一道闸。
    只丢整段都是噪声的，段落级混排（真内容 + 一句无图表）照旧保留。
    """
    kept: list[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if seg.kind == FIGURE and len(text) <= _NO_CHART_MAX and _NO_CHART.match(text):
            continue
        if len(text) <= 6 and _TOC_RESIDUE.match(text):
            continue
        kept.append(seg)
    return kept


def _plain_len(text: str) -> int:
    """去掉 HTML 标签后的可见字数，跟向量化正文同一量级。"""
    return len(re.sub(r"<[^>]+>", "", text or ""))


def _same_page(a: Segment, b: Segment) -> bool:
    return a.page is not None and a.page == b.page


def _is_table_lead(text: str) -> bool:
    """短引导句：应贴到下一张表，而不是单独成正文片。"""
    t = (text or "").strip()
    if not t or len(t) > _TABLE_LEAD_MAX:
        return False
    if _STANDALONE_ANSWER.search(t) and not _TABLE_LEAD_HINT.search(t):
        return False
    if _TABLE_LEAD_HINT.search(t):
        return True
    if _TABLE_LEAD_HEADING.match(t):
        return True
    compact = re.sub(r"\s+", "", t)
    if "。" in t:
        return False
    if compact.endswith(("：", ":")):
        return True
    # 「其他说明」这类极短词
    return len(compact) <= 16 and "\n" not in t


def _can_merge_small_tables(
    prev: Segment, nxt: Segment, *, max_chars: int
) -> bool:
    """同页、同口径的过小相邻表：合成一段，减少半年度报告里成片的碎表。"""
    if not _same_page(prev, nxt):
        return False
    if prev.section and nxt.section and prev.section != nxt.section:
        return False
    if prev.scope and nxt.scope and prev.scope != nxt.scope:
        return False
    if _plain_len(prev.text) > _SMALL_TABLE_PLAIN:
        return False
    if _plain_len(nxt.text) > _SMALL_TABLE_PLAIN:
        return False
    # 合并后仍应能整片进库；超长会走 Markdown 切分，HTML 表会被拆坏
    return len(prev.text) + len(nxt.text) + 1 <= max_chars


def _merge_table_segments(prev: Segment, nxt: Segment) -> Segment:
    return replace(
        prev,
        text=f"{prev.text.rstrip()}\n{nxt.text.lstrip()}",
        page_end=nxt.page_end or nxt.page or prev.page_end or prev.page,
        section=prev.section or nxt.section,
        scope=prev.scope or nxt.scope,
    )


def _coalesce_segments(
    segments: list[Segment], *, max_chars: int
) -> list[Segment]:
    """切片前合并：短引导句并进下一张表；同页过小相邻表拼成一段。

    半年度 OCR 产物里大量「续上表：」独立正文 + 一两行小表，各自成片会把
    1MB 文件撑到五百多片。这里只做段级合并，不改后续按 max_chars 切大表。
    """
    out: list[Segment] = []
    for seg in segments:
        if (
            seg.kind == TABLE
            and out
            and out[-1].kind == TEXT
            and _same_page(out[-1], seg)
            and _is_table_lead(out[-1].text)
        ):
            lead = out.pop()
            seg = replace(
                seg,
                text=f"{lead.text.strip()}\n{seg.text.strip()}",
                section=seg.section or lead.section,
            )
        if (
            seg.kind == TABLE
            and out
            and out[-1].kind == TABLE
            and _can_merge_small_tables(out[-1], seg, max_chars=max_chars)
        ):
            seg = _merge_table_segments(out.pop(), seg)
        out.append(seg)
    return out


def _chunk_flow(
    segments: list[Segment], *, max_chars: int, overlap: int
) -> list[Chunk]:
    """把同一节内的连续页聚合成切片流，页码取区间。"""
    section = segments[0].section
    kind = segments[0].kind
    out: list[Chunk] = []
    buf: list[str] = []
    pages: list[int] = []
    length = 0

    def flush() -> None:
        nonlocal buf, pages, length
        if buf:
            text = "\n".join(buf).strip()
            if text:
                out.append(
                    Chunk(
                        text=text,
                        page=min(pages) if pages else None,
                        kind=kind,
                        page_end=max(pages) if pages else None,
                        section=section,
                    )
                )
            buf, pages, length = [], [], 0

    for seg in segments:
        for para in _PARA_SPLIT.split(_normalize(seg.text)):
            para = para.strip()
            if not para:
                continue
            for piece_kind, piece in _html_pieces(para):
                if piece_kind == "table":
                    flush()
                    out.append(
                        Chunk(
                            text=piece,
                            page=seg.page,
                            kind=TABLE,
                            page_end=seg.page,
                            section=section,
                        )
                    )
                    continue
                if len(piece) > max_chars:
                    flush()
                    out.extend(
                        Chunk(
                            text=part,
                            page=seg.page,
                            kind=kind,
                            page_end=seg.page,
                            section=section,
                        )
                        for part in _split_long(piece, max_chars, overlap)
                    )
                    continue
                if length + len(piece) > max_chars and buf:
                    flush()
                buf.append(piece)
                if seg.page is not None:
                    pages.append(seg.page)
                # 拼接用的换行符也占长度：漏算会让切片实际超出上限
                # （原实现中位 511 字、13% 超过 550，就是这个原因）
                length += len(piece) + 1
    flush()
    return out


def chunk_text(text: str, *, max_chars: int = 500, overlap: int = 50) -> list[str]:
    text = _normalize(text)
    if not text:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        for piece_kind, piece in _html_pieces(para):
            if piece_kind == "table":
                if current:
                    chunks.append("\n".join(current))
                    current, length = [], 0
                chunks.append(piece)
                continue
            if len(piece) > max_chars:
                if current:
                    chunks.append("\n".join(current))
                    current, length = [], 0
                chunks.extend(_split_long(piece, max_chars, overlap))
                continue
            if length + len(piece) > max_chars and current:
                chunks.append("\n".join(current))
                current, length = [], 0
            current.append(piece)
            # 换行符也算长度（与 _chunk_flow 一致），否则切片会悄悄超标
            length += len(piece) + 1
    if current:
        chunks.append("\n".join(current))
    return [c.strip() for c in chunks if c.strip()]


def _normalize(text: str) -> str:
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _html_chunk_kind(seg_kind: str, text: str) -> str:
    """整段就是一张 HTML 表时，切片类型记成 table。"""
    if _HTML_TABLE.fullmatch(text.strip()):
        return TABLE
    return seg_kind


def _chunk_plain_pages(
    segments: list[Segment], *, max_chars: int, overlap: int
) -> list[Chunk]:
    """无章节的页各自切片。上一页以表结尾、下一页以表开头且列数相同，则接成一张。"""
    out: list[Chunk] = []
    for seg in segments:
        pieces = [
            Chunk(
                text=piece,
                page=seg.page,
                kind=_html_chunk_kind(seg.kind, piece),
                page_end=seg.page,
            )
            for piece in chunk_text(seg.text, max_chars=max_chars, overlap=overlap)
        ]
        if out and pieces and _can_merge_html_tables(out[-1], pieces[0]):
            out[-1] = Chunk(
                text=_merge_html_tables(out[-1].text, pieces[0].text),
                page=out[-1].page,
                kind=TABLE,
                page_end=pieces[0].page,
            )
            out.extend(pieces[1:])
        else:
            out.extend(pieces)
    return out


def _table_rows(table: str) -> list[str]:
    return _HTML_ROW.findall(table)


def _row_width(row: str) -> int:
    width = 0
    for match in _HTML_CELL.finditer(row):
        span = _HTML_COLSPAN.search(match.group(1) or "")
        width += int(span.group(1)) if span else 1
    return width


def _row_texts(row: str) -> list[str]:
    return [
        re.sub(r"\s+", "", match.group(2))
        for match in _HTML_CELL.finditer(row)
    ]


def _html_row_is_data(row: str) -> bool:
    """至少一个格子是纯数字。列名行里的「2026年1-9月」不算。"""
    return any(cell and _NUM_ONLY_CELL.match(cell) for cell in _row_texts(row))


def _html_row_is_item(row: str) -> bool:
    """科目行：第一列是名称，其余列是空的或数字。

    资产负债表跨页时，页首常常是「其他债权投资」这种金额整格为空的行，
    它没有数字，但仍然是上一张表的续表，不是新的列名。
    """
    cells = _row_texts(row)
    if not cells or not cells[0] or _html_row_is_header(row):
        return False
    return all(not cell or _NUM_ONLY_CELL.match(cell) for cell in cells[1:])


def _html_row_is_header(row: str) -> bool:
    if _html_row_is_clipped_header(row):
        return False
    cells = [cell for cell in _row_texts(row) if cell]
    if len(cells) < 2 or _html_row_is_data(row):
        return False
    if any(_NOT_HEADER_CELL.match(cell) for cell in cells):
        return False
    return any(re.search(r"[\u4e00-\u9fff]", cell) for cell in cells)


def _html_row_is_clipped_header(row: str) -> bool:
    """跨页被页眉切掉的表头：第一列以及紧挨着的格子都空，剩下「(%)」「份数量」。

    新的一行第一列不会空。只空左上角、第二格已经是「期末余额」的，是正常表头。
    """
    texts = _row_texts(row)
    if len(texts) < 2 or texts[0] or texts[1] or _html_row_is_data(row):
        return False
    return any(texts)


def _rowspan_continuations(rows: list[str]) -> set[int]:
    """被上面某一行的 rowspan 盖住的行号。这些行必须和起始行留在同一片。"""
    covered: set[int] = set()
    for index, row in enumerate(rows):
        for match in _HTML_CELL.finditer(row):
            span = _span_attr(_HTML_ROWSPAN, match.group(1) or "")
            for down in range(1, span):
                covered.add(index + down)
    return covered


def _clamp_rowspans(rows: list[str]) -> list[str]:
    """rowspan 不能超出这一片实际剩下的行，否则展示会补出空行。"""
    total = len(rows)
    clamped: list[str] = []
    for index, row in enumerate(rows):
        room = total - index

        def repl(match: re.Match[str], room: int = room) -> str:
            attrs = match.group(1) or ""
            span = _span_attr(_HTML_ROWSPAN, attrs)
            if span <= room:
                return match.group(0)
            if room <= 1:
                attrs = _HTML_ROWSPAN.sub("", attrs)
            else:
                attrs = _HTML_ROWSPAN.sub(f' rowspan="{room}"', attrs)
            open_tag = re.match(r"<\s*(t[dh])", match.group(0), re.IGNORECASE)
            name = open_tag.group(1) if open_tag else "td"
            return f"<{name}{attrs}>{match.group(2)}</{name}>"

        clamped.append(_HTML_CELL.sub(repl, row))
    return clamped


def _html_tables_from_groups(groups: list[list[str]]) -> list[str]:
    return [
        "<table>" + "".join(_clamp_rowspans(group)) + "</table>" for group in groups
    ]


def _html_header_to_carry(rows: list[str]) -> list[str] | None:
    """续表要继承的列名。

    页中途换过列名（「合并资产负债表项目」下面又印「合并利润表项目」）时，
    带过去的是最后一组列名，不是表头第一行。开头连续的列名行一起带上。
    """
    last: int | None = None
    first_data: int | None = None
    for index, row in enumerate(rows):
        if _html_row_is_header(row):
            last = index
        elif first_data is None and _html_row_is_data(row):
            first_data = index
    if last is None:
        return None
    if first_data is None or last < first_data:
        end = 0
        while end < len(rows) and _html_row_is_header(rows[end]):
            end += 1
        return rows[:end] or None
    return [rows[last]]


def _carry_html_page_headers(segments: list[Segment]) -> list[Segment]:
    """跨页续表补列名和表题，并顺着后面每一页传下去。

    财报排到页底时，后面几页都不再印「项目 | 本期 | 上期」，第一行就是科目，
    「合并年初到报告期末利润表」也只出现在第一页。补过的那一页再被切断时，
    同一组列名和表题继续往下页带。只在四件事同时成立时补：上一页以这张表结束、
    这一页的表排在最前、列数相同、首行是科目。金额为空的科目行也算。
    中途印了新列名或新表题就改带新的。表后面还有正文，这一路停下。
    """
    pending: list[str] | None = None
    pending_page: int | None = None
    pending_width = 0
    pending_caption = ""
    seen_pages: set[int] = set()
    out: list[Segment] = []
    for seg in segments:
        page = seg.page
        if seg.kind != TABLE:
            if page is not None and page == pending_page:
                pending = None
                pending_caption = ""
            if page is not None:
                seen_pages.add(page)
            out.append(seg)
            continue
        match = _HTML_TABLE.search(seg.text or "")
        rows = _table_rows(match.group(0)) if match else []
        # 表后面的正文要在补表头之前量。补完再量，旧的结束位置会落在新表头上。
        tail = seg.text[match.end() :].strip() if match else ""
        starts_page = page is not None and page not in seen_pages
        carried = False
        # 页顶第一行第一列空着，是被页眉切掉的表头，不是新的一行。先丢掉，
        # 再用上一页的列名接上后面的股东行。
        body = list(rows)
        while body and _html_row_is_clipped_header(body[0]):
            body = body[1:]
        # 去掉残头之后，股东行第二列是「境内非国有法人」，不是纯数字，
        # 套不上科目行，但第一列已经有名字，就是上一张表的续行。
        continues = bool(body) and (
            _html_row_is_item(body[0])
            or (
                len(body) < len(rows)
                and _html_row_is_data(body[0])
                and bool(_row_texts(body[0])[0])
            )
        )
        if (
            match
            and continues
            and pending
            and pending_page is not None
            and page == pending_page + 1
            and starts_page
            and _row_width(body[0]) == pending_width
        ):
            prefix = seg.text[: match.start()].strip()
            if not prefix and pending_caption:
                prefix = pending_caption
            html_table = "<table>" + "".join(pending) + "".join(body) + "</table>"
            suffix = seg.text[match.end() :]
            text = f"{prefix}\n{html_table}" if prefix else html_table
            seg = replace(seg, text=text + suffix)
            rows = [*pending, *body]
            carried = True
        out.append(seg)
        if page is not None:
            seen_pages.add(page)
        # 表后面还有正文，说明这张表在页中就结束了，不是被分页切断的
        header = _html_header_to_carry(rows) if rows and not tail else None
        if header and page is not None:
            pending = header
            pending_page = page
            pending_width = _row_width(header[0])
            current = _HTML_TABLE.search(seg.text or "")
            own_caption = seg.text[: current.start()].strip() if current else ""
            if own_caption or not carried:
                pending_caption = own_caption
        else:
            pending = None
            pending_caption = ""
    return out


def _can_merge_html_tables(prev: Chunk, nxt: Chunk) -> bool:
    if prev.kind != TABLE or nxt.kind != TABLE:
        return False
    if prev.page_end is None or nxt.page is None or nxt.page != prev.page_end + 1:
        return False
    rows_a = _table_rows(prev.text)
    rows_b = _table_rows(nxt.text)
    if not rows_a or not rows_b:
        return False
    width = _row_width(rows_a[0])
    return width > 0 and width == _row_width(rows_b[0])


def _merge_html_tables(first: str, second: str) -> str:
    """接上续页的数据行。续页若把表头又印了一遍，这几行丢掉。"""
    rows_a = _table_rows(first)
    rows_b = _table_rows(second)
    copied = 0
    limit = min(3, len(rows_a), len(rows_b))
    while copied < limit:
        left = _row_texts(rows_a[copied])
        if not any(left) or left != _row_texts(rows_b[copied]):
            break
        copied += 1
    return "<table>" + "".join(rows_a + rows_b[copied:]) + "</table>"


def _html_pieces(para: str) -> list[tuple[str, str]]:
    """把一段里的 HTML 表抽出来。表不参与后面的 500 字聚合。"""
    if not _HTML_TABLE.search(para):
        return [("text", para)]
    pieces: list[tuple[str, str]] = []
    last = 0
    for match in _HTML_TABLE.finditer(para):
        before = para[last : match.start()].strip()
        if before:
            pieces.append(("text", before))
        pieces.append(("table", match.group(0)))
        last = match.end()
    after = para[last:].strip()
    if after:
        pieces.append(("text", after))
    return pieces or [("text", para)]


def _cell_plain(raw: str) -> str:
    text = re.sub(r"<[^>]+>", "", raw or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _row_plain_line(row: str) -> str:
    return " | ".join(_cell_plain(match.group(2)) for match in _HTML_CELL.finditer(row))


def extract_table_indicators(text: str) -> list[str]:
    """表格第一列的行项目名。单位、纯数字、表头和分组标题丢掉，不存数值。"""
    cells = _table_label_cells(text or "")
    names: list[str] = []
    seen: set[str] = set()
    for cell in cells:
        name = _clean_indicator(cell)
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= _INDICATOR_LIMIT:
            break
    return names


def _table_label_cells(text: str) -> list[str]:
    tables = list(_HTML_TABLE.finditer(text))
    if tables:
        cells: list[str] = []
        for match in tables:
            rows = _table_rows(match.group(0))
            head_n = _html_header_row_count(rows)
            for row in rows[head_n:]:
                if _is_html_section_row(row):
                    continue
                found = next(_HTML_CELL.finditer(row), None)
                if found:
                    cells.append(_cell_plain(found.group(2)))
        return cells
    lines = [line for line in text.splitlines() if line.strip()]
    while lines and "|" not in lines[0]:
        lines.pop(0)
    if not lines:
        return []
    body = [
        line
        for line in lines[1:]
        if not _MD_SEPARATOR.match(line.strip())
    ]
    return [_md_first_cell(line) for line in body]


def _clean_indicator(cell: str) -> str:
    name = (cell or "").strip().strip("*").strip()
    if not (2 <= len(name) <= 40):
        return ""
    if (
        _INDICATOR_NUM.match(name)
        or _INDICATOR_HEADER.match(name)
        or _INDICATOR_SECTION.match(name)
        or _INDICATOR_TOTAL.match(name)
    ):
        return ""
    return name


def html_table_plain(text: str) -> str:
    """表格换成「单元格 | 单元格」的行。标签和 Markdown 分隔行不进向量，也不进重排。"""
    if not text:
        return text
    if _HTML_TABLE.search(text):

        def repl(match: re.Match[str]) -> str:
            lines = [_row_plain_line(row) for row in _table_rows(match.group(0))]
            return "\n".join(line for line in lines if line.strip())

        text = _HTML_TABLE.sub(repl, text).strip()
    return _markdown_table_plain(text)


def _markdown_table_plain(text: str) -> str:
    """人工修订存成 Markdown 表时，分隔行同样不进向量。"""
    if "|" not in text:
        return text
    lines = text.splitlines()
    out: list[str] = []
    changed = False
    index = 0
    while index < len(lines):
        nxt = lines[index + 1] if index + 1 < len(lines) else ""
        if _md_row(lines[index]) and _MD_SEPARATOR.match(nxt.strip()):
            changed = True
            while index < len(lines) and _md_row(lines[index]):
                if not _MD_SEPARATOR.match(lines[index].strip()):
                    joined = " | ".join(_md_cells(lines[index]))
                    if joined.strip():
                        out.append(joined)
                index += 1
            continue
        out.append(lines[index])
        index += 1
    return "\n".join(out).strip() if changed else text


def _md_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _md_cells(line: str) -> list[str]:
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    return [part.strip().replace("\\|", "|") for part in re.split(r"(?<!\\)\|", raw)]


def html_tables_to_markdown(text: str) -> str:
    """HTML 表换成 Markdown 表。只用于喂给模型的正文，库存和引用仍是 HTML。

    colspan / rowspan 展开成矩形，合并格的文字写进它盖住的每一列，
    「期末余额」横跨「合并 | 母公司」时两列都还带着这个列名。
    整行只有一个分组标题（「流动资产：」）时不横着复制，避免把它当成每列的值。
    """
    if not text or not _HTML_TABLE.search(text):
        return text
    return _HTML_TABLE.sub(lambda match: _one_html_table_to_markdown(match.group(0)), text)


def _one_html_table_to_markdown(table: str) -> str:
    grid = _expand_html_grid(_table_rows(table))
    if not grid or not any(grid):
        plain = html_table_plain(table)
        return plain or table
    width = max(len(row) for row in grid)
    lines = [
        "| " + " | ".join(_md_cell(cell) for cell in row + [""] * (width - len(row))) + " |"
        for row in grid
    ]
    separator = "| " + " | ".join("---" for _ in range(width)) + " |"
    return "\n".join([lines[0], separator, *lines[1:]])


def _md_cell(text: str) -> str:
    return (text or "").replace("|", "\\|")


def _expand_html_grid(rows: list[str]) -> list[list[str]]:
    """按 colspan / rowspan 把表铺成矩形。被上一行占住的格子跳过。"""
    grid: list[list[str | None]] = []
    for row_index, row_html in enumerate(rows):
        while len(grid) <= row_index:
            grid.append([])
        column = 0
        # 「流动资产：」这种分组行横跨整表。复制到每一列，模型会把它当成该列的数值。
        section_row = _is_html_section_row(row_html)
        for match in _HTML_CELL.finditer(row_html):
            attrs = match.group(1) or ""
            colspan = _span_attr(_HTML_COLSPAN, attrs)
            rowspan = _span_attr(_HTML_ROWSPAN, attrs)
            while column < len(grid[row_index]) and grid[row_index][column] is not None:
                column += 1
            text = _cell_plain(match.group(2))
            section = section_row and colspan > 1
            for down in range(rowspan):
                # 切片把合并格切断后，rowspan 可能比这一片剩下的行多。
                # 只铺已经存在的行，不能为了凑 rowspan 补出空行。
                if row_index + down >= len(rows):
                    break
                while len(grid) <= row_index + down:
                    grid.append([])
                target = grid[row_index + down]
                while len(target) < column + colspan:
                    target.append(None)
                for offset in range(colspan):
                    if target[column + offset] is None:
                        target[column + offset] = "" if section and offset else text
            column += colspan
    width = max((len(row) for row in grid), default=0)
    return [[cell or "" for cell in row] + [""] * (width - len(row)) for row in grid]


def _span_attr(pattern: re.Pattern[str], attrs: str) -> int:
    found = pattern.search(attrs)
    if not found:
        return 1
    try:
        return max(int(found.group(1)), 1)
    except ValueError:
        return 1


def _html_header_row_count(rows: list[str]) -> int:
    """表头占几行：第一行的 rowspan 盖住的都算表头，每片都要重复。"""
    if not rows:
        return 0
    span = 1
    for match in _HTML_CELL.finditer(rows[0]):
        found = _HTML_ROWSPAN.search(match.group(1) or "")
        if found:
            span = max(span, int(found.group(1)))
    return min(max(span, 1), len(rows))


def _is_html_section_row(row: str) -> bool:
    """「流动资产：」这种分组标题，跟在它后面的科目走，不留在上一组末尾。"""
    cells = [_cell_plain(match.group(2)) for match in _HTML_CELL.finditer(row)]
    filled = [cell for cell in cells if cell]
    return len(filled) == 1 and filled[0].endswith(("：", ":"))


def _buf_has_account(buf: list[str], head_n: int) -> bool:
    """表头和分组标题（「流动资产：」）之外，已经有一行科目。"""
    return any(not _is_html_section_row(row) for row in buf[head_n:])


def _split_html_table(
    table: str, *, max_chars: int = 500, group_mode: str = "row"
) -> list[str]:
    """HTML 表按行项目切开。每片重复表头和当前分组标题。

    「其中：」、加减行、合计跟所属科目留在同一片。单行再长也不从格子中间切。
    被 rowspan 盖住的续行跟起始行留在同一片，避免合并格的文字留在上一片、
    下面的项目名被当成新表头。

    group_mode="row"（默认）：每个行项目单独成片，自解释最强但切片数多。
    group_mode="section"：语义感知滑动窗口（见 _split_html_table_grouped），
    按 max_chars 累积、在科目行边界断、分组标题强制起新片，切片数大幅减少。
    """
    rows = _table_rows(table)
    if not rows:
        return [table]
    clipped = 0
    while clipped < len(rows) and _html_row_is_clipped_header(rows[clipped]):
        clipped += 1
    if clipped:
        rows = rows[clipped:]
        if not rows:
            return [table]
    head_n = _html_header_row_count(rows)
    # 残头后面直接是股东行时，不能把第一位股东再当成列名复制到每一片。
    if (
        clipped
        and _html_row_is_data(rows[0])
        and not _html_row_is_header(rows[0])
    ):
        head_n = 0
    header, data = rows[:head_n], rows[head_n:]
    if len(data) <= 1:
        return _html_tables_from_groups([rows]) if rows else [table]
    covered = _rowspan_continuations(rows)
    if group_mode == "section":
        return _split_html_table_grouped(header, data, max_chars, covered, head_n)
    groups: list[list[str]] = []
    section: list[str] = []
    buf = list(header)

    def restart(extra: list[str]) -> None:
        nonlocal buf
        if _buf_has_account(buf, len(header)):
            groups.append(buf)
        buf = [*header, *extra]

    for offset, row in enumerate(data):
        if head_n + offset in covered:
            buf.append(row)
            continue
        first = _row_plain_line(row).split(" | ", 1)[0].strip()
        # 同一张表中途换列名（资产负债表下面又印利润表）时，后面的科目跟新列名。
        if _html_row_is_header(row):
            if _buf_has_account(buf, len(header)):
                groups.append(buf)
                header = [row]
            elif buf == header:
                header = [*header, row]
            else:
                header = [row]
            section = []
            buf = list(header)
            continue
        if _is_html_section_row(row):
            section = [row]
            restart(section)
            continue
        sticky = _is_sub_row(first) or _is_total_row(first)
        if not sticky and _buf_has_account(buf, len(header)):
            restart(section)
        buf.append(row)
    if _buf_has_account(buf, len(header)):
        groups.append(buf)
    if not groups:
        return [table]
    return _html_tables_from_groups(groups)


def _rows_len(rows: list[str]) -> int:
    """一组表格行的字符长度（用向量化的单元格文字计，与 max_chars 同一量级）。"""
    return sum(len(_row_plain_line(row)) + 1 for row in rows)


def _split_html_table_grouped(
    header: list[str],
    data: list[str],
    max_chars: int,
    covered: set[int] | None = None,
    head_n: int = 0,
) -> list[str]:
    """section 模式：语义感知滑动窗口，替代固定行数切分。

    按 max_chars 累积行，到窗口临界时在下一个科目行边界（非「其中：」/合计、
    非分组标题）断开——绝不切断一个科目及其子项。分组标题（「流动资产：」）
    强制起新片，让同组科目聚在一片；「其中：」子项、合计/小计跟所属科目走。
    每片重复表头 + 当前分组标题，离开表题/单位也能自解释。无分组标题的扁平
    表同样适用：只是在科目边界按窗口断，不会无限聚成一片。
    """
    groups: list[list[str]] = []
    section: list[str] = []
    buf = list(header)
    cur_len = _rows_len(header)

    def flush() -> None:
        nonlocal buf, cur_len
        if _buf_has_account(buf, len(header)):
            groups.append(buf)
        buf = [*header, *section]
        cur_len = _rows_len(buf)

    for offset, row in enumerate(data):
        if head_n + offset in (covered or ()):
            buf.append(row)
            cur_len += len(_row_plain_line(row)) + 1
            continue
        first = _row_plain_line(row).split(" | ", 1)[0].strip()
        if _html_row_is_header(row):
            # 中途换列名：保留原分支语义
            if _buf_has_account(buf, len(header)):
                groups.append(buf)
                header = [row]
            elif buf == header:
                header = [*header, row]
            else:
                header = [row]
            section = []
            buf = list(header)
            cur_len = _rows_len(buf)
            continue
        if _is_html_section_row(row):
            # 分组标题强制起新片，同组科目聚在一片
            flush()
            section = [row]
            buf = [*header, *section]
            cur_len = _rows_len(buf)
            continue
        sticky = _is_sub_row(first) or _is_total_row(first)
        row_len = len(_row_plain_line(row)) + 1
        if (
            not sticky
            and _buf_has_account(buf, len(header))
            and cur_len + row_len > max_chars
        ):
            flush()
        buf.append(row)
        cur_len += row_len
    if _buf_has_account(buf, len(header)):
        groups.append(buf)
    if not groups:
        return _html_tables_from_groups([header + data])
    return _html_tables_from_groups(groups)


def _expand_long_html_tables(
    chunks: list[Chunk], *, max_chars: int = 500, group_mode: str = "row"
) -> list[Chunk]:
    """跨页接完之后再拆。先拆再接会把续页表头当成另一张表。

    必须是「整段恰好一张 ``<table>``」才拆。``fullmatch`` 配非贪婪
    ``</table>`` 会把「表A + 表B」吞成一张假表再拆烂——同页小表合并后
    会踩这个坑。
    """
    out: list[Chunk] = []
    for chunk in chunks:
        text = chunk.text.strip()
        matches = list(_HTML_TABLE.finditer(text))
        if (
            chunk.kind != TABLE
            or len(matches) != 1
            or matches[0].group(0) != text
        ):
            out.append(chunk)
            continue
        parts = _split_html_table(
            text, max_chars=max_chars, group_mode=group_mode
        )
        if len(parts) == 1:
            out.append(chunk)
            continue
        out.extend(replace(chunk, text=part, kind=TABLE) for part in parts)
    return out


def _chunk_html_table(
    text: str, *, max_chars: int = 500, group_mode: str = "row"
) -> list[str] | None:
    """表题加一张 HTML 表时按行切开，每片重复表题。不是这种形态就返回 None。"""
    matches = list(_HTML_TABLE.finditer(text))
    if len(matches) != 1 or text[matches[0].end() :].strip():
        return None
    match = matches[0]
    caption = text[: match.start()].strip()
    parts = _split_html_table(
        match.group(0), max_chars=max_chars, group_mode=group_mode
    )
    if not caption:
        return parts
    return [f"{caption}\n{part}" for part in parts]


def _chunk_table(text: str, *, max_chars: int, group_mode: str = "row") -> list[str]:
    """表格切片：短表整片保留；多行表按行项目切，每片重复表头和表题。

    表格正文开头可能有若干不含竖线的行：表题（「母公司利润表」）和
    单位 / 审计状态。这些必须在每片里重复——否则后半张表只剩
    「项目 | 本期 | 上期 + 一堆数字」，看不出这张表属于谁、单位是什么。

    「其中：」子项、金融工具三个计量类别、合计 / 总计都跟所属科目走。

    ``group_mode``：``row`` 每个科目一片；``section`` 按 max_chars 在科目
    边界累积（与 HTML 表同一策略）。文字层产出的是 Markdown 表，以前漏接
    了 section，年报会切出上千片几乎等长的单行表。
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    # HTML 表没有竖线。不走下面的 Markdown 切法，否则表题会被单独算成一张表。
    # 多行科目即使整表不到 max_chars 也切开，避免一张利润表合成一个向量。
    html_parts = _chunk_html_table(text, max_chars=max_chars, group_mode=group_mode)
    if html_parts is not None:
        return html_parts
    # row 模式短表整片留；section 模式要进打包器，才能在「流动资产：」这类
    # 分组标题处强制起新片（与 HTML section 行为一致）。
    if group_mode != "section" and len(text) <= max_chars:
        return [text]

    caption: list[str] = []
    while lines and "|" not in lines[0]:
        caption.append(lines.pop(0))

    sep_index = next(
        (i for i, line in enumerate(lines[:5]) if _MD_SEPARATOR.match(line.strip())),
        None,
    )
    if sep_index is not None:
        header, data = _demote_fake_md_header(
            caption, lines[:sep_index], lines[sep_index], lines[sep_index + 1 :]
        )
    else:
        header, data = [*caption, *lines[:1]], lines[1:]
    if not data:
        # 不是表格形态（单行超长），退回普通切分
        return chunk_text(text, max_chars=max_chars, overlap=0)

    units = _table_row_units(data)
    if group_mode == "section":
        return _pack_table_units_section(header, units, max_chars=max_chars) or [text]
    return _pack_table_units(header, units) or [text]


def _md_row_cells(line: str) -> list[str]:
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    return [part.strip() for part in re.split(r"(?<!\\)\|", raw)]


def _demote_fake_md_header(
    caption: list[str],
    header_rows: list[str],
    separator: str,
    data: list[str],
) -> tuple[list[str], list[str]]:
    """科目行误当 Markdown 表头时降回数据行，换成中性列名再切。"""
    from ai_agent.kb.ingest.table_cells import _is_data_row, _is_item_row

    if not header_rows:
        return [*caption, separator], data
    cells = _md_row_cells(header_rows[-1])
    if not (_is_item_row(cells) or _is_data_row(cells)):
        return [*caption, *header_rows, separator], data
    width = max(1, len(cells))
    synth = "| " + " | ".join(["项目"] + [f"列{i}" for i in range(2, width + 1)]) + " |"
    return [*caption, synth, separator], [*header_rows, *data]


def _md_first_cell(line: str) -> str:
    """Markdown 表行的第一列（科目名）。"""
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    parts = re.split(r"(?<!\\)\|", raw)
    return parts[0].strip() if parts else ""


def _is_sub_row(cell: str) -> bool:
    return bool(_TABLE_SUB.match(cell) or _TABLE_IFRS_SUB.match(cell))


def _is_total_row(cell: str) -> bool:
    return bool(_TABLE_TOTAL.search(cell))


def _is_section_label(line: str) -> bool:
    """「流动资产：」这种只有科目、没有数字的分组标题。"""
    cell = _md_first_cell(line)
    if not cell.endswith(("：", ":")):
        return False
    return not any(
        _NUM_ONLY_CELL.match(part.strip())
        for part in re.split(r"(?<!\\)\|", line.strip().strip("|"))
        if part.strip()
    )


def _table_row_units(data: list[str]) -> list[list[str]]:
    """把数据行收成不可拆的行组：父行 + 其中/子项，合计贴到上一组。"""
    units: list[list[str]] = []
    for line in data:
        cell = _md_first_cell(line)
        if units and (_is_sub_row(cell) or _is_total_row(cell)):
            units[-1].append(line)
            continue
        units.append([line])
    return units


def _pack_table_units(header: list[str], units: list[list[str]]) -> list[str]:
    """每个行项目一片。表头和当前分组标题每片都带，合计绝不单独开片。"""
    out: list[str] = []
    buf: list[str] = list(header)
    section: list[str] = []

    def has_account() -> bool:
        return any(line not in header and not _is_section_label(line) for line in buf)

    def emit() -> None:
        if has_account():
            out.append("\n".join(buf))

    def restart() -> None:
        nonlocal buf
        buf = [*header, *section]

    for unit in units:
        if len(unit) == 1 and _is_section_label(unit[0]):
            if has_account():
                emit()
            section = list(unit)
            restart()
            continue
        if has_account():
            emit()
            restart()
        buf.extend(unit)
    emit()
    return _merge_orphan_total_chunks(header, out)


def _lines_len(lines: list[str]) -> int:
    return sum(len(line) + 1 for line in lines)


def _pack_table_units_section(
    header: list[str], units: list[list[str]], *, max_chars: int
) -> list[str]:
    """Markdown 表的 section 模式：按 max_chars 在科目边界累积。

    与 ``_split_html_table_grouped`` 对齐——分组标题（「流动资产：」）强制
    起新片；「其中：」/合计已收进 unit，不会被窗口切断。
    """
    out: list[str] = []
    section: list[str] = []
    buf: list[str] = list(header)
    cur_len = _lines_len(buf)

    def has_account() -> bool:
        return any(line not in header and not _is_section_label(line) for line in buf)

    def emit() -> None:
        if has_account():
            out.append("\n".join(buf))

    def restart() -> None:
        nonlocal buf, cur_len
        buf = [*header, *section]
        cur_len = _lines_len(buf)

    for unit in units:
        if len(unit) == 1 and _is_section_label(unit[0]):
            if has_account():
                emit()
            section = list(unit)
            restart()
            continue
        unit_len = _lines_len(unit)
        if has_account() and cur_len + unit_len > max_chars:
            emit()
            restart()
        buf.extend(unit)
        cur_len += unit_len
    emit()
    return _merge_orphan_total_chunks(header, out)


def _merge_orphan_total_chunks(header: list[str], chunks: list[str]) -> list[str]:
    """最后一片如果只剩合计/总计，并回上一片。"""
    if len(chunks) < 2:
        return chunks
    last_lines = chunks[-1].splitlines()
    body = last_lines[len(header) :]
    if body and all(_is_total_row(_md_first_cell(line)) for line in body):
        prev = chunks[-2].splitlines()
        return [*chunks[:-2], "\n".join(prev + body)]
    return chunks


def context_prefix(
    filename: str, *, page: int | None, kind: str, section: str | None = None
) -> str:
    """切片的语境前缀：文档名 + 位置 + 章节。

    切片单看常常缺主语（"同比增长 56%" 是哪家公司的？哪一年的？），
    向量化时把这份语境一起嵌进去，召回才分得清同名指标；同时也让
    模型引用时有据可依。存储的正文保持干净——前缀只在嵌入时拼。
    """
    stem = Path(filename).stem if filename else "文档"
    suffix = Path(filename or "").suffix.lower()
    if kind == TABLE:
        if page is None:
            where = "表格"
        elif suffix in _SHEET_EXTS:
            where = f"表{page}"
        else:
            where = f"第{page}页"
    elif kind == FIGURE:
        where = f"第{page}页图表" if page is not None else "图表"
    else:
        where = f"第{page}页" if page is not None else ""
    base = f"《{stem}》{where}" if where else f"《{stem}》"
    return f"{base} · {section}" if section else base


def embedding_input(
    text: str,
    *,
    filename: str,
    page: int | None,
    kind: str,
    section: str | None = None,
) -> str:
    """送进向量模型的文本 = 语境前缀 + 正文。

    章节用完整标签（一级 + 二级）：试过只取一级，MRR 从 0.879 掉到 0.869
    ——二级标题里的「非经常性损益」「股东信息」恰恰是把数字表区分开的那几个
    字，砍掉反而更难找。

    HTML 表先换成单元格文字再嵌入。标签占掉的向量维度不区分「货币资金」
    和「应收账款」，整张表合成一个点时更是把要找的那一行稀释掉。
    """
    body = html_table_plain(text)
    return f"{context_prefix(filename, page=page, kind=kind, section=section)}\n{body}"


def _split_long(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    out: list[str] = []
    buf = ""
    for sent in (s for s in _SENT_SPLIT.split(paragraph) if s.strip()):
        if len(sent) > max_chars:
            # 单句仍超长（无标点的连续文本）：硬切，保留 overlap
            if buf:
                out.append(buf)
                buf = ""
            step = max(max_chars - overlap, 1)
            for i in range(0, len(sent), step):
                piece = sent[i : i + max_chars]
                if len(piece) > overlap or i + max_chars >= len(sent):
                    out.append(piece)
            continue
        if buf and len(buf) + len(sent) > max_chars:
            out.append(buf)
            # 保留尾部 overlap 作为下一片开头
            buf = buf[-overlap:] if overlap > 0 else ""
        buf += sent
    if buf.strip():
        out.append(buf)
    return [s.strip() for s in out if s.strip()]
