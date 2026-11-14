"""切片前置清洗：去噪声、合并小表、去页码样板、去重复片。

由原 chunking.py 拆分而来，逻辑未改动。
"""

import re

from dataclasses import replace

from ai_agent.kb.ingest.parsing import FIGURE, Segment, TABLE, TEXT

from ai_agent.core.logging import get_logger

from ai_agent.kb.ingest.chunking.common import (
    Chunk,
    _HTML_TABLE,
)

from ai_agent.kb.ingest.chunking.html import (
    _html_header_to_carry,
    _html_pieces,
    _html_row_is_clipped_header,
    _html_row_is_data,
    _html_row_is_item,
    _row_texts,
    _row_width,
    _split_html_table,
    _table_rows,
)

from ai_agent.kb.ingest.chunking.text import (
    _PARA_SPLIT,
    _normalize,
    _plain_len,
    _same_page,
    _split_long,
)


logger = get_logger(__name__)


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

def _drop_page_boilerplate(segments: list[Segment]) -> list[Segment]:
    """剔除页眉页脚：同一行在 ≥3 页重复出现就判为样板。

    实测《机器人 2026 半年报》有 14 片一模一样的内容——
    「沈阳新松机器人自动化股份有限公司2026 年半年度报告全文」，正文每一页
    的页脚都印一遍；《汇川技术》财报页脚则是「公司负责人：… 主管会计工作
    负责人：… 会计机构负责人：…」，15 页各一份。

    这些片没有信息量，却会占检索名额（问「机器人 半年报」可能命中那行页脚）。
    按"跨页重复"判断而不是写死字符串：换一份文档、换个公司名同样生效。
    """
    pages_by_line: dict[str, set[int]] = {}
    text_pages: set[int] = set()
    for seg in segments:
        if seg.kind != TEXT or not seg.page:
            continue
        text_pages.add(seg.page)
        for line in (seg.text or "").splitlines():
            text = line.strip()
            if not (_BOILER_MIN_CHARS <= len(text) <= _BOILER_MAX_CHARS):
                continue
            if _HEADING_LINE.match(text):
                continue
            pages_by_line.setdefault(text, set()).add(seg.page)
    total_pages = max(len(text_pages), 1)
    boiler = {
        line
        for line, pages in pages_by_line.items()
        if _looks_like_boilerplate(line, len(pages), total_pages)
    }
    if not boiler:
        return segments
    out: list[Segment] = []
    for seg in segments:
        if seg.kind != TEXT or not seg.page:
            out.append(seg)
            continue
        kept = [
            line
            for line in (seg.text or "").splitlines()
            if line.strip() and line.strip() not in boiler
        ]
        text = "\n".join(kept).strip()
        if text:
            out.append(
                Segment(text, seg.page, seg.kind, seg.section, seg.scope, seg.page_end)
            )
    logger.info("剔除页眉页脚样本行 %d 条（跨页重复）", len(boiler))
    return out

def _looks_like_boilerplate(line: str, pages: int, total_pages: int) -> bool:
    """这一行是不是页眉/页脚/签名栏。

    不能只看「跨 3 页重复」：实测机器人那份报告里「确定该组合依据的说明：」
    在 6 页出现、「减变动金额」在 4 页出现——它们是**正文/表格碎片**（表格行
    被切断后落进正文段），删掉会把数据弄丢。

    也不能只看「含报告类型」：解析器常把页眉与正文拼成一行（
    ``…2026 年半年度报告全文：☑（已勾选）适用``），那一行里还有真内容。

    所以只认两种**整行**形态：签名栏、页眉整行（前后没有别的字）。其余一律
    不动——宁可留几条样板，不能删掉数据。
    """
    if pages < _BOILER_MIN_PAGES:
        return False
    return bool(_SIGNATURE_LINE.match(line) or _FOOTER_ONLY.match(line))

def _drop_duplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """同一文件内正文完全相同的片只留第一份。

    来源有三类：页眉页脚（上面已按跨页重复清掉大部分）、同一张权益变动表在
    同一页被文字层和 OCR 各切一遍、以及 `_expand_long_html_tables` 二次切分
    时的重叠。重复片会同时占多个检索位次，模型看到的却是同一份内容。

    只去**同一页内**的重复：同样的表出现在不同页（正文与附录各印一次）是有
    意的，跨页合并会把两处出处压成一个，引用位置也就丢了。
    """
    # 先按 (页, 归一化正文) 精确去重，再把「被同页更大一片包含」的更小片丢掉：
    # OCR 兜底时常把同一张表切成一大一小两片（长的 872 字、短的 552 字，后者
    # 几乎是前者的子串），两片会同时进 top_k 占掉名额。
    seen: set[tuple[int | None, str]] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        key = (chunk.page, re.sub(r"\s+", "", chunk.text or ""))
        if not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)

    by_page: dict[int | None, list[tuple[str, Chunk]]] = {}
    for chunk in unique:
        by_page.setdefault(chunk.page, []).append(
            (re.sub(r"\s+", "", chunk.text or ""), chunk)
        )
    dropped: set[int] = set()
    for items in by_page.values():
        items.sort(key=lambda pair: len(pair[0]), reverse=True)
        for i, (_big, _chunk) in enumerate(items):
            for small_text, small_chunk in items[i + 1 :]:
                if small_text and id(small_chunk) not in dropped and any(
                    small_text in big for big, _ in items[:i]
                ):
                    dropped.add(id(small_chunk))
    out = [chunk for chunk in unique if id(chunk) not in dropped]
    if len(out) != len(chunks):
        logger.info("切片去重 %d 片（含同页包含关系）", len(chunks) - len(out))
    return out

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

# 页眉页脚候选：一行在多页重复出现。太长的不算（正文里重复的句子不该删），
# 太短的不算（「合计」「不适用」这类单元格值本身没信息量，但也不能乱删）
_BOILER_MIN_CHARS = 5

_BOILER_MAX_CHARS = 60

# 出现在几页以上算样板
_BOILER_MIN_PAGES = 3

# 标题行不参与（章节名常常每页重印，但它有用）
_HEADING_LINE = re.compile(r"^(?:[一二三四五六七八九十]+[、．.]|[（(][一二三四五六七八九十]+[）)])")

# 签名栏：财报每页底部「公司负责人：X」这类整行
_SIGNATURE_LINE = re.compile(r"^[\u4e00-\u9fa5]{0,8}负责人\s*[:：]\s*\S{1,12}$")

# 页眉整行：公司名 + 报告类型 + 「全文」
_FOOTER_ONLY = re.compile(r"^.{0,40}(?:年度报告|半年度报告|季度报告)全文$")
