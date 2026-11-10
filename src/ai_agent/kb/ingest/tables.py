"""PDF 表格识别与页面组织。

从 parsing.py 拆出(S3.5 二次拆分自 tables.py)。PDF 没有表格对象，只有画
出来的线和摆好的字：pymupdf find_tables 靠线条反推单元格，无框释义表
("词语 …… 指 ……")靠文字对齐单独识别。

本模块负责逐页找出表格、跨页/同页续表合并、补表题与元信息(单位、币种、
审计类型)、标注合并/母公司口径、剔除签字栏，最后调用 table_cells 把原始
网格洗净成二维行，交给切片层整表成片。
"""

from __future__ import annotations

import re
from pathlib import Path

from ai_agent.core.logging import get_logger
from ai_agent.kb.ingest import sections
from ai_agent.kb.ingest.table_cells import (
    _CJK,
    _CJK_RE,
    _clean_table,
    _is_header_row,
    _is_item_row,
    _tidy_header,
)

logger = get_logger(__name__)

logger = get_logger(__name__)
# 跨页续表：表格顶边在这个纵坐标以内就算"从页顶开始"
_HEAD_CARRY_Y = 130.0
# 表题离表体多远以内还算它的标题（pt）。财务报表的「母公司利润表」离表体
# 约 58pt，中间还夹着「2026 年1—9 月」「编制单位：」「单位：」三行，
# 窗口太窄就只找到元信息行
_CAPTION_WINDOW = 120.0
# 表题候选里要跳过的是表格元信息，不是表名
_CAPTION_NOISE = re.compile(
    r"^(单位|币种|金额单位|编制单位|审计类型|数据来源|资料来源|注\s*[:：])"
    r"|^[√□■☑]|^(适用|不适用)|^\d{4}\s*年|^\d+\s*/\s*\d+$"
)
# 表题的结构：以「表」结尾（合并资产负债表/母公司利润表）、带（续）、
# 或以节编号开头（（一）主要会计数据和财务指标）。只认这三种，宁缺毋滥——
# 认错表题比没有表题更坏，会把别的表的主体安到这张表上
_CAPTION_SHAPE = re.compile(
    r"(表$|（续）$"
    r"|^\(?[一二三四五六七八九十]+[、.]\s*[^\d\s]"
    r"|^（[一二三四五六七八九十]+）"
    # 编号后面必须接非数字：「1、截至本报告期末」是标题，「3.1%，较上年…」是句子
    r"|^\d{1,2}\s*[、.．]\s*[^\d\s])"
)
# 以这些标点收尾的是句子/引导语，不是表题：「…如下：」「…提升0.2 个百分点。」
_CAPTION_SENTENCE = re.compile(r"[。：；，、]$")
# 表题的长度上限：再长就是正文句子了
_CAPTION_MAX_CHARS = 40
# 上一页最后一张表的底边离页底不足这么远，说明它是被页面切断的、不是自己结束的
# 表格底边离页底多近才算「被分页截断」。实测季报/年报的页脚只有一行页码，
# 表体常停在距页底 100–105pt（Y=738~746，页高 842），90 会漏判一半续表。
_PAGE_BOTTOM_SLACK = 110.0
# 同页被识别器拆成两截时，两框之间允许的最大空隙。再大更像上下两张独立表
_SAME_PAGE_GAP = 72.0


# ——— PDF 释义表 ———
# 招股书开头的「词语 …… 指 …… 含义」表没有框线，find_tables 靠线条
# 反推单元格，这种表整页漏掉。好在它的版式极其固定：孤立的「指」作为
# 连接词竖着对成一列，左边是词条、右边是释义——拿「指」当锚点，按左右
# 两列拼行就能还原，不需要猜任何表格线。

# 连接词：锚点必须是整个词（「本指导」里的指不算）
_GLOSS_ANCHORS = {"指", "系指", "是指"}
# 一页至少这么多行才当释义表——正文里偶尔也有孤立的「指」，数量撑不起来
_GLOSS_MIN_ROWS = 3
# 锚点列的对齐容差：同一列的「指」x0 相差不会超过这个数
_GLOSS_X_TOL = 18.0


def _pdf_glossary_tables(
    doc, exclude: list[tuple[int, tuple[float, float, float, float]]]
) -> list[tuple[int, float, tuple, list[list[str]]]]:
    """识别无框释义表，返回值结构与 _pdf_tables 相同（可直接合并归并）。"""
    out: list[tuple[int, float, tuple, list[list[str]]]] = []
    for pno in range(doc.page_count):
        try:
            words = doc[pno].get_text("words")
        except Exception:
            continue
        rows, bbox = _glossary_rows(_word_lines(words))
        if len(rows) < _GLOSS_MIN_ROWS or bbox is None:
            continue
        if any(
            pno + 1 == page and _overlap_ratio(bbox, box) > 0.3
            for page, box in exclude
        ):
            # 这片区域 find_tables 已经抽到了（页上还残留些框线），
            # 再出一份就是重复切片
            continue
        out.append(
            (
                pno + 1,
                float(bbox[1]),
                bbox,
                # Markdown 表格首行即表头（前端渲染成灰底），补一列语义
                # 表头，别把第一条释义顶上去当列名
                [["词语", "含义"], *rows],
            )
        )
    return out


def _word_lines(
    words: list[tuple[float, float, float, float, str, int, int, int]],
) -> list[list[tuple[float, float, float, float, str]]]:
    """pymupdf 的 words → 视觉行（按 y0 聚类，行内按 x 排序）。"""
    out: list[list[tuple[float, float, float, float, str]]] = []
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        item = (w[0], w[1], w[2], w[3], w[4])
        if out and item[1] - out[-1][0][1] <= 4:
            out[-1].append(item)
        else:
            out.append([item])
    for line in out:
        line.sort(key=lambda w: w[0])
    return out


def _glossary_rows(
    lines: list[list[tuple[float, float, float, float, str]]],
) -> tuple[list[list[str]], tuple | None]:
    """视觉行 → [词条, 释义] 二维行。

    每条释义由锚点行开头，随后的行只要没有字压在锚点列上（词条换行 /
    释义换行，或两边同时换行）都算这条的延续；一旦某行的字横跨锚点列，
    说明表格结束了、回到正文。
    """
    anchors: list[tuple[int, tuple[float, float, float, float, str]]] = []
    for i, line in enumerate(lines):
        for w in line:
            if w[4] in _GLOSS_ANCHORS:
                anchors.append((i, w))
                break
    if len(anchors) < _GLOSS_MIN_ROWS:
        return [], None
    xs = sorted(a[1][0] for a in anchors)
    if xs[-1] - xs[0] > _GLOSS_X_TOL:
        return [], None

    def cell_text(span: list[tuple]) -> str:
        """一列里的若干词拼回文字：同视觉行的按间距拼，跨行直接连。"""
        parts: list[str] = []
        for line in span:
            text = ""
            prev = None
            for w in line:
                if prev is not None and w[0] - prev[2] > 2.5:
                    # 中文之间的小间隙是排版字距（"集团公司 有"），直连；
                    # 西文之间留真空格
                    gap_cjk = _CJK_RE.search(prev[4][-1]) or _CJK_RE.match(w[4])
                    text += "" if gap_cjk else " "
                text += w[4]
                prev = w
            if text:
                parts.append(text)
        return "".join(parts)

    rows: list[list[str]] = []
    box: list[float] | None = None
    for k, (i, anchor) in enumerate(anchors):
        ax0, ay0, ax1, _ay1, _t = anchor
        stop = anchors[k + 1][0] if k + 1 < len(anchors) else len(lines)
        term_span: list[list[tuple]] = []
        def_span: list[list[tuple]] = []
        ended = True
        for line in lines[i:stop]:
            # 锚点行上锚点自己的字就在锚点列，不算「压线」
            if all(w is anchor or w[2] < ax0 - 2 or w[0] > ax1 + 2 for w in line):
                left = [w for w in line if w[2] < ax0 - 2 and w is not anchor]
                right = [w for w in line if w[0] > ax1 + 2 and w is not anchor]
                if left:
                    term_span.append(left)
                if right:
                    def_span.append(right)
                for w in line:
                    box = box if box is not None else list(w[:4])
                    box[0] = min(box[0], w[0])
                    box[1] = min(box[1], w[1])
                    box[2] = max(box[2], w[2])
                    box[3] = max(box[3], w[3])
            else:
                # 字压在锚点列上 → 表格结束、回到正文。这条本身在压线行
                # 之前已经收完，照常保留；后面的锚点（正文里恰好孤立的
                # 「指」）不能再收割
                ended = False
                break
        term, definition = cell_text(term_span), cell_text(def_span)
        if term and definition:
            rows.append([term, definition])
        if not ended:
            break
    return rows, tuple(box) if box else None


def _overlap_ratio(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ox = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ox * oy / max((a[2] - a[0]) * (a[3] - a[1]), 1.0)


def _header_has_cjk(raw: list[list[str | None]]) -> bool:
    """表头行是否带中文列名。三线表被切掉「资产」列时，表头只剩日期数字。"""
    if not raw:
        return False
    return any(_CJK_RE.search(cell or "") for cell in raw[0])


def _replace_clipped_tables(found: list, alt: list) -> list:
    """用 lines_strict 的结果替换表头没有中文的表（bbox 重叠的那张）。"""
    used: set[int] = set()
    out: list = []
    for table in found:
        raw = table.extract()
        if _header_has_cjk(raw):
            out.append(table)
            continue
        tb = tuple(table.bbox)
        pick = None
        for j, other in enumerate(alt):
            if j in used:
                continue
            if _overlap_ratio(tb, tuple(other.bbox)) < 0.3:
                continue
            if not _header_has_cjk(other.extract()):
                continue
            pick = (j, other)
            break
        if pick is not None:
            used.add(pick[0])
            out.append(pick[1])
        else:
            out.append(table)
    return out


def _find_page_tables(page) -> list:
    """默认按线条抽表；表头被切掉中文列名时改用 lines_strict。

    招股书三线表只有横线和数字区的短竖线，默认策略会把「资产」列和
    最右一列年份切到框外，抽成纯数字网格。lines_strict 认完整外框，
    但不能当默认——合并单元格的表（股东名册、发行费用）列数会变少。
    """
    try:
        found = list(page.find_tables().tables)
    except Exception:
        found = []
    clipped = [t for t in found if not _header_has_cjk(t.extract())]
    if not found or not clipped:
        return found
    try:
        alt = list(page.find_tables(strategy="lines_strict").tables)
    except Exception:
        return found
    if not alt:
        return found
    return _replace_clipped_tables(found, alt)


# 单元格里「数字被列宽折成两行」的两段：都要是数字片段，拼起来还得是合法数字
_NUM_FRAGMENT = re.compile(r"^[-\d,.%]+$")
_NUMBER_WHOLE = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$|^-?\d+(?:\.\d+)?%?$")


def _ascii_word(ch: str) -> bool:
    return bool(ch) and ch.isascii() and ch.isalnum()


def _join_word_line(words: list) -> str:
    """同一视觉行里的词拼成一段：汉字之间不留空格，拉丁词之间留一个。

    只认「两边都是 ASCII 字母数字」才补空格——中文不补（「商业承兑票」+
    「据」），百分号/逗号前也不补（「14.56」+「%」）。
    """
    text = ""
    for word in words:
        piece = word[4]
        if not text:
            text = piece
        elif _ascii_word(text[-1]) and _ascii_word(piece[0]):
            text += " " + piece
        else:
            text += piece
    return text


def _join_cell_lines(texts: list[str]) -> str:
    out = texts[0] if texts else ""
    for text in texts[1:]:
        if out and text and _ascii_word(out[-1]) and _ascii_word(text[0]):
            out += " " + text
        else:
            out += text
    return out


def _cell_text_from_words(words: list, rect) -> str:
    """按坐标重建一个单元格的文本。

    `table.extract()` 在这一类表上是**错的**：数字被列宽折成两行时，它按字符
    坐标拼出「850075 / , , / 1434 / .」这种乱序串。这里改用 word 坐标自己排：
    先按 y 分行、行内按 x，再把「数字折行」拼回去（两段都是数字、右边缘对齐、
    拼起来是合法数字才合并，例如 12,380,4 + 18.39 → 12,380,418.39）。
    """
    inside = [
        w
        for w in words
        if rect[0] <= (w[0] + w[2]) / 2 <= rect[2]
        and rect[1] <= (w[1] + w[3]) / 2 <= rect[3]
    ]
    if not inside:
        return ""
    inside.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines: list[list] = []
    for word in inside:
        yc = (word[1] + word[3]) / 2
        if lines and abs((lines[-1][0][1] + lines[-1][0][3]) / 2 - yc) <= 4:
            lines[-1].append(word)
        else:
            lines.append([word])
    texts = [_join_word_line(line) for line in lines]
    if len(texts) == 2 and all(_NUM_FRAGMENT.match(text) for text in texts):
        right = [round(line[-1][2], 1) for line in lines]
        joined = texts[0] + texts[1]
        if abs(right[0] - right[1]) <= 3 and _NUMBER_WHOLE.match(joined):
            return joined
    return _join_cell_lines(texts)


def _table_rows_from_words(page, table, words: list) -> list[list[str]]:
    """用 word 坐标重建整张表；任何一行都取不到词就返回空（调用方回退 extract）。"""
    rows: list[list[str]] = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            cells.append("" if cell is None else _cell_text_from_words(words, cell))
        rows.append(cells)
    return rows


def _page_table_payload(page, pno: int) -> list[tuple[int, float, tuple, list[list[str]]]]:
    """单页 find_tables → 可跨进程传递的清洗后表格。"""
    out: list[tuple[int, float, tuple, list[list[str]]]] = []
    words = page.get_text("words")
    for table in _find_page_tables(page):
        # 一张图里可能拆出多张表（见 _split_repeated_header）：共用同一
        # 个 bbox 即可，正文行照样整块剔掉。外框常比最后一行格子大一截，
        # 会把表下的签字栏（法定代表人）也当成表格内文本删掉。
        bbox = _table_content_bbox(table)
        rebuilt = _table_rows_from_words(page, table, words)
        # 重建结果若与 extract 的非空格数差太多（合并单元格的矩形取错等），
        # 宁可回退到 pymupdf 自己的结果，也不要凭空造空表
        raw = table.extract()
        source = raw
        if rebuilt and _filled(rebuilt) >= _filled(raw) * 0.8:
            source = rebuilt
        for rows in _clean_table(source):
            if len(rows) < 2 or max(len(r) for r in rows) < 2:
                continue
            out.append((pno + 1, float(bbox[1]), bbox, rows))
    return out


def _filled(rows: list[list[str]]) -> int:
    return sum(1 for row in rows for cell in row if (cell or "").strip())


def _pdf_tables(
    doc,
) -> tuple[list[tuple[int, float, tuple, list[list[str]]]], dict[int, float]]:
    """逐页找表格，返回 (表列表, 页高)。表是 (页码, 顶部纵坐标, bbox, 清洗后的二维行)。

    同进程调用：单测与调试用。生产路径走 ``_pdf_tables_isolated``，避免
    MuPDF find_tables SIGSEGV 拖死整个解析子进程。
    """
    out: list[tuple[int, float, tuple, list[list[str]]]] = []
    heights: dict[int, float] = {}
    for pno in range(doc.page_count):
        try:
            heights[pno + 1] = float(doc[pno].rect.height)
        except Exception:
            heights[pno + 1] = 0.0
        try:
            out.extend(_page_table_payload(doc[pno], pno))
        except Exception:
            logger.warning("PDF 第 %d 页表格识别失败，跳过", pno + 1, exc_info=True)
            continue
    return _carry_page_headers(out, heights), heights


def _pdf_tables_worker_main(path: str, start: int, out_path: str) -> None:
    """独立解释器入口：从 ``start``（0-based）页跑 find_tables，逐页写 checkpoint。

    用 ``python -c`` 拉起，不依赖 multiprocessing spawn（嵌套在解析子进程里、
    或父进程是 stdin 时 spawn 会踩 ``__main__``）。每页写完 pickle，父进程在
    SIGSEGV 后仍能读到 last_ok 并跳过崩溃页。
    """
    import pickle

    import pymupdf

    tables: list[tuple[int, float, tuple, list[list[str]]]] = []
    heights: dict[int, float] = {}
    last_ok = start - 1

    def _checkpoint(*, done: bool) -> None:
        import os

        tmp = f"{out_path}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(
                {
                    "tables": tables,
                    "heights": heights,
                    "last_ok": last_ok,
                    "done": done,
                },
                fh,
            )
        os.replace(tmp, out_path)

    doc = pymupdf.open(path)
    try:
        for pno in range(start, doc.page_count):
            try:
                heights[pno + 1] = float(doc[pno].rect.height)
            except Exception:
                heights[pno + 1] = 0.0
            try:
                tables.extend(_page_table_payload(doc[pno], pno))
            except Exception:
                # 单页 Python 异常：记空页继续，别让整段抽表停掉
                pass
            last_ok = pno
            _checkpoint(done=False)
        _checkpoint(done=True)
    finally:
        doc.close()


def _pdf_tables_isolated(
    path: Path,
) -> tuple[list[tuple[int, float, tuple, list[list[str]]]], dict[int, float]]:
    """在短命子进程里跑 find_tables；SIGSEGV 只丢崩溃页，正文解析继续。"""
    import pickle
    import subprocess
    import sys
    import tempfile

    pdf = Path(path)
    try:
        import pymupdf
    except ImportError:
        return [], {}
    try:
        doc = pymupdf.open(str(pdf))
        page_count = doc.page_count
        doc.close()
    except Exception:
        logger.warning("PDF 打开失败，跳过表格识别：%s", pdf.name, exc_info=True)
        return [], {}
    if page_count <= 0:
        return [], {}

    out: list[tuple[int, float, tuple, list[list[str]]]] = []
    heights: dict[int, float] = {}
    start = 0
    while start < page_count:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            out_path = tmp.name
        code = (
            "from ai_agent.kb.ingest.tables import _pdf_tables_worker_main as _m; "
            f"_m({str(pdf)!r}, {start}, {out_path!r})"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=max(120, page_count * 5),
            )
        except subprocess.TimeoutExpired:
            logger.error("PDF 表格子进程超时 file=%s start_page=%d", pdf.name, start + 1)
            try:
                Path(out_path).unlink(missing_ok=True)
            except OSError:
                pass
            break

        done = False
        last_ok = start - 1
        try:
            raw = Path(out_path).read_bytes()
            # 子进程在写首个 checkpoint 前就 SIGSEGV 时，文件仍是 NamedTemporaryFile
            # 创建的空文件——不当成读失败。
            if raw:
                state = pickle.loads(raw)
                batch = state.get("tables") or []
                out.extend(batch)
                heights.update(state.get("heights") or {})
                last_ok = int(state.get("last_ok", last_ok))
                done = bool(state.get("done"))
        except FileNotFoundError:
            pass
        except Exception:
            logger.warning("PDF 表格 checkpoint 读取失败 file=%s", pdf.name, exc_info=True)
        finally:
            try:
                Path(out_path).unlink(missing_ok=True)
            except OSError:
                pass
            try:
                Path(f"{out_path}.tmp").unlink(missing_ok=True)
            except OSError:
                pass

        if done and proc.returncode == 0:
            break
        # 原生崩溃：正在处理的是 last_ok+1；若一页都没写出则是 start
        crash_page = last_ok + 1
        if crash_page < start:
            crash_page = start
        if crash_page >= page_count:
            logger.error(
                "PDF 表格子进程异常退出 exit=%s（已无后续页可跳过）file=%s",
                proc.returncode,
                pdf.name,
            )
            break
        logger.warning(
            "PDF 第 %d 页 find_tables 原生崩溃 exit=%s，跳过该页继续 file=%s",
            crash_page + 1,
            proc.returncode,
            pdf.name,
        )
        start = crash_page + 1
    return _carry_page_headers(out, heights), heights


# 续表清洗时常丢掉的空列名。table-stitcher 叫 width drift：列数差几列
# 仍可能是同一张表，不能因为「附注」空列被删就放弃补头。
_OPTIONAL_HEADER_COLS = frozenset({"附注", "注释", "行次", "序号"})
_MAX_WIDTH_DRIFT = 2


def _fit_header_to_width(header: list[str], width: int) -> list[str] | None:
    """把上页表头压到本页列数。压不动就返回 None（不硬凑）。"""
    cells = list(header)
    if len(cells) == width:
        return _tidy_header(cells)
    if len(cells) < width or len(cells) - width > _MAX_WIDTH_DRIFT:
        return None
    trimmed = [c for c in cells if c not in _OPTIONAL_HEADER_COLS]
    if len(trimmed) == width:
        return _tidy_header(trimmed)
    trimmed = [c for c in trimmed if c]
    if len(trimmed) == width:
        return _tidy_header(trimmed)
    trimmed = [c for c in cells if c]
    if len(trimmed) == width:
        return _tidy_header(trimmed)
    return None


def _pad_rows_to_header(
    rows: list[list[str]], header: list[str]
) -> list[list[str]] | None:
    """本页列数偏少时，按表头里的可选列位置补空格，对齐到上页宽度。"""
    hw = len(header)
    if hw <= 0:
        return None
    out: list[list[str]] = []
    for row in rows:
        cells = list(row)
        if len(cells) == hw:
            out.append(cells)
            continue
        if len(cells) > hw or hw - len(cells) > _MAX_WIDTH_DRIFT:
            return None
        for i, name in enumerate(header):
            if len(cells) >= hw:
                break
            if name in _OPTIONAL_HEADER_COLS and (
                i >= len(cells) or cells[i] != name
            ):
                cells.insert(i, "")
        if len(cells) < hw:
            cells.extend([""] * (hw - len(cells)))
        if len(cells) != hw:
            return None
        out.append(cells)
    return out


def _carry_page_headers(
    tables: list[tuple[int, float, tuple, list[list[str]]]],
    page_heights: dict[int, float],
) -> list[tuple[int, float, tuple, list[list[str]]]]:
    """跨页续表补表头。

    PDF 里一张表排到页底、下一页接着排时，第二页**不重印列名**，抽出来
    第一行就是数据。原样渲染成 Markdown 后首行被当成表头，于是
    「基本每股收益（元/股） | 0.01 | …」成了列名，后面所有数字都没有
    列名可对——而 `_chunk_table` 又会把这一行当表头复制到每一片。

    补的条件同时成立才动手，宁可漏补也不误补：
    - 表格在页面顶部（续表只会从顶部开始）；
    - 首行不是表头行（有数字、或是科目/股东名一类的数据行）；
    - 上一页最后一张表排到了页底（说明它是被切断的，不是自己结束的）；
    - 列数一致，或仅差可选空列（附注等，width drift ≤ 2）。
    """
    prev_bottom: dict[int, float] = {}
    for pno, _y, bbox, _rows in tables:
        prev_bottom[pno] = float(bbox[3])

    out: list[tuple[int, float, tuple, list[list[str]]]] = []
    last_header: list[str] | None = None
    for pno, y, bbox, rows in tables:
        width = max(len(row) for row in rows)
        if (
            y <= _HEAD_CARRY_Y
            # 「不是表头行」而非「必须是科目行」：股东名册这类表的金额列旁边
            # 还有「股东性质=其他」的文本列，按科目行判会漏掉续表
            and not _is_header_row(rows[0])
            and last_header is not None
            and _reached_page_bottom(
                prev_bottom.get(pno - 1), page_heights.get(pno - 1)
            )
        ):
            header = _tidy_header(last_header)
            if len(header) == width:
                rows = [header, *rows]
            else:
                fitted = _fit_header_to_width(header, width)
                if fitted is not None:
                    rows = [fitted, *rows]
                else:
                    padded = _pad_rows_to_header(rows, header)
                    if padded is not None:
                        rows = [header, *padded]
        out.append((pno, y, bbox, rows))
        if _is_header_row(rows[0]):
            last_header = rows[0]
    return out


def _reached_page_bottom(bottom: float | None, height: float | None) -> bool:
    if not bottom or not height:
        return False
    return bottom >= height - _PAGE_BOTTOM_SLACK


def _merge_continued_tables(
    tables: list[tuple[int, float, tuple, list[list[str]], str | None]],
    page_heights: dict[int, float],
) -> list[tuple[int, float, tuple, list[list[str]], str | None, int]]:
    """把续表拼回一张。

    断开的常见原因有两种，判据分开写：
    - 跨页：上一张排到页底、下一张从页顶接着（列名可能重印，也可能只剩数据）；
    - 同页：识别器在缺线、注释处把一张表切成上下两框，框距很近、列数相同。

    左右并排的表 y 接近但 x 几乎不重叠，不能并。表题不同也不能并。
    """
    if not tables:
        return []
    ordered = sorted(tables, key=lambda item: (item[0], item[1]))
    out: list[tuple[int, float, tuple, list[list[str]], str | None, int]] = []
    for page, y, bbox, rows, caption in ordered:
        if out and _is_table_continuation(
            out[-1], (page, y, bbox, rows, caption), page_heights
        ):
            p0, y0, b0, r0, c0, _end = out[-1]
            out[-1] = (p0, y0, b0, _concat_table_rows(r0, rows), c0 or caption, page)
        else:
            out.append((page, y, bbox, rows, caption, page))
    return out


def _normalize_table_widths(
    head: list[list[str]], tail: list[list[str]]
) -> tuple[list[list[str]], list[list[str]]] | None:
    """合并前把两段表对齐到同一列数；差太多则放弃。"""
    hw = max(len(r) for r in head)
    tw = max(len(r) for r in tail)
    if hw == tw:
        return head, tail
    if abs(hw - tw) > _MAX_WIDTH_DRIFT:
        return None
    if hw > tw and _is_header_row(head[0]):
        padded = _pad_rows_to_header(tail, head[0])
        if padded is not None:
            return head, padded
        fitted = _fit_header_to_width(head[0], tw)
        if fitted is None:
            return None
        # 表头可压到本页宽度：整段 head 也压（去掉对应可选列）
        drop = [
            i
            for i, cell in enumerate(head[0])
            if cell in _OPTIONAL_HEADER_COLS and cell not in fitted
        ]
        if not drop:
            return None
        slim_head = [
            [cell for i, cell in enumerate(row) if i not in drop]
            for row in head
        ]
        return slim_head, tail
    if tw > hw and _is_header_row(tail[0]):
        padded = _pad_rows_to_header(head, tail[0])
        if padded is not None:
            return padded, tail
    return None


def _concat_table_rows(
    head: list[list[str]], tail: list[list[str]]
) -> list[list[str]]:
    aligned = _normalize_table_widths(head, tail)
    if aligned is None:
        return [*head, *tail]
    head, tail = aligned
    if tail and head and (
        tail[0] == head[0] or _is_header_row(tail[0])
    ):
        # 重印表头或补过头的续表：丢掉下段表头行
        return [*head, *tail[1:]]
    return [*head, *tail]


def _is_table_continuation(
    prev: tuple[int, float, tuple, list[list[str]], str | None, int],
    cur: tuple[int, float, tuple, list[list[str]], str | None],
    page_heights: dict[int, float],
) -> bool:
    p_page, _py, p_bbox, p_rows, p_cap, _pend = prev
    c_page, c_y, c_bbox, c_rows, c_cap = cur
    if not p_rows or not c_rows:
        return False
    pw = max(len(r) for r in p_rows)
    cw = max(len(r) for r in c_rows)
    if abs(pw - cw) > _MAX_WIDTH_DRIFT:
        return False
    if pw != cw and _normalize_table_widths(p_rows, c_rows) is None:
        return False
    if p_cap and c_cap and p_cap != c_cap:
        return False
    # 续表要么重印同一行表头，要么第一行已经是科目（金额可以为空）
    if p_rows[0] != c_rows[0] and not _is_item_row(c_rows[0]) and not _is_header_row(
        c_rows[0]
    ):
        return False
    if not _boxes_vertically_stacked(p_bbox, c_bbox):
        return False
    if c_page == p_page + 1:
        return c_y <= _HEAD_CARRY_Y and _reached_page_bottom(
            float(p_bbox[3]), page_heights.get(p_page)
        )
    if c_page == p_page:
        gap = c_y - float(p_bbox[3])
        return -8.0 <= gap <= _SAME_PAGE_GAP
    return False


def _boxes_vertically_stacked(prev_bbox: tuple, cur_bbox: tuple) -> bool:
    """左右并排的表 x 几乎不重叠，不能当成上下续表。"""
    px0, _py0, px1, _py1 = (float(v) for v in prev_bbox)
    cx0, _cy0, cx1, _cy1 = (float(v) for v in cur_bbox)
    overlap = min(px1, cx1) - max(px0, cx0)
    width = min(px1 - px0, cx1 - cx0)
    return width > 0 and overlap >= width * 0.5


def _attach_captions(
    tables: list[tuple[int, float, tuple, list[list[str]]]],
    lines: list[sections.Line],
) -> list[tuple[int, float, tuple, list[list[str]], str | None]]:
    """给每张表找回表题，返回多带一个 caption 字段的五元组。

    为什么必须做：表格切片的正文只有「项目 | 本期 | 上期」和一堆数字，
    **表题（「母公司利润表」）不进来**。于是"母公司营业收入是多少"这种问题，
    数值所在的切片里根本没有"母公司"三个字——向量和关键词都够不着，只能
    靠相邻的标题切片勉强召回，而那一片没有数值。同理，图谱抽取也拿不到
    "这张表属于谁"，metric 实体因此连不到 org 上（实测孤立率 44%）。

    用 kept（已剔页眉页脚）而不是原始行：页眉每页重复，直接往上找会把
    「XX公司2026年第三季度报告」误当表题。
    """
    out: list[tuple[int, float, tuple, list[list[str]], str | None]] = []
    prev_header: list[str] | None = None
    prev_caption: str | None = None
    for page, y, bbox, rows in tables:
        caption = _caption_above(lines, page, y)
        if (
            caption is None
            and y <= _HEAD_CARRY_Y
            and prev_header is not None
            and rows[0] == prev_header
        ):
            # 跨页续表：表题只在首页印一次。必须同时满足「从页顶开始」和
            # 「表头与上一张表一致」（_carry_page_headers 补过的表头就是上一张的），
            # 否则会把上一节的标题安到一张无关的表上——顺丰第 5 页三张
            # 页顶表就差点被冠上「（二）非经常性损益项目和金额」
            caption = prev_caption
        # 只跟紧邻的上一张表接续：中间隔了一张没表题的表，链就断了，
        # 免得把更早那张表的标题递到后面去
        prev_caption = caption
        prev_header = rows[0]
        out.append((page, y, bbox, rows, caption))
    return out


def _lines_above(
    lines: list[sections.Line], page: int, y0: float
) -> list[sections.Line]:
    """表体上方窗口内的行，从近到远。表题和单位都从这里捡。"""
    candidates = [
        ln
        for ln in lines
        if ln.page == page
        and ln.bbox
        and y0 - _CAPTION_WINDOW <= ln.bbox[3] <= y0 + 1
    ]
    candidates.sort(key=lambda ln: ln.bbox[3], reverse=True)
    return candidates


def _caption_above(lines: list[sections.Line], page: int, y0: float) -> str | None:
    """在表体上方找最近的一条「像表题」的行，找不到返回 None。

    从近往远走、跳过元信息行：正常的报表上方依次是「单位：千元」「编制单位：」
    「2026 年1—9 月」，真正的表题在更上面一行。
    """
    for ln in _lines_above(lines, page, y0):
        text = ln.text.strip()
        if not text or len(text) > _CAPTION_MAX_CHARS:
            continue
        if _CAPTION_NOISE.search(text):
            continue
        if _CAPTION_SENTENCE.search(text):
            continue
        if _CAPTION_SHAPE.search(text):
            return text
    return None


# 报表上方常把单位、币种、审计状态粘在一行：
# 「单位：千元币种：人民币审计类型：未经审计」。编制单位是主体名，太长
# 也容易和表题抢戏，不收。
_UNIT_FIELD = re.compile(
    r"(?:金额单位|(?<![编准])单位)[:：]\s*(.+?)(?=币种|审计类型|编制单位|$)"
)
_CURRENCY_FIELD = re.compile(
    r"币种[:：]\s*(.+?)(?=审计类型|金额单位|(?<![编准])单位|编制单位|$)"
)
_AUDIT_FIELD = re.compile(r"(?:审计类型[:：]\s*)?(未经审计|已审计|经审计)")
# 报表抬头里的编制单位与日期：不并进前缀的话，它们会单独切成一片只有
# 「合并资产负债表 / 2026 年9 月30 日 / 编制单位：…」的碎片，且与表段前缀重复。
_ORG_FIELD = re.compile(r"编制单位[:：]\s*(\S{2,40})")
_DATE_LINE = re.compile(r"^(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日$")


def _meta_fields_above(
    lines: list[sections.Line], page: int, y0: float
) -> tuple[str | None, str | None, str | None, list[sections.Line]]:
    """表上方的单位 / 币种 / 审计状态，以及取自哪几行。

    返回取用到的行：解析层要据此把这些行从正文里剔掉，否则「单位：股」
    会同时出现在表段前缀和正文切片里，各存一份。
    """
    unit = currency = audit = org = date = None
    used: list[sections.Line] = []
    for ln in _lines_above(lines, page, y0):
        text = ln.text.strip()
        if not text:
            continue
        hit = False
        if unit is None:
            m = _UNIT_FIELD.search(text)
            if m:
                unit = m.group(1).strip(" 。；;，,")
                hit = True
        if currency is None:
            m = _CURRENCY_FIELD.search(text)
            if m:
                currency = m.group(1).strip(" 。；;，,")
                hit = True
        if audit is None:
            m = _AUDIT_FIELD.search(text)
            if m:
                audit = m.group(1)
                hit = True
        if org is None:
            m = _ORG_FIELD.search(text)
            if m:
                org = m.group(1).strip(" 。；;，,")
                hit = True
        if date is None:
            m = _DATE_LINE.match(text)
            if m:
                date = f"{m.group(1)} 年{int(m.group(2))} 月{int(m.group(3))} 日"
                hit = True
        if hit:
            used.append(ln)
        if unit and currency and audit and org and date:
            break
    return (unit, currency, audit, org, date), used


def _caption_line_ids(
    captioned: list[tuple[int, float, tuple, list[list[str]], str | None]],
    lines: list[sections.Line],
) -> set[int]:
    """被某张表当作表题的正文行 id 集合。

    表题会写进表段前缀（`合并资产负债表\n单位：…`），正文里那一行若留着，
    就成了「只有一行表名」的碎片切片，和表段前缀重复。
    只在表体上方窗口内、文本与表题完全一致的行里挑最近的一条。
    """
    ids: set[int] = set()
    for page, y0, _bbox, _rows, caption in captioned:
        if not caption:
            continue
        for ln in _lines_above(lines, page, y0):
            if ln.text.strip() == caption.strip():
                ids.add(id(ln))
                break
    return ids


def _table_meta_line_ids(
    lines: list[sections.Line],
    tables: list[tuple[int, float, tuple, list[list[str]]]],
) -> set[int]:
    """被任意一张表用作前缀（单位/币种/审计）的正文行 id 集合。"""
    ids: set[int] = set()
    for page, y0, _bbox, _rows in tables:
        _fields, used = _meta_fields_above(lines, page, y0)
        ids.update(id(ln) for ln in used)
    return ids


def _table_meta_above(lines: list[sections.Line], page: int, y0: float) -> str | None:
    """从表上方捡单位 / 币种 / 审计状态，拼成跟在表题后面的一行。

    这些字不进切片的话，后半张表被切开后只剩数字，模型会把百万元当成元。
    近的行优先：两张表叠在一起时，要的是紧挨着这张表的那行。
    """
    unit, currency, audit, org, date = _meta_fields_above(lines, page, y0)[0]
    parts: list[str] = []
    if unit:
        parts.append(unit if unit.startswith("单位") else f"单位：{unit}")
    if currency:
        parts.append(currency if "币种" in currency else f"币种：{currency}")
    if audit:
        parts.append(audit)
    if date:
        parts.append(date)
    if org:
        parts.append(f"编制单位：{org}")
    return " · ".join(parts) if parts else None


def _table_prefix(caption: str | None, meta: str | None) -> str:
    """表题 + 单位行。口径判断只用表题，所以两者分开传、拼的时候再合。"""
    return "\n".join(part for part in (caption, meta) if part)


# 报表口径：同一个指标在「合并报表」与「母公司报表」里各有一份数，值不一样，
# 而用户的问法通常不带口径（「中国能建期末现金余额」默认问集团口径）。
# 不区分会错得毫无痕迹：能建期末现金，母公司 2,721,444、合并 77,436,294，
# 相差 28 倍，两片都含「期末现金及现金等价物余额」这串字。
SCOPE_CONSOLIDATED = "consolidated"
SCOPE_PARENT = "parent"
_PARENT_SCOPE = re.compile(r"母公司|个别报表|单体报表")
_CONSOLIDATED_SCOPE = re.compile(r"合并")


def _table_scope(caption: str | None) -> str | None:
    """从表题判断报表口径，判不出返回 None（不猜）。

    只看表题不看表体：表内的「其中：」小计项在两张表里都出现，拿它判口径
    会把合并表和母公司表判成同一个。
    """
    if not caption:
        return None
    if _PARENT_SCOPE.search(caption):
        return SCOPE_PARENT
    if _CONSOLIDATED_SCOPE.search(caption):
        return SCOPE_CONSOLIDATED
    return None


def _table_content_bbox(table) -> tuple:
    """用最后一行格子收紧表格外框，避免把表下的签字栏包进去。"""
    outer = tuple(table.bbox)
    try:
        rows = table.rows
    except Exception:
        return outer
    bottoms = [
        float(row.bbox[3])
        for row in (rows or [])
        if getattr(row, "bbox", None)
    ]
    if not bottoms:
        return outer
    return (outer[0], outer[1], outer[2], min(float(outer[3]), max(bottoms) + 6.0))


# 签字人姓名：葛海蛟 / 张辉，印在职务上一行，和职务一起留
_SIGNATORY_NAME = re.compile(rf"^[{_CJK}]{{2,4}}$")


def _signatory_pages(lines: list[sections.Line]) -> dict[int, list[float]]:
    pages: dict[int, list[float]] = {}
    for ln in lines:
        if sections.is_signatory(ln):
            pages.setdefault(ln.page, []).append(ln.y_ratio)
    return pages


def _keep_signatory_block(ln: sections.Line, pages: dict[int, list[float]]) -> bool:
    if sections.is_signatory(ln):
        return True
    ys = pages.get(ln.page)
    if not ys or not _SIGNATORY_NAME.match(ln.norm):
        return False
    return any(abs(ln.y_ratio - y) <= 0.08 for y in ys)


def _drop_table_lines(
    lines: list[sections.Line],
    tables: list[tuple[int, float, tuple, list[list[str]]]],
) -> list[sections.Line]:
    """剔掉落在表格框内的文本行：表格已单独成段，正文里不该再有一份。"""
    if not tables:
        return lines
    boxes = [(page, bbox) for page, _y, bbox, _rows in tables]
    sign_pages = _signatory_pages(lines)
    kept: list[sections.Line] = []
    for ln in lines:
        if ln.bbox and _hit_any_box(ln, boxes) and not _keep_signatory_block(ln, sign_pages):
            continue
        kept.append(ln)
    return kept


def _hit_any_box(
    ln: sections.Line, boxes: list[tuple[int, tuple[float, float, float, float]]]
) -> bool:
    """行框与表格框的重叠面积占行面积一半以上，就算这行属于表格。"""
    x0, y0, x1, y1 = ln.bbox
    area = max((x1 - x0) * (y1 - y0), 1.0)
    for page, (bx0, by0, bx1, by1) in boxes:
        if page != ln.page:
            continue
        ox = max(0.0, min(x1, bx1) - max(x0, bx0))
        oy = max(0.0, min(y1, by1) - max(y0, by0))
        if ox * oy / area >= 0.5:
            return True
    return False
