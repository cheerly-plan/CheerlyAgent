"""PDF 表格单元格清洗与表头对齐。

从 tables.py 二次拆出。PDF 抽出来的原始单元格满是排版噪声：数字被断行
拆碎("5176 63" 后跟一个孤立的小数点，实为 5176.63)、千分位逗号与小数点
被画成独立块甩到单元格末尾、中文之间夹着排版字距空格、多行表头需要压成
一行列名。

本模块只做一件事：把二维字符串数组洗净——逐格还原数字与标点，再把分组
表头对齐合并成单行列名。不涉及页面、bbox 和表题。
"""

from __future__ import annotations

import re

from ai_agent.kb.parsing import _escape_cell, normalize_choices


# ——— PDF 表格 ———
# PDF 里没有「表格」这个对象，只有画出来的线和摆好的字。pymupdf 的
# find_tables 靠线条（无框表格靠文字对齐）反推单元格，抽出来的是二维
# 字符串数组，转成 Markdown 表格交给切片层（整表成片、超长重复表头）。

_CELL_WS = re.compile(r"\s+")
# 单元格内的换行是排版断行：数字常被拆成 "5176 63\n."（小数点另起一行）
_NUM_DECIMAL = re.compile(r"(\d[\d,]*)\s+(\d+)\s*\.\s*(\d*)")
# 千分位也逃不过断行："114 376 427 06 , , ." 是 114,376,427.06
# （PDF 把逗号和小数点排到单元格末尾，前面只剩一串数字碎片）
_NUM_THOUSANDS = re.compile(r"^(\d+(?:\s+\d+)+)\s*(?:[,]\s*)*\.\s*(\d*)$")
# "2006年 1 6月 -" 是 2006年1-6月：连字符被甩到单元格最后
_NUM_RANGE = re.compile(r"(\d+)\s+(\d+月)\s*-\s*$")
# 中文字之间的空格也是排版字距（"木粉 复合油 ，"），不是真空格
_CJK = "\u4e00-\u9fff\u3040-\u30ff"
_CJK_SPACE = re.compile(f"([{_CJK}])\\s+([{_CJK}])")
_CJK_RE = re.compile(f"[{_CJK}]")
# 标点被单独排版时前后会多出空格："复合油 ，" / "（ 注"
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([，。、；：？！）】》」』%])")
_SPACE_AFTER_PUNCT = re.compile(r"([（【《「『])\s+")
# 表头续行（"元）"、"（%"）都很短，超过这个长度基本是数据行不是续行
_MAX_HEADER_CELL = 6
# 纯数字（允许千分位、百分比、负号）：表头碎片不会是这个形状
_NUM_ONLY = re.compile(r"^[\d.,%+\-]+$")
# 列名单元格的字数上限：超过基本都是解释性长句
_HEADER_CELL_MAX = 20
# 带这些标点的是句子，不是列名。刻意不含「、」「，」——「质押、标记或冻结
# 情况」这种真列名就用顿号，排掉它反而漏认
_HEADER_SENTENCE = re.compile(r"[。；！？]")
# 另一种断法：整数连写、小数点/逗号被画成独立块甩到单元格末尾
# 「401825 , .」= 4,018.25、「5358 .」= 53.58、「人民币100元 .」= 人民币1.00元
# （有小数点时末两位是小数，只有逗号时是纯千分位）
_NUM_TAIL_BOTH = re.compile(r"^([^\d]{0,8})(\d{2,})([^\d]{0,8})\s*,\s*\.\s*$")
_NUM_TAIL_DOT = re.compile(r"^([^\d]{0,8})(\d{2,})([^\d]{0,8})\s*\.\s*$")
_NUM_TAIL_COMMA = re.compile(r"^([^\d]{0,8})(\d{4,})([^\d]{0,8})\s*,\s*$")
# 逗号被单独排版时连成一片："总股本 ,， 比例" 只留一个中文逗号
_MULTI_COMMA = re.compile(r"\s*[，,](\s*[，,])+")
# 逗号、负号被画成独立块甩在数字后面（还常一起出现）：
# "471,784,966.36 ," → 471,784,966.36；"30,360,622.17 - ," → -30,360,622.17
_NUM_TRAIL_SIGNS = re.compile(r"^([\d.,]*\d)\s*(-)?\s*,?\s*$")
# 日期的连字符被单独排版："2016 12 31 - -" 是 2016-12-31
_DATE_DASH = re.compile(r"(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s*-\s*-")
# 商标有效期："20090807 . ." 是 2009.08.07（小数点被画成独立块）
_DATE_DOTS = re.compile(r"(\d{4})(\d{2})(\d{2})\s*\.\s*\.\s*")
# 表头日期月份不补零："2020630 . ." 是 2020.6.30（8 位的走上面，避免 20191231 被切成 2019.1.231）
_DATE_DOTS_MD = re.compile(r"(\d{4})(\d)(\d{2})\s*\.\s*\.\s*")
# 标点后的空格："总股本 ， 比例"
_SPACE_AFTER_PUNCT_2 = re.compile(r"([，。、；：？！%])\s+")
# 单位括号/百分号/数字后甩出来的孤立逗号：
# 「资产负债率（母公司%），」「1,031,675,913.97，」
_TRAIL_COMMA = re.compile(r"(?<=[）)%\d])\s*[，,]$")
# 千分位逗号被甩到后面，落在两个中文之间："换向器 , 项目"
# （只认英文逗号，中文逗号多半是真标点）
_CJK_COMMA = re.compile(f"(?<=[{_CJK}])\\s*,\\s*(?=[{_CJK}])")
# 合计行常把标签写在表头列上（和「项目」同一列居中），不能当成「表头位出现了真数据」
_TOTAL_LABEL = re.compile(r"合计|小计|总计")
# 整行只剩标点的「孤儿行」：千分位逗号、句读被 PDF 排成独立块插在多行
# 文本中间（", ，" / ". 。 ，"）。夹着英文点/逗号的是数字碎片，位置
# 不可考，删掉；纯中文标点（「， 、 ；」）能插回上一行的空格缝。
_PUNCT_ROW = re.compile(
    r"^[\s.,;:!？?%\-—–…()（）【】\[\]《》「」、，。；：\"'“”‘’]+$"
)
# 职责表这类长单元格：正文一行、标点单独一行。只含中文标点才敢回填，
# 英文逗号/点可能是千分位或小数点（见 _DOT_ROW / 发行概况表）。
_WRAP_PUNCT = re.compile(r"^[\s]*[，。、；：！？][\s，。、；：！？]*$")
_WRAP_PUNCT_MARK = re.compile(r"[，。、；：！？]")
# 只有一个点的孤儿行（前面可带千分位逗号）：是数字被甩出的小数点
_DOT_ROW = re.compile(r"^,?\s*\.$")
# 被甩出小数点的数字串：多段连写（"3 672 71"、"114 376 427 06"——末段是
# 小数，前面是整数部分），或单个整数（"365"、"200000"）。
# 前后不能挨着数字、点、千分位逗号，排除已格式化的「2,000.00」
_NUM_RUN = re.compile(r"(?<![\d.,])\d+(?:\s+\d{1,3})*(?![\d.,])")
# 键值表补头：内容列里这些形状算「表头样」——短且无中文标点，或是纯数字
_KV_PUNCT = re.compile(r"[（）【】《》「」、，。；：]")


def _tidy_header(row: list[str]) -> list[str]:
    """表头首格常和第一条数据粘在一起（「项目流动资产：」），还原成「项目」。"""
    return [re.sub(r"^项目[^：]{1,10}：$", "项目", cell) if cell else cell for cell in row]


def _is_data_row(row: list[str]) -> bool:
    """数据行：至少有一个纯数字单元格。"""
    return any(cell and _NUM_ONLY.match(cell) for cell in row)


def _is_header_row(row: list[str]) -> bool:
    """列名行：至少两个非空单元格、不含纯数字、带中文。"""
    cells = [cell for cell in row if cell]
    if len(cells) < 2 or _is_data_row(row):
        return False
    return any(_CJK_RE.search(cell) for cell in cells)


def _headerish_row(row: list[str]) -> bool:
    """更像列名行：每格都短、都不含数字和句读。

    比 `_is_header_row` 更严，用来在页内认「第二张表的列名」——那里没有
    重复列名可比，只能看形状。
    """
    cells = [cell for cell in row if cell]
    if len(cells) < 2 or _is_data_row(row):
        return False
    return all(
        len(cell) <= _HEADER_CELL_MAX and not _HEADER_SENTENCE.search(cell)
        for cell in cells
    )


def _clean_table(raw: list[list[str | None]]) -> list[list[list[str]]]:
    """单元格清洗 → 拆开拼在一起的表 → 表头归位 → 删空列。

    返回的是「一张图里可能拆出的多张表」：pymupdf 常把上下排的两个表
    识成一个（第二段的表头变成数据行），导致表头和数据对不上列。
    """
    rows = [[_clean_cell(cell) for cell in row] for row in raw]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    out: list[list[list[str]]] = []
    for part in _split_repeated_header(rows):
        # 先删空列：合并单元格常切出 6 列网格，表头在 1/4、数据在 0/3，
        # 去空后才变成 split_interleaved 认得的隔列交错
        part = _drop_empty_cols(part)
        part = _merge_header(part)
        part = _split_interleaved(part)
        part = _align_header(part)
        out.append(_add_kv_header(_fill_header_corner(_drop_empty_cols(part))))
    return out


def _fill_header_corner(rows: list[list[str]]) -> list[list[str]]:
    """表头左上角空着就补「项目」。

    合并单元格切出来的表头常是「空 | 本报告期末 | 上年度末」——Markdown
    首行即列名，空着的那格就成了无名列。
    """
    if len(rows) < 2 or len(rows[0]) < 3 or rows[0][0]:
        return rows
    filled = sum(1 for row in rows[1:] if row[0])
    if filled * 2 < len(rows) - 1:
        return rows
    return [["项目", *rows[0][1:]], *rows[1:]]


def _add_kv_header(rows: list[list[str]]) -> list[list[str]]:
    """两列键值表（"股票类型 | 人民币普通股（A 股）"）没有表头行。

    Markdown 表格首行必被渲染成表头（灰底），不补的话第一条发行概况就
    被顶成了列名。判据：每行都是「短标签 + 内容」，且没有一行的内容列
    长得像表头（短而无中文标点、或纯数字）——后者说明首行本来就是列名。
    """
    if len(rows) < 2 or any(len(r) != 2 for r in rows):
        return rows
    if not all(r[0] and r[1] and len(r[0]) <= 10 for r in rows):
        return rows
    if any(_kv_headerish(r[1]) for r in rows):
        return rows
    return [["项目", "内容"], *rows]


def _kv_headerish(cell: str) -> bool:
    """内容列是否长得像表头：「数值」「含义」这类一两个字，或纯数字。

    刻意只认 ≤2 字——「余额包销」这种 4 字短内容是真数据；代价是
    「解释说明」这类 4 字表头认不出来，但那比把数据行当表头少见得多。
    """
    return _NUM_ONLY.match(cell) is not None or (
        len(cell) <= 2 and not _KV_PUNCT.search(cell)
    )


def _split_repeated_header(rows: list[list[str]]) -> list[list[list[str]]]:
    """body 里又出现表头行 → 从那里断成两张表。

    两条判据任一成立即可，但都要求前面已有表体、前面那段表体多数行带
    数字（数据表才有；释义是整句）：

    - **重印列名**：非空列与首行完全一致，且有 ≥2 字同名的列名。这是
      「同一张表翻页后重印表头」；
    - **页内叠表**：这一行长得像列名（每格都短、无数字、无句读），且紧
      挨着的上一行是数据行。这是「上下排的两张表被识成一张」，第二张的
      列名接在第一张的数据后面，列占用常与首行不同（合并单元格切出的
      空列位置不一样），所以不能要求列占用一致。

    释义表（"X 指 Y"）每条的行形状都一样，只靠「重印列名」会被逐条切开，
    一张表碎成十几张两行小表——所以那条还要求 `_same_head_label`。
    """
    if len(rows) < 3:
        return [rows]
    head_cols = {i for i, cell in enumerate(rows[0]) if cell}
    parts: list[list[list[str]]] = []
    cur: list[list[str]] = [rows[0]]
    for row in rows[1:]:
        cols = {i for i, cell in enumerate(row) if cell}
        numeric = any(cell and _NUM_ONLY.match(cell) for cell in row)
        reprinted = cols == head_cols and _same_head_label(rows[0], row)
        stacked = _headerish_row(row) and any(_is_data_row(r) for r in cur[-2:])
        if (reprinted or stacked) and len(cur) > 1 and not numeric and _mostly_numeric(cur[1:]):
            parts.append(cur)
            cur = [row]
        else:
            cur.append(row)
    parts.append(cur)
    return parts


def _same_head_label(head: list[str], row: list[str]) -> bool:
    """重复出现的表头得跟首行有共同列名（≥2 字，排掉「指」这类连接词）。"""
    pairs = zip(head, row)
    return any(len(a) >= 2 and a == b for a, b in pairs)


def _mostly_numeric(body: list[list[str]]) -> bool:
    """表体多数行带数字才像数据表——用来把数据表和整句的释义区分开。"""
    if not body:
        return False
    hits = sum(1 for row in body if any(re.search(r"\d", cell) for cell in row))
    return hits * 2 >= len(body)


def _fill_wrap_punct(text: str, marks: list[str]) -> str:
    """把单独成行的中文标点填进上一行短语之间的空格，多出来的贴行尾。

    「负责行业研究 收集…投资 合作机会」+「， 、 ，」→
    「负责行业研究，收集…投资、合作机会」。空格比标点多时，剩下的
    空格留给后面的 _CJK_SPACE 去拼（行尾被切断的「新项」「目」）。
    """
    if not marks:
        return text
    parts = re.split(r"(\s+)", text)
    i = 0
    for k, part in enumerate(parts):
        if part.isspace() and i < len(marks):
            parts[k] = marks[i]
            i += 1
    return "".join(parts) + "".join(marks[i:])


def _clean_cell(value: str | None) -> str:
    text = value or ""
    if "\n" in text:
        # 多行单元格里的孤儿标点行分三类（都是 PDF 把标点排成独立块）：
        # - 只有点的「. / , .」：是前面数字被甩出的小数点（可带千分位逗号），
        #   按顺序补回数字里——「365元/股\n.\n…」是 3.65、「200000万元」
        #   是 2,000.00；
        # - 纯中文标点（「， 、 ；」）：插回上一行空格缝，职责表靠这个还原；
        # - 中英文混杂（", ，" / ". 。 ，"）：数字碎片和句读搅在一起，
        #   位置不可考，删掉。末行句号仍保留。
        dots = 0
        kept: list[str] = []
        lines = text.split("\n")
        for i, ln in enumerate(lines):
            if _DOT_ROW.match(ln.strip()):
                dots += 1
            elif _WRAP_PUNCT.match(ln) and kept:
                kept[-1] = _fill_wrap_punct(
                    kept[-1], _WRAP_PUNCT_MARK.findall(ln)
                )
            elif _PUNCT_ROW.match(ln) and i < len(lines) - 1:
                continue
            else:
                kept.append(ln)
        text = "\n".join(kept)
        if dots:
            text = _restore_dots(text, dots)
    text = text.replace("\n", " ")
    text = _CELL_WS.sub(" ", text).strip()
    # 先修连写的日期（"20090807 . ."），否则会被下面的数字重组当成
    # 「整数连写 + 甩到末尾的小数点」切坏
    text = _DATE_DOTS.sub(r"\1.\2.\3", text)
    text = _DATE_DOTS_MD.sub(r"\1.\2.\3", text)
    text = _regroup_tail(text)
    m = _NUM_THOUSANDS.match(text)
    if m:
        groups = m.group(1).split()
        text = ",".join(groups[:-1]) + "." + groups[-1] + m.group(2)
    else:
        text = _NUM_DECIMAL.sub(
            lambda m: f"{m.group(1)}.{m.group(2)}{m.group(3)}", text
        )
        text = _NUM_RANGE.sub(r"\1-\2", text)
    text = _CJK_SPACE.sub(r"\1\2", text)
    # 数字与中文之间也是排版间距："2005年12月31 日" → "2005年12月31日"
    text = re.sub(rf"([{_CJK}])\s+(\d)", r"\1\2", text)
    text = re.sub(rf"(\d)\s+([{_CJK}])", r"\1\2", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_PUNCT.sub(r"\1", text)
    text = _MULTI_COMMA.sub("，", text)
    text = _DATE_DASH.sub(r"\1-\2-\3", text)
    # 日期区间里的空格："2009.08.07- 2019.08.06"
    text = re.sub(r"(\d)\s*-\s+(\d)", r"\1-\2", text)
    m = _NUM_TRAIL_SIGNS.match(text)
    if m:
        text = ("-" if m.group(2) else "") + m.group(1)
    text = _SPACE_AFTER_PUNCT_2.sub(r"\1", text)
    text = _TRAIL_COMMA.sub("", text)
    text = _CJK_COMMA.sub("", text)
    # 表格里也有勾选（「是否 √是 □否」），和正文同一个符号字体问题
    return _escape_cell(normalize_choices(text))


def _regroup_tail(text: str) -> str:
    """把甩到单元格末尾的小数点/逗号拼回数字里。"""
    for pattern, decimal in (
        (_NUM_TAIL_BOTH, True),
        (_NUM_TAIL_DOT, True),
        (_NUM_TAIL_COMMA, False),
    ):
        m = pattern.match(text)
        if not m:
            continue
        prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
        if decimal:
            if len(digits) > 2:
                return f"{prefix}{_thousands(digits[:-2])}.{digits[-2:]}{suffix}"
            return f"{prefix}0.{digits}{suffix}"
        return f"{prefix}{_thousands(digits)}{suffix}"
    return text


def _thousands(digits: str) -> str:
    """4018 → 4,018（三位一组，从右往左）"""
    if len(digits) <= 3:
        return digits
    out: list[str] = []
    while len(digits) > 3:
        out.insert(0, digits[-3:])
        digits = digits[:-3]
    out.insert(0, digits)
    return ",".join(out)


def _restore_dots(text: str, count: int) -> str:
    """把甩成孤儿行的小数点按出现顺序补回数字（招股书惯例：两位小数）。

    「365元/股」→ 3.65 元/股、「200000万元」→ 2,000.00 万元；多段连写
    （"5176 63"）末段是小数 → 5,176.63 同 _NUM_THOUSANDS 的拼法。只动前
    count 处：日期（2016年12月31日）排在数字主体后面，不会误伤。
    """
    remaining = count

    def sub(m: re.Match[str]) -> str:
        nonlocal remaining
        parts = m.group(0).split()
        if len(parts) == 1 and len(parts[0]) < 2:
            return m.group(0)  # 单个一位数字不可能是「整数+甩点」，不消耗
        if remaining <= 0:
            return m.group(0)
        remaining -= 1
        if len(parts) > 1:
            return ",".join(parts[:-1]) + "." + parts[-1]
        digits = parts[0]
        if len(digits) > 2:
            return f"{_thousands(digits[:-2])}.{digits[-2:]}"
        return f"0.{digits}"

    return _NUM_RUN.sub(sub, text)


def _align_header(rows: list[list[str]]) -> list[list[str]]:
    """表头归位：合并单元格常把表头挤到右边一列，和数据对不上。

    「销量（吨）」的格子横跨两列时，网格切出来的表头落在 col1，而数值
    落在 col1/col2 —— 直接渲染的话每列都串了位。这里按「有数据的列」
    依次安放表头文字：数量对得上才动，对不上说明不是这种表，不猜。
    """
    if len(rows) < 2:
        return rows
    header, body = rows[0], rows[1:]
    data_cols = [i for i in range(len(header)) if any(row[i] for row in body)]
    fills = [cell for cell in header if cell]
    if not fills or len(fills) != len(data_cols):
        return rows
    aligned = [""] * len(header)
    for col, cell in zip(data_cols, fills):
        aligned[col] = cell
    return [aligned] + body


def _merge_header(rows: list[list[str]]) -> list[list[str]]:
    """表头占了多行时拼回一行，最多吃三行（表头再复杂就认命）。

    三种形态，判据是「第二行的非空列和第一行重不重叠」：

    - **续行**：第二行的列是第一行的子集，且都是短碎片——"销售收入（万"
      + "元）"、"占总收入比例" + "（%"）。长内容说明那是数据行，不动；
    - **分散**：两行的非空列完全不重叠——招股书常见的斜线表头把「年份」
      放第 0 列、「2006年1-6月」放第 1~4 列，分了两三行才写完。
    - **分组**：第一行是合并单元格的分组名（「中国银行集团」「中国银行」），
      第二行填期间 / 审计状态。两行部分重叠，按分组名向右沿用再拼到每一列。

    部分重叠但第二行已经是数字，一律不动——宁可留着多行表头，
    也不能把两行数据粘成一行。
    """
    for _ in range(3):
        if len(rows) < 2:
            break
        first, second = rows[0], rows[1]
        f_cols = {i for i, cell in enumerate(first) if cell}
        s_cols = {i for i, cell in enumerate(second) if cell}
        if not f_cols or not s_cols:
            break
        if s_cols <= f_cols:
            cells = [second[i] for i in s_cols]
            if any(len(cell) > _MAX_HEADER_CELL for cell in cells):
                break
            # 纯数字的一律是数据行（"100"、"200"），拼进表头就串味了
            if any(_NUM_ONLY.match(cell) for cell in cells):
                break
            merged = _concat_header_rows(first, second)
        elif s_cols & f_cols:
            merged = _flatten_group_header(first, second)
            if merged is None:
                break
        elif _alternating(f_cols, s_cols):
            # 两行的非空列均匀交替（表头在 1/4/7 列、数据在 0/3/6 列）不是
            # 分两行写的表头，而是合并单元格把表头和数据挤进了同一套网格。
            # 合并起来就成了「1 | 序号 | 孙瑞良 | 股东名称」这种串味的行，
            # 交给后面的 _align_header 按数据列归位
            break
        else:
            merged = _concat_header_rows(first, second)
        rows = [merged] + rows[2:]
    return rows


def _concat_header_rows(first: list[str], second: list[str]) -> list[str]:
    merged = list(first)
    for i, cell in enumerate(second):
        if not cell:
            continue
        merged[i] = merged[i] + cell if merged[i] else cell
    return merged


def _flatten_group_header(first: list[str], second: list[str]) -> list[str] | None:
    """把「分组名跨列 + 下面一行期间」摊成每列一个完整列名。

    只在第二行像列名、不像数据时动手。股东名册那种表头/数据交错
    走的是另一条路，不会进这里（非空列互不重叠）。
    """
    if _is_data_row(second) or not _headerish_row(second):
        return None
    filled = _carry_span_labels(first)
    return [_join_header_cell(a, b) for a, b in zip(filled, second)]


def _carry_span_labels(row: list[str]) -> list[str]:
    """合并单元格留下的空列，用左侧分组名填上。左上角「项目」不往右传。"""
    out = list(row)
    corner = out[0] if out else ""
    last = ""
    for i, cell in enumerate(out):
        if cell:
            last = cell
            continue
        if i > 0 and last and last != corner:
            out[i] = last
    return out


def _join_header_cell(top: str, bottom: str) -> str:
    if not bottom:
        return top
    if not top or bottom.startswith(top) or top in bottom:
        return bottom
    if top.startswith(bottom):
        return top
    return top + bottom


def _split_interleaved(rows: list[list[str]]) -> list[list[str]]:
    """表头被并进数据行：拆回「一行表头 + 纯数据行」。

    跨页续表、合并单元格常抽成交错列 [数据1, 表头1, 数据2, 表头2 …]：
    表头只在首行出现、后续行对应位置留空，直接渲染就是「1 | 序号 | 孙瑞良
    | 股东名称」这种串味的行。

    判据刻意很严，宁可不动也不能把真的宽表拆坏：列数成对；表头位有中文
    且不是纯数字；其余行在表头位要么空、要么与首行一致（不一致说明那是
    真数据列）。合计行例外：标签常写在表头列上，不能因此放弃拆分。
    """
    if len(rows) < 2:
        return rows
    width = len(rows[0])
    if width < 4 or width % 2:
        return rows
    for start in (1, 0):
        head = rows[0][start::2]
        if sum(1 for cell in head if cell) < 2:
            continue
        if not any(_CJK_RE.search(cell) for cell in head):
            continue
        if any(cell and _NUM_ONLY.match(cell) for cell in head):
            continue
        if any(
            not _header_slot_ok(row[start::2], head) for row in rows[1:]
        ):
            continue
        data_start = 1 - start
        body = [_data_slots(row, data_start, start) for row in rows[1:]]
        if not any(any(row) for row in body):
            continue
        return _drop_empty_cols([list(head)] + body)
    return rows


def _header_slot_ok(slots: list[str], head: list[str]) -> bool:
    """表头位要么空/与首行相同，要么是合计行把标签写在了表头列。"""
    if _TOTAL_LABEL.search("".join(slots)):
        return True
    return all(not cell or cell == h for cell, h in zip(slots, head))


def _data_slots(row: list[str], data_start: int, head_start: int) -> list[str]:
    data = list(row[data_start::2])
    if any(data):
        return data
    heads = list(row[head_start::2])
    return heads if any(heads) else data


def _alternating(cols_a: set[int], cols_b: set[int]) -> bool:
    """两组列号是否均匀交替（a 在 1/4/7，b 在 0/3/6）。"""
    # 两列表被切成 4 列交错时各只有 2 个非空列，3 会漏掉（南岭发行费用表）
    if len(cols_a) < 2 or len(cols_b) < 2:
        return False
    seq = sorted(cols_a | cols_b)
    if len(seq) != len(cols_a) + len(cols_b):
        return False
    flags = [i in cols_a for i in seq]
    return all(x != y for x, y in zip(flags, flags[1:]))


def _drop_empty_cols(rows: list[list[str]]) -> list[list[str]]:
    """删掉整列为空的列：跨列合并的单元格会留下一串空列。"""
    width = max(len(row) for row in rows)
    keep = [i for i in range(width) if any(row[i] for row in rows)]
    return [[row[i] for i in keep] for row in rows]
