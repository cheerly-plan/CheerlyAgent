"""PDF 表格抽取的单测。

PDF 里没有「表格」对象，pymupdf 反推出来的单元格带着一堆排版痕迹：
断行、合并单元格留下的空列、被甩到末尾的千分位。这些还原规则的用例
全部取自招股书实录（南岭民爆第 4/14 页），改坏了要能立刻发现。
"""

from ai_agent.kb.parsing import (
    _align_header,
    _clean_cell,
    _clean_table,
    _drop_empty_cols,
    _drop_table_lines,
    _merge_header,
    _rows_to_markdown,
    _split_repeated_header,
)
from ai_agent.kb.sections import Line

BODY = 12.0


def line(text: str, page: int = 1, y: float = 0.3, bbox=None) -> Line:
    return Line(text, page, BODY, y, bbox)


def test_clean_cell_joins_broken_decimal():
    # 小数点被排到单元格的下一视觉行
    assert _clean_cell("5176 63\n.") == "5176.63"
    assert _clean_cell("47 60\n.") == "47.60"


def test_clean_cell_rebuilds_thousands_separator():
    # 千分位逗号和小数点都被甩到末尾
    assert _clean_cell("114 376 427 06\n,\n,\n.") == "114,376,427.06"
    assert _clean_cell("3 672 71\n,\n.") == "3,672.71"


def test_clean_cell_restores_month_range():
    assert _clean_cell("2006年\n1 6月\n-") == "2006年1-6月"


def test_clean_cell_drops_cjk_typography_space():
    # 中文字之间的空格是字距不是真空格；数字与中文之间同理
    assert _clean_cell("木粉\n复合油\n，") == "木粉复合油，"
    assert _clean_cell("2005年12月31 日") == "2005年12月31日"


def test_clean_cell_escapes_pipe():
    assert _clean_cell("a|b") == "a\\|b"


def test_merge_header_continuation_row():
    # 「销售收入（万」+「元）」：续行且都是短碎片
    rows = [
        ["项目", "销量（吨）", "", "销售收入（万", "", "占总收入比例"],
        ["", "", "", "元）", "", "（%）"],
        ["2005年", "18959", "8897.28", "", "46.43", ""],
    ]
    merged = _merge_header(rows)
    assert merged[0][3] == "销售收入（万元）"
    assert merged[0][5] == "占总收入比例（%）"
    # 数据行不能被吃掉
    assert merged[1][0] == "2005年"


def test_merge_header_leaves_data_rows_alone():
    # 释义表：第二行是完整数据（有长单元格），不能并进表头
    rows = [
        ["南岭化工厂", "指", "湖南省南岭化工厂"],
        ["股东大会", "指", "湖南南岭民用爆破器材股份有限公司股东大会"],
    ]
    assert _merge_header(rows) == rows


def test_merge_header_combines_split_header():
    # 斜线表头：年份与年份值分散在两行，非空列互不重叠
    rows = [
        ["年份", "", "", "", "2003年"],
        ["", "2006年1-6月", "2005年", "2004年", ""],
        ["主营业务收入", "100", "200", "300", "400"],
    ]
    merged = _merge_header(rows)
    assert merged[0] == ["年份", "2006年1-6月", "2005年", "2004年", "2003年"]
    assert merged[1][0] == "主营业务收入"


def test_drop_empty_cols_removes_merged_cells():
    rows = [["项目", "销量", ""], ["2005年", "18959", ""]]
    assert _drop_empty_cols(rows) == [["项目", "销量"], ["2005年", "18959"]]


def test_align_header_fixes_off_by_one():
    # 合并单元格让表头右移一列：表头数 == 有数据的列数时按数据列归位
    rows = [
        ["项目", "销量（吨）", "", "销售收入（万元）"],
        ["2005年", "18959", "8897.28", ""],
    ]
    aligned = _align_header(rows)
    assert aligned[0] == ["项目", "销量（吨）", "销售收入（万元）", ""]
    assert aligned[1] == ["2005年", "18959", "8897.28", ""]


def test_align_header_skips_when_counts_mismatch():
    rows = [["项目", "销量"], ["2005年", "18959", "多出一列"]]
    assert _align_header(rows) == rows


def test_clean_table_end_to_end():
    raw = [
        ["项目", "销量（吨）", "", "销售收入（万", "", "", "占总收入比例", ""],
        ["", "", "", "元）", "", "", "（%）", ""],
        ["2006年\n1 6月\n-", "9841", "5176 63\n.", "", "", "47 60\n.", ""],
    ]
    rows = _clean_table(raw)[0]
    assert rows[0] == ["项目", "销量（吨）", "销售收入（万元）", "占总收入比例（%）"]
    assert rows[1][:3] == ["2006年1-6月", "9841", "5176.63"]


def test_rows_to_markdown_shape():
    md = _rows_to_markdown([["a", "b"], ["1", "2"]])
    assert md.splitlines()[0] == "| a | b |"
    assert md.splitlines()[1] == "|---|---|"
    assert md.splitlines()[2] == "| 1 | 2 |"


def test_drop_table_lines_removes_rows_inside_table():
    # 表格框覆盖 y=100~200，正文行在框外
    tables = [(1, 100.0, (0.0, 100.0, 500.0, 200.0), [["a", "b"]])]
    lines = [
        line("表格内的字", bbox=(10.0, 120.0, 200.0, 130.0)),
        line("表格外的正文", bbox=(10.0, 300.0, 200.0, 310.0)),
    ]
    kept = _drop_table_lines(lines, tables)
    assert [ln.text for ln in kept] == ["表格外的正文"]


def test_drop_table_lines_keeps_partial_overlap():
    # 只压住一点点（<50%）的行属于正文，不能误删
    tables = [(1, 100.0, (0.0, 100.0, 500.0, 200.0), [["a", "b"]])]
    lines = [line("半截在表里", bbox=(10.0, 190.0, 200.0, 300.0))]
    assert len(_drop_table_lines(lines, tables)) == 1


def test_drop_table_lines_without_tables():
    lines = [line("正文")]
    assert _drop_table_lines(lines, []) == lines


# ——— 华瑞电器实录（股东表 / 现金流量表 / 财务指标表）———

def test_clean_cell_regroups_runon_number():
    # 整数连写、小数点被画成独立块甩到末尾
    assert _clean_cell("401825 , .") == "4,018.25"
    assert _clean_cell("5358 .") == "53.58"
    assert _clean_cell("115375 , .") == "1,153.75"
    # 单位夹在数字和小数点之间
    assert _clean_cell("人民币100元 .") == "人民币1.00元"


def test_clean_cell_regroups_tail_comma():
    # 已经是千分位了，末尾还多画了一个逗号
    assert _clean_cell("471,784,966.36 ,") == "471,784,966.36"
    assert _clean_cell("10000 ,") == "10,000"


def test_clean_cell_moves_trailing_minus():
    # 负号被画成独立块，落在数字（和多余逗号）后面
    assert _clean_cell("30,360,622.17 - ,") == "-30,360,622.17"
    assert _clean_cell("100 -") == "-100"


def test_clean_cell_restores_date_with_dashes():
    assert _clean_cell("2016 12 31 - -") == "2016-12-31"
    assert _clean_cell("2015 12 31 - -") == "2015-12-31"


def test_clean_cell_collapses_dup_comma():
    assert _clean_cell("总股本 ,， 比例") == "总股本，比例"


def test_merge_header_skips_interleaved_rows():
    """表头列(1,4,7,10) 与数据列(0,3,6,9) 交替时不能当成「分散表头」合并。

    合并出来就是「1 | 序号 | 孙瑞良 | 股东名称」这种串味的行。合并单元格
    常把表头和数据挤进同一套网格，此时应保持两行，交给 _align_header 归位。
    """
    rows = [
        ["", "序号", "", "股东名称", "", "持股数量（万股）", ""],
        ["1", "", "孙瑞良", "", "401825", ""],
    ]
    merged = _merge_header(rows)
    assert merged == rows


def test_clean_table_realigns_interleaved_header():
    """华瑞第 50 页股东表：12 列网格里表头与数据交错，归位后 4 列。"""
    raw = [
        ["", "序号", "", "", "股东名称", "", "", "持股数量（万股）", ""],
        ["1", None, None, "孙瑞良", None, None, "401825\n, .", None, None],
        ["2", None, None, "联创投资", None, None, "115375\n, .", None, None],
    ]
    rows = _clean_table(raw)[0]
    assert rows[0] == ["序号", "股东名称", "持股数量（万股）"]
    assert rows[1] == ["1", "孙瑞良", "4,018.25"]
    assert rows[2] == ["2", "联创投资", "1,153.75"]


def test_clean_cell_drops_stray_comma_between_cjk():
    # 千分位逗号被甩到两个中文之间（"7,500万只换向器项目"）
    assert _clean_cell("换向器 , 项目") == "换向器项目"


def test_clean_cell_keeps_real_comma_between_cjk():
    # 中文逗号是真标点，不能删
    assert _clean_cell("总股本 ,， 比例") == "总股本，比例"


def test_split_repeated_header_separates_stacked_tables():
    """华瑞第 30 页：两个表被识别成一个，第二段表头要断开。"""
    rows = [
        ["", "项目", "", "", "2016-12-31", ""],
        ["资产负债率", "", "", "59.07", "", ""],
        ["流动比率", "", "", "0.96", "", ""],
        ["", "项目", "", "", "2016年度", ""],
        ["应收账款周转率", "", "", "3.88", "", ""],
    ]
    parts = _split_repeated_header(rows)
    assert len(parts) == 2
    assert parts[0][0] == rows[0]
    assert parts[1] == [rows[3], rows[4]]


def test_clean_table_returns_one_part_for_plain_table():
    assert len(_clean_table([["a", "b"], ["1", "2"]])) == 1


def test_clean_cell_restores_dotted_date():
    # 商标有效期："20090807 . ." 是 2009.08.07
    assert _clean_cell("20090807 . . - 20190806 . .") == "2009.08.07-2019.08.06"
