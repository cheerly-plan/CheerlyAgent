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
    rows = _clean_table(raw)
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
