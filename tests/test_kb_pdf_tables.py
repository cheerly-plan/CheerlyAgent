"""PDF 表格抽取的单测。

PDF 里没有「表格」对象，pymupdf 反推出来的单元格带着一堆排版痕迹：
断行、合并单元格留下的空列、被甩到末尾的千分位。这些还原规则的用例
全部取自招股书实录（南岭民爆第 4/14 页），改坏了要能立刻发现。
"""

from ai_agent.kb.parsing import (
    _align_header,
    _attach_captions,
    _caption_above,
    _carry_page_headers,
    _clean_cell,
    _clean_table,
    _drop_empty_cols,
    _drop_table_lines,
    _fill_header_corner,
    _find_page_tables,
    _header_has_cjk,
    _merge_header,
    _glossary_rows,
    _pdf_glossary_tables,
    _rows_to_markdown,
    _split_repeated_header,
    _tidy_header,
    _word_lines,
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


def test_split_repeated_header_accepts_new_header_with_other_column_layout():
    """三季度报第 2 页：EPS 表下面直接接「资产负债表摘要」。

    第二段的列名跟首行对不上（「本报告期末」vs「本报告期」，空列位置
    也不同），只能靠「长得像列名 + 上一行是数据」认出来。
    """
    rows = [
        ["项目", "本报告期", "增减(%)", "年初至报告期末", "增减(%)"],
        ["营业收入", "100,924,954", "-9.45", "309,237,206", "-4.42"],
        ["利润总额", "1,823,608", "-6.22", "7,633,400", "-10.27"],
        ["", "本报告期末", "上年度末", "", "增减幅度(%)"],
        ["总资产", "993,101,072", "941,597,382", "", "5.47"],
    ]
    parts = _split_repeated_header(rows)
    assert len(parts) == 2
    assert parts[0][1:] == rows[1:3]
    assert parts[1] == rows[3:5]


def test_split_repeated_header_keeps_plain_data_table_intact():
    # 每行都是数据，没有列名形状的行：不许切
    rows = [
        ["项目", "本期", "上期"],
        ["营业收入", "100", "90"],
        ["营业成本", "60", "55"],
        ["毛利率", "40.00%", "38.89%"],
    ]
    assert len(_split_repeated_header(rows)) == 1


def test_split_repeated_header_keeps_sentence_rows_in_one_table():
    # 释义表：行的形状一样但没有数字，不该被逐条切开
    rows = [
        ["术语", "指"],
        ["发行人", "指本公司的全称"],
        ["证监会", "指中国证券监督管理委员会"],
        ["交易所", "指上海证券交易所"],
    ]
    assert len(_split_repeated_header(rows)) == 1


def test_carry_page_headers_prepends_header_to_continuation():
    header = ["项目", "本报告期", "年初至报告期末"]
    prev = (1, 577.0, (0.0, 577.0, 500.0, 800.0), [header, ["营业收入", "100", "309"]])
    cont = (2, 72.0, (0.0, 72.0, 500.0, 300.0), [["基本每股收益", "0.01", "0.06"]])
    out = _carry_page_headers([prev, cont], {1: 842.0, 2: 842.0})
    assert out[1][3][0] == header
    assert out[1][3][1] == ["基本每股收益", "0.01", "0.06"]


def test_carry_page_headers_skips_when_previous_table_ended_midpage():
    # 上一页的表在页中结束（528 < 842-90），说明它没被切断，下一张不是它的续表
    header = ["项目", "本报告期", "年初至报告期末"]
    prev = (1, 300.0, (0.0, 300.0, 500.0, 528.0), [header, ["a", "1", "2"]])
    cont = (2, 72.0, (0.0, 72.0, 500.0, 300.0), [["b", "3", "4"]])
    out = _carry_page_headers([prev, cont], {1: 842.0, 2: 842.0})
    assert out[1][3] == [["b", "3", "4"]]


def test_carry_page_headers_skips_width_mismatch():
    header = ["项目", "本报告期", "年初至报告期末"]
    prev = (1, 577.0, (0.0, 577.0, 500.0, 800.0), [header, ["a", "1", "2"]])
    cont = (2, 72.0, (0.0, 72.0, 500.0, 300.0), [["b", "3", "4", "5"]])
    out = _carry_page_headers([prev, cont], {1: 842.0, 2: 842.0})
    assert len(out[1][3]) == 1


def test_carry_page_headers_skips_table_starting_midpage():
    header = ["项目", "本报告期", "年初至报告期末"]
    prev = (1, 577.0, (0.0, 577.0, 500.0, 800.0), [header, ["a", "1", "2"]])
    cont = (1, 500.0, (0.0, 500.0, 500.0, 700.0), [["b", "3", "4"]])
    out = _carry_page_headers([prev, cont], {1: 842.0})
    assert out[1][3] == [["b", "3", "4"]]


def test_carry_page_headers_never_borrows_a_data_row():
    # 上一张表自己就没表头（首行是数据），拿它当列名只会把错的东西固化
    prev = (1, 577.0, (0.0, 577.0, 500.0, 800.0), [["营业收入", "100", "309"]])
    cont = (2, 72.0, (0.0, 72.0, 500.0, 300.0), [["总资产", "1", "2"]])
    out = _carry_page_headers([prev, cont], {1: 842.0, 2: 842.0})
    assert out[1][3] == [["总资产", "1", "2"]]


def test_tidy_header_restores_item_label_glued_to_first_row():
    assert _tidy_header(["项目流动资产："]) == ["项目"]
    assert _tidy_header(["项目", "期末余额"]) == ["项目", "期末余额"]


def test_fill_header_corner_names_the_empty_corner():
    rows = [["", "本报告期末", "上年度末"], ["总资产", "993", "941"], ["负债", "500", "480"]]
    assert _fill_header_corner(rows)[0][0] == "项目"


def test_fill_header_corner_skips_when_first_column_is_empty():
    rows = [["", "期末", "期初"], ["", "993", "941"]]
    assert _fill_header_corner(rows)[0][0] == ""


def test_clean_table_returns_one_part_for_plain_table():
    assert len(_clean_table([["a", "b"], ["1", "2"]])) == 1


def test_clean_cell_restores_dotted_date():
    # 商标有效期："20090807 . ." 是 2009.08.07
    assert _clean_cell("20090807 . . - 20190806 . .") == "2009.08.07-2019.08.06"


def test_clean_table_nanling_issue_fee_two_columns():
    """南岭第 17 页发行费用表：有框两列，pymupdf 切成 6 列交错，合计写在表头列。

    修之前会把首行数据并进表头，渲染成「保荐和承销费用 | 项目 | 860 | 费用金额」。
    """
    raw = [
        ["", "项目", "", "", "费用金额（万元）", ""],
        ["保荐和承销费用", None, None, "860", None, None],
        ["审计费用", None, None, "148", None, None],
        ["律师费用", None, None, "80", None, None],
        ["发行手续费", None, None, "185", None, None],
        ["登记托管费", None, None, "6", None, None],
        ["", "合计", "", "", "1279", ""],
    ]
    rows = _clean_table(raw)[0]
    assert rows[0] == ["项目", "费用金额（万元）"]
    assert rows[1] == ["保荐和承销费用", "860"]
    assert rows[2] == ["审计费用", "148"]
    assert rows[-1] == ["合计", "1279"]
    assert _rows_to_markdown(rows).startswith("| 项目 | 费用金额（万元） |")


def test_clean_cell_reattaches_wrapped_cjk_punct():
    """华瑞第 59 页职能部门表：长职责被抽成「正文行 + 标点行」。

    修之前会把「， ； ；」硌在句子中间，渲染成「经营计划,;; 负责市场调研」。
    """
    raw = (
        "负责行业研究 收集与公司发展战略匹配的投资 合作机会 对企业投资\n"
        "， 、 ， 、\n"
        "资本运作等项目进行研究论证 提供决策支持 制订公司战略及经营计划\n"
        "， ； ；\n"
        "负责市场调研 分析市场数据为公司战略制定提供参考\n"
        "， ；"
    )
    got = _clean_cell(raw)
    assert "负责行业研究，收集与公司发展战略匹配的投资、合作机会，对企业投资、" in got
    assert "进行研究论证，提供决策支持；制订公司战略及经营计划；" in got
    assert "，；；" not in got
    assert ",;" not in got


def test_clean_cell_wrap_punct_joins_split_word():
    # 行尾被切断的「新项 / 目」：标点填进空格后，剩下的空格由字距规则拼回
    raw = (
        "制定公司技术研发计划并有效实施 工装 模具 夹具设计及管理 新项\n"
        "； 、 、 ；\n"
        "目市场调研与可行性分析 提出并完成公司革新项目 建立公司技术开发\n"
        "； ；\n"
        "信息系统 公司技术管理制度的制定 负责重大质量事故处理\n"
        "； ； 。"
    )
    assert _clean_cell(raw) == (
        "制定公司技术研发计划并有效实施；工装、模具、夹具设计及管理；"
        "新项目市场调研与可行性分析；提出并完成公司革新项目；"
        "建立公司技术开发信息系统；公司技术管理制度的制定；"
        "负责重大质量事故处理。"
    )


def test_clean_table_huarui_department_duties():
    """华瑞第 59 页：6 列交错网格 + 长职责单元格。

    表头应是「部门 | 职责」，不能把「证券部」并进表头再当成每一片的列名。
    """
    raw = [
        ["", "部 门", "", "", "职 责", ""],
        [
            "证券部",
            None,
            None,
            "负责行业研究 收集与公司发展战略匹配的投资 合作机会 对企业投资\n"
            "， 、 ， 、\n"
            "资本运作等项目进行研究论证 提供决策支持 制订公司战略及经营计划\n"
            "， ； ；",
            None,
            None,
        ],
        [
            "技术研发中心",
            None,
            None,
            "制定公司技术研发计划并有效实施 工装 模具 夹具设计及管理 新项\n"
            "； 、 、 ；\n"
            "目市场调研与可行性分析 提出并完成公司革新项目\n"
            "； ；",
            None,
            None,
        ],
    ]
    rows = _clean_table(raw)[0]
    assert rows[0] == ["部门", "职责"]
    assert rows[1][0] == "证券部"
    assert rows[1][1].startswith("负责行业研究，收集与公司发展战略匹配的投资、合作机会")
    assert "，；；" not in rows[1][1]
    assert rows[2][0] == "技术研发中心"
    assert "工装、模具、夹具设计及管理；新项目市场调研" in rows[2][1]
    assert _rows_to_markdown(rows).startswith("| 部门 | 职责 |")


def test_clean_cell_restores_unpadded_month_date():
    # 三线表表头：「2020.6.30」被抽成「2020630 . .」；8 位日期仍走原规则
    assert _clean_cell("2020630\n. .") == "2020.6.30"
    assert _clean_cell("20191231\n. .") == "2019.12.31"
    assert _clean_cell("20171231 . .") == "2017.12.31"


def test_clean_cell_strips_trailing_comma_after_money():
    # 千分位逗号被甩到末尾，重组后还剩一个中文逗号
    assert _clean_cell("103167591397\n, , , .") == "1,031,675,913.97"
    assert _clean_cell("78632704\n, .") == "786,327.04"


def test_header_has_cjk_detects_clipped_balance_sheet():
    assert not _header_has_cjk([["2020630\n. .", "20191231\n. .", "20181231\n. ."]])
    assert _header_has_cjk([["资产", "2020.6.30", "2019.12.31"]])
    assert _header_has_cjk([["", "项目", "", "费用金额（万元）", ""]])


def test_clean_table_quecheng_balance_sheet_keeps_labels():
    """确成硅第 43 页三线表：lines_strict 抽到的 5 列（含「资产」和 2017）。"""
    raw = [
        ["资产", "2020630\n. .", "20191231\n. .", "20181231\n. .", "20171231\n. ."],
        [
            "流动资产",
            "103167591397\n, , , .",
            "111323056636\n, , , .",
            "103241301002\n, , , .",
            "109540283099\n, , , .",
        ],
        ["非流动资产", "95930810363\n, , .", "90003145273\n, , .", "75717845598\n, , .", "55157074381\n, , ."],
        [
            "归属于母公\n司所有者权\n益合计",
            "168787665480\n, , , .",
            "167511263744\n, , , .",
            "147168025976\n, , , .",
            "124701963902\n, , , .",
        ],
    ]
    rows = _clean_table(raw)[0]
    assert rows[0] == ["资产", "2020.6.30", "2019.12.31", "2018.12.31", "2017.12.31"]
    assert rows[1][0] == "流动资产"
    assert rows[1][1] == "1,031,675,913.97"
    assert rows[1][4] == "1,095,402,830.99"
    assert rows[3][0] == "归属于母公司所有者权益合计"


def test_find_page_tables_recovers_quecheng_balance_sheet():
    from pathlib import Path

    path = Path(
        "data/uploads/f927dca2-3ed0-49c3-8229-aa3f60dc5919/确成硅化学股份有限公司.pdf"
    )
    if not path.exists():
        return
    import pymupdf

    doc = pymupdf.open(path)
    tables = _find_page_tables(doc[42])
    doc.close()
    assert tables
    rows = _clean_table(tables[0].extract())[0]
    assert rows[0][0] == "资产"
    assert len(rows[0]) == 5
    assert rows[1][0] == "流动资产"


# ——— 无框释义表（"X 指 Y"）：国电电力第 6 页实录 ———

def _w(x0, y0, x1, y1, text):
    return (x0, y0, x1, y1, text)


def _glossary_lines() -> list[list[tuple]]:
    """模拟国电第 6 页的版式：词条 @x≈90、指 @x≈271、释义 @x≈316，
    词条和释义都带一次换行。"""
    return [
        [_w(400, 44, 520, 56, "招股意向书")],                # 页眉
        [_w(240, 80, 300, 96, "第一节"), _w(310, 80, 350, 96, "释义")],
        [_w(110, 139, 480, 152, "在本招股意向书中，除非另有说明")],
        [_w(90, 190, 260, 204, "国电电力、发行人、本公司、"),   # 锚点行
         _w(271, 190, 285, 204, "指"),
         _w(321, 190, 500, 204, "国电电力发展股份有限公司")],
        [_w(90, 209, 130, 222, "公司")],                     # 词条续行
        [_w(90, 236, 180, 250, "国电集团"),
         _w(271, 236, 285, 250, "指"),
         _w(321, 236, 460, 250, "中国国电集团公司")],
        [_w(90, 264, 180, 278, "龙源集团"),
         _w(271, 264, 285, 278, "指"),
         _w(316, 264, 420, 278, "龙源电力集团公司"),
         _w(430, 264, 470, 278, "有")],                      # 释义行内两个词
        # 正文回来了：字横跨锚点列，释义表到此为止
        [_w(90, 320, 500, 336, "本招股意向书中，除非文义另有所指，下列词语具有如下含义")],
        [_w(90, 350, 200, 366, "普通股"), _w(271, 350, 285, 366, "指"),
         _w(300, 350, 400, 366, "每股面值一元的股份")],
    ]


def test_glossary_rows_extracts_term_and_definition():
    rows, bbox = _glossary_rows(_glossary_lines())
    assert len(rows) == 3
    assert rows[0] == ["国电电力、发行人、本公司、公司", "国电电力发展股份有限公司"]
    assert rows[1] == ["国电集团", "中国国电集团公司"]
    assert rows[2] == ["龙源集团", "龙源电力集团公司有"]


def test_glossary_rows_returns_bbox_of_table_area():
    _rows, bbox = _glossary_rows(_glossary_lines())
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    # 覆盖锚点行到最后一行，页眉和表后正文不在里面
    assert 185 <= y0 <= 195
    assert 270 <= y1 <= 280
    assert x0 <= 90 and x1 >= 460


def test_glossary_rows_ignores_pages_without_anchor_column():
    # 孤立的「指」只有一两个、不成列：不是释义表
    lines = [
        [_w(90, 100, 200, 114, "除非文义另有所指")],
        [_w(90, 140, 285, 154, "词语"), _w(300, 140, 314, 154, "指"),
         _w(330, 140, 430, 154, "含义如下")],
    ]
    rows, bbox = _glossary_rows(lines)
    assert rows == [] and bbox is None


def test_pdf_glossary_tables_wraps_rows_with_markdown_shape():
    class FakePage:
        def get_text(self, _kind):
            return [
                (90, 190, 260, 204, "国电集团", 0, 0, 0),
                (271, 190, 285, 204, "指", 0, 0, 0),
                (321, 190, 500, 204, "国电电力发展股份有限公司", 0, 0, 0),
                (90, 236, 180, 250, "龙源集团", 0, 0, 0),
                (271, 236, 285, 250, "指", 0, 0, 0),
                (316, 236, 460, 250, "龙源电力集团公司", 0, 0, 0),
                (90, 264, 180, 278, "中国证监会", 0, 0, 0),
                (271, 264, 285, 278, "指", 0, 0, 0),
                (316, 264, 480, 278, "中国证券监督管理委员会", 0, 0, 0),
            ]

    class FakeDoc:
        page_count = 1

        def __getitem__(self, _i):
            return FakePage()

    out = _pdf_glossary_tables(FakeDoc(), [])
    assert len(out) == 1
    page, y0, bbox, rows = out[0]
    assert page == 1 and y0 == 190
    assert rows[0] == ["词语", "含义"]
    assert rows[1] == ["国电集团", "国电电力发展股份有限公司"]


def test_split_repeated_header_keeps_glossary_in_one_table():
    """释义表每条形状都一样、但列名各不相同（只有「指」重复），不该逐条切开。"""
    rows = [
        ["词语", "指", "含义"],
        ["国电电力", "指", "国电电力发展股份有限公司"],
        ["国电集团", "指", "中国国电集团公司"],
        ["龙源集团", "指", "龙源电力集团公司"],
    ]
    assert _split_repeated_header(rows) == [rows]


def test_split_repeated_header_still_needs_shared_label():
    """共用列名才拆：第二条表的表头跟首行有相同的「项目」。"""
    rows = [
        ["项目", "2016-12-31", "2015-12-31"],
        ["资产负债率", "59.07", "60.12"],
        ["流动比率", "0.96", "0.88"],
        ["项目", "2016年度", "2015年度"],
        ["应收账款周转率", "3.88", "4.02"],
    ]
    parts = _split_repeated_header(rows)
    assert len(parts) == 2
    assert parts[1][0] == rows[3]


# ——— 键值表补头 + 孤儿标点行（华瑞第 3/32 页「本次发行的基本情况」实录）———

def test_clean_cell_restores_dots_from_orphan_line():
    # 「3.65」的小数点被甩到两行之外，成了孤立的「.」行
    assert _clean_cell(
        "365元/股（按照2016年12月31日经审计的归属于母公\n."
        "\n司所有者权益除以本次发行前总股本计算）"
    ) == "3.65元/股（按照2016年12月31日经审计的归属于母公司所有者权益除以本次发行前总股本计算）"


def test_clean_cell_restores_thousands_with_orphan_dot_row():
    # 千分位逗号+小数点一起甩出：「2,000.00」「310.00」
    assert (
        _clean_cell("承销费用 200000万元\n, .\n保荐费用 31000万元\n.")
        == "承销费用2,000.00万元保荐费用310.00万元"
    )


def test_clean_cell_drops_prose_punct_rows_keeps_final_period():
    # 句读孤儿行（含中文标点）删掉，句末句号（末行）保留
    assert (
        _clean_cell(
            "本次公开发行股份总数不超过 2500 万股 占发行后总股本\n, ，\n"
            "比例不低于2500% 本次发行全部为新股发行 原股东不公\n. 。 ，\n开发售股份\n。"
        )
        == "本次公开发行股份总数不超过2500万股占发行后总股本比例不低于2500%"
        "本次发行全部为新股发行原股东不公开发售股份。"
    )


def test_add_kv_header_prepends_for_label_value_table():
    # 华瑞第 32 页发行概况：每行「短标签 + 长内容」，没有表头行；
    # 净资产的「3.65」小数点甩成孤立行（实录原样）
    rows = _clean_table(
        [
            ["股票类型", "人民币普通股（A股）"],
            ["每股面值", "人民币100元\n."],
            ["发行前每股净资产", "365元/股（按照2016年12月31日经审计）\n."],
            ["承销方式", "余额包销"],
        ]
    )[0]
    assert rows[0] == ["项目", "内容"]
    assert rows[2] == ["每股面值", "人民币1.00元"]
    assert rows[3][1].startswith("3.65元/股")


def test_add_kv_header_skips_table_with_real_header():
    # 首行本身就是表头（「指标 | 数值」/「词语 | 含义」）时不补
    for header in (["指标", "数值"], ["词语", "含义"]):
        rows = _clean_table([header, ["国电集团", "中国国电集团公司"]])[0]
        assert rows[0] == header


# --- 表题（caption）--------------------------------------------------------
# 实录：能建三季报 p16，表体从 136.8pt 开始，上方依次是
# 「母公司利润表」「2026 年1—9 月」「编制单位：…」「单位：千元…」。
# 没有表题时，母公司利润表的切片里一个「母公司」都没有。


def test_caption_above_skips_unit_and_date_lines():
    lines = [
        line("母公司利润表", 16, 0.09, (60.0, 78.8, 300.0, 90.0)),
        line("2026 年1—9 月", 16, 0.11, (60.0, 92.9, 300.0, 104.0)),
        line("编制单位：中国能源建设股份有限公司", 16, 0.13, (60.0, 110.0, 400.0, 122.0)),
        line("单位：千元币种：人民币审计类型：未经审计", 16, 0.15, (60.0, 124.0, 500.0, 136.0)),
    ]
    assert _caption_above(lines, 16, 137.0) == "母公司利润表"


def test_caption_above_accepts_section_heading():
    # 正文数据表没有「XX表」的表题，最近的一行是节标题
    lines = [line("（一）主要会计数据和财务指标", 1, 0.5, (60.0, 500.0, 400.0, 512.0))]
    assert _caption_above(lines, 1, 577.0) == "（一）主要会计数据和财务指标"


def test_caption_above_ignores_body_sentence():
    # 「…如下：」是正文，不是表题；认错表题会把别的主体安到这张表上
    lines = [line("报告期内公司主要经营情况如下：", 3, 0.1, (60.0, 100.0, 400.0, 110.0))]
    assert _caption_above(lines, 3, 130.0) is None


def test_caption_above_ignores_other_pages_and_far_lines():
    lines = [line("合并资产负债表", 7, 0.2, (60.0, 180.0, 300.0, 192.0))]
    assert _caption_above(lines, 8, 240.0) is None
    # 超出窗口（120pt）不算：那是上一张表的标题
    assert _caption_above(lines, 7, 400.0) is None


def test_attach_captions_inherits_title_for_page_top_continuation():
    """跨页续表：表题只在首页印一次，后续页从页顶接着排。"""
    header = ["项目", "2026年9月30日"]
    tables = [
        (7, 240.0, (0.0, 240.0, 500.0, 800.0), [header, ["营业收入", "1"]]),
        (8, 76.0, (0.0, 76.0, 500.0, 700.0), [header, ["总资产", "2"]]),
    ]
    lines = [line("合并资产负债表", 7, 0.2, (60.0, 180.0, 300.0, 192.0))]
    out = _attach_captions(tables, lines)
    assert out[0][4] == "合并资产负债表"
    assert out[1][4] == "合并资产负债表"


def test_attach_captions_does_not_borrow_for_midpage_table():
    """页中开始的表是新表，不能套上一张表的表题。"""
    tables = [
        (7, 240.0, (0.0, 240.0, 500.0, 400.0), [["项目", "值"], ["营业收入", "1"]]),
        (7, 600.0, (0.0, 600.0, 500.0, 760.0), [["项目", "值"], ["手续费收入", "2"]]),
    ]
    lines = [line("合并资产负债表", 7, 0.2, (60.0, 180.0, 300.0, 192.0))]
    out = _attach_captions(tables, lines)
    assert out[0][4] == "合并资产负债表"
    assert out[1][4] is None
