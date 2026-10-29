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
    _glossary_rows,
    _pdf_glossary_tables,
    _rows_to_markdown,
    _split_repeated_header,
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
