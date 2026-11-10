"""OTSL 收成带合并单元格的 HTML 表，正文不含标记时保持原样。"""

from ai_agent.kb.ingest.otsl import finalize_document, render_otsl

_PAGE = (
    "用<fcel>不适用<fcel>→1,170,945,414.69<fcel>不适用<nl>"
    "<fcel>基本每股收益(元/股)<fcel>0.03<fcel>200.00<fcel>0.20<fcel>0.00<nl>"
    "<fcel>本报告期末<fcel>上年度末<fcel>增减变动幅度(%)<nl>"
    "<fcel>总资产<fcel>11,606,276,474.35<fcel>12,542,760,941.90<fcel>-7.47<nl>"
)


def test_plain_text_unchanged():
    text = "营业收入 12.3 亿元，同比增长 4%。"
    assert render_otsl(text) == text
    assert finalize_document(text) == text


def test_financial_table_becomes_html():
    table = render_otsl(_PAGE)
    assert "<fcel>" not in table
    assert "<ecel>" not in table
    assert "<nl>" not in table
    assert table.startswith("<table>")
    assert "基本每股收益(元/股)" in table
    assert "0.03" in table
    assert "本报告期末" in table
    assert "上年度末" in table
    assert "总资产" in table
    assert "11,606,276,474.35" in table


def test_merged_cells_keep_span():
    table = render_otsl(
        "<fcel>项目<lcel><fcel>本期<nl><fcel>收入<fcel><ucel><fcel>100<nl>"
    )
    assert 'colspan="2"' in table
    assert 'rowspan="2"' in table
    assert table.count("项目") == 1
    assert "<lcel>" not in table
    assert "<ucel>" not in table


def test_prose_around_table_is_kept():
    text = "主要会计数据如下。\n\n" + _PAGE + "\n\n以上为合并口径。"
    rendered = render_otsl(text)
    assert rendered.startswith("主要会计数据如下。")
    assert rendered.rstrip().endswith("以上为合并口径。")
    assert "<fcel>" not in rendered
    assert "<table>" in rendered
    assert "总资产" in rendered


def test_finalize_normalizes_math_and_escapes_cells():
    text = "公式 \\[a<b\\]\n\n<fcel>甲&乙<nl>"
    rendered = finalize_document(text)
    assert "$$ a<b $$" in rendered
    assert "<td>甲&amp;乙</td>" in rendered
