"""OTSL 标记收成 Markdown 表格，正文不含标记时保持原样。"""

from ai_agent.kb.otsl import render_otsl

_PAGE = (
    "用<fcel>不适用<fcel>→1,170,945,414.69<fcel>不适用<nl>"
    "<fcel>基本每股收益(元/股)<fcel>0.03<fcel>200.00<fcel>0.20<fcel>0.00<nl>"
    "<ecel>本报告期末<fcel>上年度末<fcel>增减变动幅度(%)<nl>"
    "<fcel>总资产<fcel>11,606,276,474.35<fcel>12,542,760,941.90<fcel>-7.47<nl>"
)


def test_plain_text_unchanged():
    text = "营业收入 12.3 亿元，同比增长 4%。"
    assert render_otsl(text) == text


def test_financial_table_becomes_markdown():
    table = render_otsl(_PAGE)
    assert "<fcel>" not in table
    assert "<ecel>" not in table
    assert "<nl>" not in table
    lines = [line for line in table.splitlines() if line.startswith("|")]
    assert lines[0] == "| 用 | 不适用 | →1,170,945,414.69 | 不适用 |  |"
    assert "基本每股收益(元/股)" in lines[2]
    assert "0.03" in lines[2]
    assert any("本报告期末" in line and "上年度末" in line for line in lines)
    assert any(line.startswith("| 总资产 |") and "11,606,276,474.35" in line for line in lines)


def test_prose_around_table_is_kept():
    text = "主要会计数据如下。\n\n" + _PAGE + "\n\n以上为合并口径。"
    rendered = render_otsl(text)
    assert rendered.startswith("主要会计数据如下。")
    assert rendered.rstrip().endswith("以上为合并口径。")
    assert "<fcel>" not in rendered
    assert "| 总资产 |" in rendered
