"""PDF 由 WeasyPrint 从 HTML 排出。数字沿用正文，工具不把二进制交回调用方。"""

import base64

import pymupdf

from ai_agent.report.compare import Measure, compare_measures
from ai_agent.report.compose import ReportDocument, compose_report
from ai_agent.report.extract import Evidence, IndicatorFact
from ai_agent.report.parse import OutlineItem, ReportPlan
from ai_agent.report.pdf import (
    ChartSpec,
    _chart_title,
    _chart_title_lines,
    _format_axis_tick,
    apply_print_template,
    chart_svg,
    charts_from_series,
    find_cjk_font,
    render_report_pdf,
)
from ai_agent.tools.registry import TOOLS
from ai_agent.tools.report.report_pdf import report_pdf_tool


def _fact(period: str, value: str, name: str = "营业收入") -> IndicatorFact:
    return IndicatorFact(
        subject="中国能源建设股份有限公司",
        period=period,
        name=name,
        value=value,
        unit="元",
        column="本报告期",
        scope="consolidated",
        evidence=Evidence(
            file_id="file-1",
            filename="中国能源建设2025年第三季度报告.pdf",
            page=12,
            page_end=12,
            section="主要会计数据",
            quote="营业收入 | " + value,
        ),
    )


def _plan() -> ReportPlan:
    return ReportPlan(
        title="中国能源建设股份有限公司 2025Q3 财报深度分析",
        kind="company",
        deliverable="pdf",
        subjects=["中国能源建设股份有限公司"],
        period="2025Q3",
        dimensions=["同比"],
        indicators=["营业收入"],
        outline=[
            OutlineItem("本期结论", ""),
            OutlineItem("主要指标与出处", ""),
            OutlineItem("同比与环比", ""),
        ],
    )


def _document():
    return compose_report(
        _plan(),
        [_fact("2025Q3", "100,924,954"), _fact("2024Q3", "111,000,000")],
    )


def test_template_sets_header_footer_and_repeating_table_head():
    font = find_cjk_font()
    assert font is not None
    logo = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    printable = apply_print_template(
        _document().html,
        title="中国能源建设股份有限公司 2025Q3 财报深度分析",
        font=font,
        logo=logo,
        charts=charts_from_series(
            compare_measures(
                [
                    Measure("中国能源建设股份有限公司", "营业收入", "2024Q3", "111,000,000", "元"),
                    Measure("中国能源建设股份有限公司", "营业收入", "2025Q3", "100,924,954", "元"),
                ]
            )
        ),
    )
    assert "@page" in printable
    assert "counter(pages)" in printable
    assert "table-header-group" in printable
    assert "data:image/png;base64," in printable
    assert "100,924,954" in printable
    assert printable.lstrip().startswith("<style>") or "<style>" in printable
    assert "%PDF" not in printable


def test_chart_uses_the_given_numbers():
    svg = chart_svg(
        charts_from_series(
            compare_measures(
                [
                    Measure("中国能源建设股份有限公司", "营业收入", "2024Q3", "111,000,000", "元"),
                    Measure("中国能源建设股份有限公司", "营业收入", "2025Q3", "100,924,954", "元"),
                ]
            )
        )[0]
    )
    assert "111,000,000" in svg
    assert "100,924,954" in svg
    # 两点同比一律柱状，禁止折线伪趋势
    assert "<rect" in svg
    assert "<polyline" not in svg
    charts = charts_from_series(
        compare_measures(
            [
                Measure("中国能源建设股份有限公司", "营业收入", "2024Q3", "111,000,000", "元"),
                Measure("中国能源建设股份有限公司", "营业收入", "2025Q3", "100,924,954", "元"),
            ]
        )
    )
    assert charts[0].kind == "bar"
    assert "两点对比" in charts[0].title
    bar = chart_svg(ChartSpec(title="毛利率", kind="bar", labels=["2025Q3"], values=["12.5%"], unit="%"))
    assert "12.5%" in bar
    assert "<rect" in bar
    assert chart_svg(ChartSpec(title="空", kind="bar", labels=["2025Q3"], values=["不适用"])) == ""


def test_chart_title_keeps_two_point_suffix():
    title = (
        "兆易创新科技集团股份有限公司 经营活动产生的现金流量净额"
        "（累计口径，非单季）（两点对比）"
    )
    shown = _chart_title(title)
    assert "两点对比" in shown
    assert "（两" not in shown.replace("（两点对比）", "")
    lines = _chart_title_lines(title, width_chars=34)
    assert any("两点对比" in line for line in lines)
    assert all("（两" != line[-2:] for line in lines)
    assert all(not line.endswith("（两") for line in lines)
    svg = chart_svg(
        ChartSpec(
            title=title,
            kind="bar",
            labels=["2025Q3", "2026Q3"],
            values=["1000000000", "2000000000"],
            unit="元",
        )
    )
    assert "两点对比" in svg
    assert "（两<" not in svg
    assert svg.count("<text") >= 2  # 至少标题多行或标题+刻度


def test_axis_ticks_use_yi_consistently_for_loss_chart():
    """扣非亏损图：峰值过亿时正负刻度一律「亿」，禁止 36,946,835 裸数字。"""
    import re
    from decimal import Decimal

    high = Decimal("36946835")
    assert _format_axis_tick(
        high, "元", span=Decimal("250000000"), use_yi=True
    ) == "0.37亿"
    svg = chart_svg(
        ChartSpec(
            title="深圳中科飞测 扣非净利润（累计口径，非单季）（两点对比）",
            kind="bar",
            labels=["2025Q3", "2026Q3"],
            values=["-139958727.02", "-205260194.78"],
            unit="元",
        )
    )
    assert "两点对比" in svg
    assert "36,946,835" not in svg
    ticks = re.findall(r">([^<]+)<", svg)
    money_ticks = [t for t in ticks if "亿" in t or re.fullmatch(r"-?[\d,]+", t)]
    # 金额刻度必须带「亿」，不能混入千分位裸整数。
    bare = [
        t
        for t in money_ticks
        if "亿" not in t and re.search(r"\d{5,}", t.replace(",", ""))
    ]
    assert not bare, bare
    assert any("亿" in t for t in ticks)


def test_bar_chart_y_axis_does_not_go_negative_for_positive_values():
    """全为正的柱状图：下沿钉在 0，不再 pad 出负刻度。"""
    import re

    from ai_agent.report.pdf import _chart_y_domain
    from decimal import Decimal

    low, high = _chart_y_domain(
        [Decimal("100"), Decimal("200")], include_zero=True
    )
    assert low == 0
    assert high > 200

    svg = chart_svg(
        ChartSpec(
            title="营收",
            kind="bar",
            labels=["2024Q3", "2025Q3"],
            values=["1000000", "2000000"],
            unit="元",
        )
    )
    ticks = re.findall(r">(-?[\d,]+(?:\.\d+)?)<", svg)
    assert any(t.replace(",", "") == "0" or t == "0" for t in ticks)
    assert not any(t.startswith("-") for t in ticks if t not in {"-"})


def test_line_chart_scales_to_data_range_not_zero():
    """比率两期接近时，Y 轴应按数据区间缩放，不能贴在图顶变成一条平线。"""
    import re

    svg = chart_svg(
        ChartSpec(
            title="净息差",
            kind="line",
            labels=["2025Q3", "2026Q3"],
            values=["1.26%", "1.27%"],
            unit="%",
        )
    )
    circles = re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)"', svg)
    assert len(circles) == 2
    y0, y1 = float(circles[0][1]), float(circles[1][1])
    # 旧实现强制从 0 起，两点 cy 都挤在 ~28；修好后应拉开到绘图区中部附近
    assert abs(y0 - y1) > 20
    assert min(y0, y1) > 40
    assert max(y0, y1) < 220


def test_pdf_keeps_chinese_and_extracted_number():
    charts = charts_from_series(
        compare_measures(
            [
                Measure("中国能源建设股份有限公司", "营业收入", "2024Q3", "111,000,000", "元", "consolidated"),
                Measure("中国能源建设股份有限公司", "营业收入", "2025Q3", "100,924,954", "元", "consolidated"),
            ]
        )
    )
    logo = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    pdf = render_report_pdf(_document(), charts=charts, logo=logo)
    assert pdf.content.startswith(b"%PDF")
    assert pdf.filename.endswith(".pdf")
    assert pdf.pages >= 1
    text = _text(pdf.content)
    assert "100,924,954" in text
    assert "111,000,000" in text
    assert "中国能源建设" in text
    assert "-10,075,046" in text
    assert "-9.08%" in text
    assert "第 1 页" in text


def test_long_table_breaks_across_pages():
    rows = "".join(
        "<tr><td>中国能源建设股份有限公司</td><td>营业收入</td><td>100,924,954元</td></tr>"
        for _ in range(80)
    )
    html = (
        "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"></head><body>"
        "<h1>长表</h1><table><thead><tr><th>公司</th><th>指标</th><th>数值</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>"
    )
    pdf = render_report_pdf(ReportDocument(title="长表", markdown="", html=html, deliverable="pdf"))
    assert pdf.pages >= 2
    assert "100,924,954" in _text(pdf.content)


def test_tool_writes_pdf_and_stays_off_the_chat_loop(tmp_path):
    target = tmp_path / "out" / "report.pdf"
    message = report_pdf_tool.run(html=_document().html, title="中国能源建设 2025Q3", output_path=str(target))
    assert target.is_file()
    assert target.read_bytes().startswith(b"%PDF")
    assert "%PDF" not in message
    assert "共" in message and "页" in message
    assert "report_pdf" not in [tool.name for tool in TOOLS]


def _text(content: bytes) -> str:
    document = pymupdf.open(stream=content, filetype="pdf")
    try:
        return "".join(page.get_text() for page in document)
    finally:
        document.close()
