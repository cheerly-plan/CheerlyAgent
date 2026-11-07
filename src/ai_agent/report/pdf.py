"""把研报 HTML 排成 PDF。二进制只由 WeasyPrint 写出，不由模型生成。

页眉、页脚、Logo 和中文字体写在打印模板里。趋势图和柱状图是根据已算好的数字画的 SVG，
再交给 WeasyPrint 分页。Mermaid 不在这里执行。
"""

from __future__ import annotations

import base64
import html
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from ai_agent.report import registry
from ai_agent.report.compare import SeriesCompare, parse_number
from ai_agent.report.compose import ReportDocument
from ai_agent.report.extract import IndicatorFact
from ai_agent.report.parse import ReportPlan
from ai_agent.report.skills import get_skill

_FONT_FAMILY = "ReportCJK"
_KNOWN_FONTS = (
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
)
_FONT_NEEDLES = ("notosanscjk", "notosanssc", "sourcehansans", "sourcehanserif", "arialunicode")
_FONT_SUFFIXES = {".ttf", ".otf", ".ttc", ".otc"}
_FILENAME = re.compile(r'[\\/:*?"<>|\s]+')


@dataclass
class ChartSpec:
    title: str
    labels: list[str]
    values: list[str]
    kind: str = "bar"
    unit: str = ""


@dataclass
class PdfFile:
    filename: str
    content: bytes
    pages: int = 0
    gaps: list[str] = field(default_factory=list)


def find_cjk_font() -> Path | None:
    """找一个可以嵌进 PDF 的中文字体。找不到就不要生成，否则中文会变成方框。"""
    env = os.environ.get("REPORT_CJK_FONT", "").strip()
    if env and Path(env).is_file():
        return Path(env)
    for path in _KNOWN_FONTS:
        if path.is_file():
            return path
    roots = (Path.home() / "Library/Fonts", Path("/Library/Fonts"), Path("/usr/share/fonts"))
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in _FONT_SUFFIXES or not path.is_file():
                continue
            name = path.name.lower().replace(" ", "").replace("-", "").replace("_", "")
            if any(needle in name for needle in _FONT_NEEDLES):
                return path
    return None


def charts_from_series(series: list[SeriesCompare]) -> list[ChartSpec]:
    """每个至少两期的指标画一张趋势图。用的是对比结果里的原始数字。"""
    charts: list[ChartSpec] = []
    for item in series:
        if len(item.points) < 2:
            continue
        # 只画同口径的点。1-3月 / 1-6月 / 1-9月的累计值连成折线天然递增，
        # 会被误读成「营收逐季增长」——那不是趋势，是期数在变。
        groups: dict[str, list[tuple[str, str]]] = {}
        for period, value in item.points:
            groups.setdefault(item.bases.get(period, ""), []).append((period, value))
        usable = [points for points in groups.values() if len(points) >= 2]
        if usable:
            basis_key = max(groups, key=lambda key: len(groups[key]))
            points = groups[basis_key]
        elif item.bases:
            continue  # 有口径信息但没有任何可比的两点，宁可不画
        else:
            basis_key = ""
            points = item.points  # 旧调用没标口径，保持兼容
        suffix = {
            "cumulative": "（累计口径，非单季）",
            "single": "（单季）",
            "point": "（比率，单位 %）",
            "stock": "（期末时点余额，非累计）",
        }.get(basis_key.split(":", 1)[0], "")
        # 两点／三点画柱状，避免「折线伪趋势」；至少四期同口径才画折线。
        n = len(points)
        kind = "line" if n >= 4 else "bar"
        shape = "多期趋势" if kind == "line" else "两点对比" if n == 2 else "多期对比"
        charts.append(
            ChartSpec(
                title=f"{item.subject} {item.name}{suffix}（{shape}）",
                kind=kind,
                labels=[period for period, _value in points],
                values=[value for _period, value in points],
                unit=item.unit,
            )
        )
    return charts


def charts_for_report(
    plan: ReportPlan,
    facts: list[IndicatorFact],
    series: list[SeriesCompare],
) -> list[ChartSpec]:
    """行业赛道报告优先画「组内横比」柱状图；其它报告沿用多期对比图。"""
    if plan.skill_id == "semiconductor_industry_report" or any(
        item.title == "组内对比" for item in plan.outline
    ):
        peer = _segment_peer_charts(plan, facts)
        if peer:
            return peer
    return charts_from_series(series)


def _segment_peer_charts(
    plan: ReportPlan, facts: list[IndicatorFact]
) -> list[ChartSpec]:
    """每赛道每指标一张横比柱状图（公司作 X 轴）；单公司赛道跳过。"""
    skill = get_skill(plan.skill_id) if plan.skill_id else None
    if skill is None or not skill.companies:
        return []
    current = [fact for fact in facts if fact.period == plan.period]
    indicators = [
        name
        for name in (plan.indicators or list(registry.ordering_list("pdf_default_indicators")))
        if registry.flag(
            name,
            "chart",
            default=name in registry.names_with("chart_default"),
        )
    ]
    charts: list[ChartSpec] = []
    for segment, names in skill.companies_by_segment():
        if len(names) < 2:
            continue
        for indicator in indicators:
            rows: list[tuple[str, str, str]] = []
            for company in names:
                match = next(
                    (
                        fact
                        for fact in current
                        if fact.name == indicator
                        and (
                            fact.subject == company
                            or company in fact.subject
                            or fact.subject in company
                        )
                    ),
                    None,
                )
                if match is None:
                    continue
                rows.append((company, match.value, match.unit))
            if len(rows) < 2:
                continue
            units = {unit for _n, _v, unit in rows}
            unit = next(iter(units)) if len(units) == 1 else rows[0][2]
            # 单位不齐时跳过该图，避免元/千元横比误导
            if len(units) > 1 and unit != "%":
                continue
            charts.append(
                ChartSpec(
                    title=f"{segment}·{indicator}（组内横比）",
                    kind="bar",
                    labels=[name for name, _v, _u in rows],
                    values=[value for _n, value, _u in rows],
                    unit=unit,
                )
            )
    return charts


def _chart_y_domain(
    amounts: list[Decimal], *, include_zero: bool
) -> tuple[Decimal, Decimal]:
    """折线按数据区间加边距缩放；柱状图仍含 0，否则高低对比会失真。

    全部为正的柱状图：下沿钉在 0，禁止「pad 后出现负刻度」导致柱高失真。
    旧实现折线也强制从 0 起：净息差 1.26→1.27、营收近邻两期都会贴在图顶，
    看起来像一条几乎水平的斜线，变化读不出来。
    """
    low = min(amounts)
    high = max(amounts)
    all_nonneg = min(amounts) >= 0
    if include_zero:
        low = min(low, Decimal(0))
        high = max(high, Decimal(0))
    if high == low:
        pad = abs(high) * Decimal("0.08") if high != 0 else Decimal(1)
        low, high = low - pad, high + pad
    else:
        pad = (high - low) * Decimal("0.18")
        low, high = low - pad, high + pad
    if include_zero:
        if all_nonneg:
            low = Decimal(0)
        else:
            low = min(low, Decimal(0))
        high = max(high, Decimal(0))
    if high == low:
        high = low + Decimal(1)
    return low, high


def _chart_display_value(raw: str, unit: str) -> str:
    text = str(raw).strip()
    if unit == "%" and text and not text.endswith("%") and not text.endswith("％"):
        return f"{text}%"
    return text


def _chart_title_suffix(title: str) -> tuple[str, str]:
    """拆出末尾连续全角括号（口径／图型），换行时整块下移，禁止出现「（两」。"""
    text = (title or "").strip()
    suffix = ""
    rest = text
    while True:
        match = re.search(r"（[^）]*）\s*$", rest)
        if not match:
            break
        suffix = match.group(0) + suffix
        rest = rest[: match.start()]
    return rest.rstrip(), suffix


def _chart_title(title: str, max_chars: int = 56) -> str:
    """单行兜底：过长时省略中间，末尾括号必须完整。"""
    text = (title or "").strip()
    if len(text) <= max_chars:
        return text
    rest, suffix = _chart_title_suffix(text)
    if suffix:
        budget = max_chars - len(suffix) - 1
        if budget >= 8:
            return rest[:budget].rstrip() + "…" + suffix
    return text[: max_chars - 1] + "…"


def _chart_title_lines(title: str, *, width_chars: int = 34) -> list[str]:
    """按图宽折成多行；后缀括号独占末行尾，绝不半截。"""
    text = (title or "").strip()
    if not text:
        return [""]
    rest, suffix = _chart_title_suffix(text)
    if len(text) <= width_chars:
        return [text]
    lines: list[str] = []
    cursor = rest
    # 前几行只排主体；最后一行留给「主体尾 + 后缀」。
    while len(cursor) + len(suffix) > width_chars and len(cursor) > 8:
        # 尽量在空格／顿号处断，找不到再硬切。
        window = cursor[:width_chars]
        break_at = max(window.rfind(" "), window.rfind("·"), window.rfind("，"))
        if break_at < width_chars // 3:
            break_at = width_chars
        lines.append(cursor[:break_at].rstrip())
        cursor = cursor[break_at:].lstrip()
        if len(lines) >= 2:
            break
    tail = (cursor + suffix).strip()
    if len(tail) <= width_chars:
        lines.append(tail)
    else:
        # 主体仍过长：主体省略，后缀完整跟在同一行。
        budget = max(width_chars - len(suffix) - 1, 8)
        lines.append(cursor[:budget].rstrip() + "…" + suffix)
    return [line for line in lines if line]


def chart_svg(spec: ChartSpec) -> str:
    """按给定数字画柱状图或折线图。读不出的数字跳过，不另编一个值。"""
    rows: list[tuple[str, Decimal, str]] = []
    for label, raw in zip(spec.labels, spec.values):
        parsed = parse_number(str(raw))
        if parsed is None:
            continue
        rows.append((str(label), parsed[0], _chart_display_value(str(raw), spec.unit)))
    if not rows:
        return ""
    kind = "line" if spec.kind in {"line", "趋势"} else "bar"
    width, height = 520, 268
    amounts = [amount for _label, amount, _raw in rows]
    low, high = _chart_y_domain(amounts, include_zero=(kind == "bar"))
    use_yi = _axis_use_yi(spec.unit, low=low, high=high)
    # 金额轴用紧凑刻度时左边距可小一些；超大整数刻度需加宽以免裁字。
    sample_tick = _format_axis_tick(
        high, spec.unit, span=high - low, use_yi=use_yi
    )
    left = 72 if len(sample_tick) > 8 else 52
    right, bottom = 24, 44
    title_lines = _chart_title_lines(spec.title, width_chars=max(28, (width - left - right) // 12))
    title_block = 4 + 14 * len(title_lines)
    top = max(40, title_block + 10)
    # 多行标题时加高画布，避免挤占绘图区。
    if len(title_lines) > 1:
        height = 268 + 14 * (len(title_lines) - 1)
    plot_w = width - left - right
    plot_h = height - top - bottom

    def y_of(amount: Decimal) -> float:
        return top + float((high - amount) / (high - low)) * plot_h

    zero_y = y_of(Decimal(0)) if low <= 0 <= high else None
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
    ]
    for index, line in enumerate(title_lines):
        y = 14 + index * 14
        parts.append(
            f'<text x="{left}" y="{y}" font-family="{_FONT_FAMILY},sans-serif" font-size="11">'
            f"{html.escape(line)}</text>"
        )
    parts.extend(
        [
        # 坐标轴
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#98a2b3" '
        'stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" '
        'stroke="#98a2b3" stroke-width="1"/>',
        ]
    )
    # 三条浅网格，标出高/中/低，避免「只有一根线、看不出量级」
    for fraction in (Decimal(0), Decimal("0.5"), Decimal(1)):
        amount = high - (high - low) * fraction
        y = y_of(amount)
        if fraction < 1:
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                'stroke="#eaecf0" stroke-width="1"/>'
            )
        tick = _format_axis_tick(
            amount, spec.unit, span=high - low, use_yi=use_yi
        )
        parts.append(
            f'<text x="{left - 6}" y="{y + 3:.1f}" text-anchor="end" font-size="9" '
            f'fill="#667085" font-family="{_FONT_FAMILY},sans-serif">'
            f"{html.escape(tick)}</text>"
        )
    if zero_y is not None and kind == "bar":
        parts.append(
            f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}" '
            'stroke="#d0d5dd" stroke-width="1"/>'
        )

    slot = plot_w / len(rows)
    plotted: list[tuple[float, float, str, str]] = []
    for index, (label, amount, display) in enumerate(rows):
        x_mid = left + (index + 0.5) * slot
        y_val = y_of(amount)
        parts.append(
            f'<text x="{x_mid:.1f}" y="{height - 14}" text-anchor="middle" font-size="10" '
            f'font-family="{_FONT_FAMILY},sans-serif">{html.escape(label)}</text>'
        )
        plotted.append((x_mid, y_val, label, display))
        if kind != "line":
            baseline = zero_y if zero_y is not None else top + plot_h
            bar_w = slot * 0.56
            x = x_mid - bar_w / 2
            y = min(y_val, baseline)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{abs(baseline - y_val) or 1:.1f}" fill="#1d4e89"/>'
            )
            # 柱顶数值：一律放在柱上方，避免压进柱体
            label_y = y - 6
            if label_y < top + 10:
                label_y = y + 14
            parts.append(
                f'<text x="{x_mid:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="9" '
                f'fill="#344054" font-family="{_FONT_FAMILY},sans-serif">'
                f"{html.escape(display)}</text>"
            )
    if kind == "line" and plotted:
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y, _label, _display in plotted)
        parts.append(
            f'<polyline fill="none" stroke="#1d4e89" stroke-width="2.2" '
            f'stroke-linejoin="round" stroke-linecap="round" points="{coords}"/>'
        )
        for x, y, _label, _display in plotted:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="#ffffff" '
                'stroke="#1d4e89" stroke-width="2"/>'
            )
        # 数值标签上下交错；相邻点纵坐标接近时强制分居两侧，避免叠字压线
        side_above: list[bool] = []
        for index, (x, y, _label, display) in enumerate(plotted):
            if index == 0:
                above = True
            elif abs(y - plotted[index - 1][1]) < 28:
                above = not side_above[-1]
            else:
                above = index % 2 == 0
            side_above.append(above)
            label_y = y - 14 if above else y + 18
            if label_y < top + 12:
                label_y = y + 18
            if label_y > top + plot_h - 4:
                label_y = y - 14
            parts.append(
                f'<text x="{x:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="9" '
                f'fill="#344054" font-family="{_FONT_FAMILY},sans-serif">'
                f"{html.escape(display)}</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


def _axis_use_yi(unit: str, *, low: Decimal, high: Decimal) -> bool:
    """整图峰值达亿元量级时，正负刻度一律用「亿」，禁止与裸数字混排。"""
    if unit not in {"", "元", "人民币元"}:
        return False
    peak = max(abs(low), abs(high))
    return peak >= Decimal("100000000")


def _format_axis_tick(
    amount: Decimal,
    unit: str,
    *,
    span: Decimal | None = None,
    use_yi: bool | None = None,
) -> str:
    """Y 轴刻度：区间很窄时多留小数，避免高/中/低三格显示成同一个数。"""
    narrow = span is not None and span > 0 and span < Decimal("1")
    if unit == "%":
        places = Decimal("0.001") if narrow else Decimal("0.01")
        text = f"{amount.quantize(places):f}".rstrip("0").rstrip(".")
        return f"{text}%"
    # 亿元量级图：0.37亿／-2.05亿 等同单位，不再出现 36,946,835 裸数字。
    yi_mode = use_yi if use_yi is not None else (
        unit in {"", "元", "人民币元"} and abs(amount) >= Decimal("100000000")
    )
    if yi_mode:
        yi = amount / Decimal("100000000")
        text = f"{yi.quantize(Decimal('0.01')):f}".rstrip("0").rstrip(".")
        if text in {"", "-"}:
            text = "0"
        return f"{text}亿"
    if abs(amount) < 1000 or (span is not None and span < 1000):
        places = Decimal("0.01") if not narrow else Decimal("0.001")
        text = f"{amount.quantize(places):,}"
    else:
        text = f"{amount.to_integral_value():,}"
    if text.endswith(".00"):
        text = text[:-3]
    return text


def apply_print_template(
    source: str,
    *,
    title: str,
    font: Path,
    logo: bytes | None = None,
    charts: list[ChartSpec] | None = None,
) -> str:
    """套上 A4 模板：页眉页脚、可嵌入字体、表格跨页时重复表头。"""
    if source.lstrip().startswith("%PDF"):
        raise ValueError("正文是 PDF 二进制，渲染只接受 HTML")
    style = _print_css(title, font)
    logo_tag = _logo_tag(logo) if logo else ""
    figures = _chart_section(charts or [])
    html_text = source
    if "<head>" in html_text:
        html_text = html_text.replace("<head>", "<head>\n" + style, 1)
    else:
        html_text = style + html_text
    if logo_tag:
        html_text = html_text.replace("<body>", "<body>\n" + logo_tag, 1)
    if figures:
        html_text = html_text.replace("</body>", figures + "\n</body>", 1)
    return html_text


def render_report_pdf(
    document: ReportDocument,
    *,
    charts: list[ChartSpec] | None = None,
    logo: bytes | None = None,
    font_path: Path | None = None,
) -> PdfFile:
    """渲染一份研报 PDF。缺中文字体时直接停下，不写出乱码文件。"""
    font = font_path or find_cjk_font()
    if font is None or not font.is_file():
        raise RuntimeError("没有找到可嵌入的中文字体，已停止生成 PDF")
    printable = apply_print_template(
        document.html or document.body,
        title=document.title or "研报",
        font=font,
        logo=logo,
        charts=charts,
    )
    content = _write_pdf(printable)
    if not content.startswith(b"%PDF"):
        raise RuntimeError("WeasyPrint 没有写出 PDF")
    return PdfFile(
        filename=_filename(document.title),
        content=content,
        pages=_page_count(content),
        gaps=list(document.gaps),
    )


def _print_css(title: str, font: Path) -> str:
    header = _css_string(title.strip() or "研报")
    if len(header) > 42:
        header = header[:41] + "…"
    return f"""<style>
@font-face {{
  font-family: "{_FONT_FAMILY}";
  src: {_font_src(font)};
}}
@page {{
  size: A4;
  margin: 18mm 14mm 16mm 14mm;
  @top-left {{ content: element(page-logo); }}
  @top-center {{
    content: "{header}";
    font-family: "{_FONT_FAMILY}", sans-serif;
    font-size: 9pt;
    color: #667085;
  }}
  @bottom-center {{
    content: "第 " counter(page) " 页 / 共 " counter(pages) " 页";
    font-family: "{_FONT_FAMILY}", sans-serif;
    font-size: 9pt;
    color: #667085;
  }}
}}
body {{
  font-family: "{_FONT_FAMILY}", sans-serif;
  font-size: 11pt;
  line-height: 1.55;
  color: #1d2939;
}}
h1 {{ font-size: 16pt; break-after: avoid; }}
h2 {{ font-size: 13pt; break-after: avoid; }}
p, li {{ orphans: 2; widows: 2; }}
table {{ width: 100%; border-collapse: collapse; page-break-inside: auto; }}
thead {{ display: table-header-group; }}
tr, th, td {{ break-inside: avoid; page-break-inside: avoid; }}
th, td {{ border: 0.3pt solid #d0d5dd; padding: 1.0mm 1.2mm; font-size: 8pt; vertical-align: top; }}
th {{ background: #f2f4f7; }}
/* 禁止千分位数字中途断行（overflow-wrap:anywhere 会把 87,654,926,728.13 拆成两截） */
td, p, li {{ word-break: keep-all; overflow-wrap: break-word; }}
.num {{ white-space: nowrap; }}
.page-logo {{ position: running(page-logo); height: 8mm; }}
figure {{ break-inside: avoid; page-break-inside: avoid; margin: 4mm 0; }}
</style>
"""


def _font_src(font: Path) -> str:
    uri = font.resolve().as_uri()
    if font.suffix.lower() == ".otf":
        return f'url("{uri}") format("opentype")'
    if font.suffix.lower() in {".ttc", ".otc"}:
        return f'url("{uri}")'
    return f'url("{uri}") format("truetype")'


def _css_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")


def _logo_tag(logo: bytes) -> str:
    if logo.startswith(b"\x89PNG"):
        mime = "image/png"
    elif logo.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
    else:
        raise ValueError("Logo 只接受 PNG 或 JPEG")
    encoded = base64.b64encode(logo).decode("ascii")
    return f'<img class="page-logo" alt="" src="data:{mime};base64,{encoded}">'


def _chart_section(charts: list[ChartSpec]) -> str:
    figures: list[str] = []
    for spec in charts:
        svg = chart_svg(spec)
        if not svg:
            continue
        caption = "；".join(
            f"{html.escape(label)} {html.escape(value)}{html.escape(spec.unit)}"
            for label, value in zip(spec.labels, spec.values)
        )
        figures.append(f"<figure>{svg}<figcaption>{caption}</figcaption></figure>")
    if not figures:
        return ""
    return '<section class="charts"><h2>图表</h2>' + "".join(figures) + "</section>"


def _prepare_native_libs() -> None:
    """让 WeasyPrint 能找到 Homebrew 里的 Pango。进程启动后再设这条路径也来得及。"""
    for directory in ("/opt/homebrew/lib", "/usr/local/lib"):
        if not Path(directory).is_dir():
            continue
        current = [part for part in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":") if part]
        if directory not in current:
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join([directory, *current])
        return


def _filename(title: str) -> str:
    cleaned = _FILENAME.sub(" ", title).strip()
    return (cleaned or "研报") + ".pdf"


def _write_pdf(source: str) -> bytes:
    _prepare_native_libs()
    try:
        from weasyprint import HTML
    except ImportError as exc:
        raise RuntimeError("未安装 WeasyPrint，无法生成 PDF") from exc
    except OSError as exc:
        raise RuntimeError("WeasyPrint 缺少 Pango、Cairo 等系统库，无法生成 PDF") from exc
    data = HTML(string=source).write_pdf()
    if not isinstance(data, bytes):
        raise RuntimeError("WeasyPrint 没有返回 PDF 字节")
    return data


def _page_count(content: bytes) -> int:
    import pymupdf

    document = pymupdf.open(stream=content, filetype="pdf")
    try:
        return document.page_count
    finally:
        document.close()
