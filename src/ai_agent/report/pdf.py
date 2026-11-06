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

from ai_agent.report.compare import SeriesCompare, parse_number
from ai_agent.report.compose import ReportDocument

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
        charts.append(
            ChartSpec(
                title=f"{item.subject} {item.name}{suffix}",
                kind="line",
                labels=[period for period, _value in points],
                values=[value for _period, value in points],
                unit=item.unit,
            )
        )
    return charts


def chart_svg(spec: ChartSpec) -> str:
    """按给定数字画柱状图或折线图。读不出的数字跳过，不另编一个值。"""
    rows: list[tuple[str, Decimal, str]] = []
    for label, raw in zip(spec.labels, spec.values):
        parsed = parse_number(str(raw))
        if parsed is None:
            continue
        rows.append((str(label), parsed[0], str(raw)))
    if not rows:
        return ""
    width, height = 520, 220
    left, right, top, bottom = 36, 16, 28, 36
    plot_w = width - left - right
    plot_h = height - top - bottom
    amounts = [amount for _label, amount, _raw in rows]
    low = min(min(amounts), Decimal(0))
    high = max(max(amounts), Decimal(0))
    if high == low:
        high = low + Decimal(1)

    def y_of(amount: Decimal) -> float:
        return top + float((high - amount) / (high - low)) * plot_h

    zero = y_of(Decimal(0))
    kind = "line" if spec.kind in {"line", "趋势"} else "bar"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<text x="{left}" y="18" font-family="{_FONT_FAMILY},sans-serif" font-size="12">'
        f"{html.escape(spec.title)}</text>",
        f'<line x1="{left}" y1="{zero:.1f}" x2="{width - right}" y2="{zero:.1f}" stroke="#d0d5dd"/>',
    ]
    slot = plot_w / len(rows)
    coords: list[str] = []
    for index, (label, amount, raw) in enumerate(rows):
        x_mid = left + (index + 0.5) * slot
        y_val = y_of(amount)
        parts.append(
            f'<text x="{x_mid:.1f}" y="{height - 12}" text-anchor="middle" font-size="10" '
            f'font-family="{_FONT_FAMILY},sans-serif">{html.escape(label)}</text>'
        )
        parts.append(
            f'<text x="{x_mid:.1f}" y="{min(y_val, zero) - 4:.1f}" text-anchor="middle" '
            f'font-size="9" font-family="{_FONT_FAMILY},sans-serif">{html.escape(raw)}</text>'
        )
        if kind == "line":
            coords.append(f"{x_mid:.1f},{y_val:.1f}")
            continue
        bar_w = slot * 0.56
        x = x_mid - bar_w / 2
        y = min(y_val, zero)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{abs(zero - y_val) or 1:.1f}" '
            'fill="#1d4e89"/>'
        )
    if coords:
        parts.append(
            f'<polyline fill="none" stroke="#1d4e89" stroke-width="2" points="{" ".join(coords)}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


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
th, td {{ border: 0.3pt solid #d0d5dd; padding: 1.4mm 2mm; font-size: 9pt; vertical-align: top; }}
th {{ background: #f2f4f7; }}
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
    import fitz

    document = fitz.open(stream=content, filetype="pdf")
    try:
        return document.page_count
    finally:
        document.close()
