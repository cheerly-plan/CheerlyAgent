"""按章节切块的原型（只读，不落库）。

目的：在动 schema 之前，先在真实 PDF 上验证「章节栅栏 + 页眉过滤」切出来的
切片是否干净、边界是否严格隔离。产出 JSON 预览 + 控制台校验报告。

与线上 chunking.py 的关系：
- 500 字聚合、超长段落按句切分（含 50 字重叠）的逻辑与 chunk_text() 保持一致，
  差别只在外层加了「一级节栅栏」：一片只属于一个节，绝不跨节；
- 线上是在整页文本上切、切完不知道属于哪一节；原型先按字号/位置识别标题，
  再在节内切，切片自带 section_path。

页眉是这份招股书最大的噪声源：215 行「第X节」里 203 行是每页重复的 10.5pt
页眉，和 16pt 的真标题文本一字不差。只按「文本重复」去重会把真标题一起误杀
（实测第四节、第十~十二节就是这么消失的），所以判据必须是
「字号 < 正文 + 贴顶/贴底」，文本重复只作为辅助。

用法：
    uv run python scripts/proto_section_chunks.py \
        --pdf data/uploads/xxx/南岭.pdf \
        --out data/proto/section_preview.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# ——— 判据 ———
# 一级节标题：第X节 / 第X章（招股书用「节」，公司章程类文档用「章」）
_SEC_L1 = re.compile(r"^第[一二三四五六七八九十百]+[节章]")
# 二级标题：一、二、…（原型只做识别与统计，不作切分边界）
_SEC_L2 = re.compile(r"^[一二三四五六七八九十]+、")
# 目录行的点线引导符：………… 或 …… 或 .....
_TOC_DOTS = re.compile(r"[.．·…]{3,}")
# 纯页码/页脚编号：1-1-141、- 12 -
_FOOTER_NO = re.compile(r"^[\s\-—–]*\d+([-—–]\d+)*[\s\-—–]*$")

TOP_RATIO = 0.07  # 页眉区：页面上部 7%（实测页眉 y=0.052，真标题 y≥0.086）
BOTTOM_RATIO = 0.90  # 页脚区：页面下部 10%（实测页码 y=0.93）
REPEAT_TOP_RATIO = 0.09  # 跨页重复文本只在贴顶时才当页眉
HEADING_SCALE = 1.2  # 标题字号至少是正文的 1.2 倍（12pt 正文 → 14.4pt 起）
MAX_TITLE_LEN = 40  # 标题不会太长


@dataclass
class Line:
    page: int
    text: str
    size: float
    y_ratio: float
    seq: int

    @property
    def norm(self) -> str:
        return re.sub(r"\s+", "", self.text)


@dataclass
class Section:
    section_id: str  # 第一节 / 前置
    title: str
    start_page: int
    end_page: int = 0
    heading_size: float = 0.0
    heading_y: float = 0.0
    lines: list[Line] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)


def extract_lines(doc: pymupdf.Document) -> list[Line]:
    """把每页文本拆成带字号、纵向位置的行。PDF 解析层将来要保留同样的元信息。"""
    out: list[Line] = []
    seq = 0
    for pno in range(doc.page_count):
        page = doc[pno]
        height = page.rect.height or 1.0
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"]).strip()
                if not text:
                    continue
                size = round(max(span["size"] for span in line["spans"]), 1)
                y_ratio = line["bbox"][1] / height
                out.append(Line(pno + 1, text, size, y_ratio, seq))
                seq += 1
    return out


def body_size(lines: list[Line]) -> float:
    """正文字号：按字符量加权取众数（招股书实测 12.0pt）。"""
    weight: Counter[float] = Counter()
    for ln in lines:
        weight[ln.size] += len(ln.text)
    return weight.most_common(1)[0][0] if weight else 12.0


def drop_headers_footers(
    lines: list[Line], body: float
) -> tuple[list[Line], dict, set[str]]:
    """剔除页眉页脚：贴顶/贴底 + 字号不大于正文；重复出现的顶部行也算页眉。

    注意不能用「文本重复」单独判定——真标题和页眉文本往往一模一样。
    """
    # 文本重复辅助判据：同一文本出现在 ≥3 页、且字号小于正文 1.05 倍
    pages_by_text: dict[str, set[int]] = defaultdict(set)
    for ln in lines:
        pages_by_text[ln.norm].add(ln.page)
    repeated = {
        t for t, pages in pages_by_text.items() if len(pages) >= 3 and len(t) > 2
    }

    # 第一遍：先认定「哪些文本是页眉」。判定只看第一条证据出现的形态，
    # 之后无论它被排版挤到页面哪个高度，都按页眉处理（实测有个别页的
    # 页眉掉到 y=0.117，纯位置规则会漏）
    header_texts: set[str] = set()
    for ln in lines:
        size_small = ln.size < body * 1.05
        if ln.y_ratio <= TOP_RATIO and size_small:
            header_texts.add(ln.norm)
        elif ln.norm in repeated and size_small and ln.y_ratio <= REPEAT_TOP_RATIO:
            header_texts.add(ln.norm)

    kept: list[Line] = []
    stats = Counter()
    for ln in lines:
        size_small = ln.size < body * 1.05
        if ln.y_ratio <= TOP_RATIO and size_small:
            stats["页眉(贴顶小字号)"] += 1
            continue
        if ln.norm in header_texts and size_small:
            stats["页眉(同文本异位)"] += 1
            continue
        if ln.y_ratio >= BOTTOM_RATIO and (size_small or _FOOTER_NO.match(ln.text)):
            stats["页脚(贴底/页码)"] += 1
            continue
        # 跨页重复只在贴顶时才算页眉：财务表的列头（"2005 年"）也会跨页重复，
        # 但它们在表格中部（实测 y≈0.17），一律按重复删会把表头数据吃光
        if ln.norm in repeated and size_small and ln.y_ratio <= REPEAT_TOP_RATIO:
            stats["页眉(跨页重复)"] += 1
            continue
        if _TOC_DOTS.search(ln.text) and size_small:
            stats["目录点线行"] += 1
            continue
        kept.append(ln)
    return kept, dict(stats), header_texts


_NUM = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8,
    "九": 9, "十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14,
    "十五": 15, "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
}


def _ordinal(section_id: str) -> int | None:
    m = re.match(r"^第([一二三四五六七八九十]+)[节章]$", section_id)
    return _NUM.get(m.group(1)) if m else None


def is_l1_heading(ln: Line, body: float) -> bool:
    return (
        bool(_SEC_L1.match(ln.norm))
        and len(ln.norm) <= MAX_TITLE_LEN
        and ln.size >= body * HEADING_SCALE
    )


def split_title(norm: str) -> tuple[str, str]:
    """「第十节管理层讨论与分析」→ ("第十节", "管理层讨论与分析")。"""
    m = re.match(r"^(第[一二三四五六七八九十百]+[节章])(.*)$", norm)
    if not m:
        return norm, ""
    return m.group(1), m.group(2)


def build_sections(lines: list[Line], body: float) -> list[Section]:
    sections: list[Section] = []
    current = Section("前置", "封面/目录/释义等", start_page=1)
    for ln in lines:
        if is_l1_heading(ln, body):
            sid, title = split_title(ln.norm)
            # 同一节标题重复出现（个别文档每页重复且字号也放大）时忽略
            if sections and sections[-1].section_id == sid:
                continue
            # 空章节也要收（扫描页章节只有标题行），否则章节树少一章
            current.end_page = max(ln.page, current.end_page)
            sections.append(current)
            current = Section(
                sid,
                title,
                start_page=ln.page,
                heading_size=ln.size,
                heading_y=ln.y_ratio,
            )
            continue
        current.lines.append(ln)
    current.end_page = max((ln.page for ln in current.lines), default=current.start_page)
    sections.append(current)
    # 扫描页章节（只有标题、正文是图片）也保留：章节树要完整，
    # 由调用方标记「无文本层」，而不是让它悄悄消失
    return [s for s in sections if s.lines or s.section_id != "前置"]


# ——— 节内切块：与 chunking.chunk_text 同语义，只多记页码 ———
_PARA_SPLIT = re.compile(r"\n+")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")


def _split_long(paragraph: str, page: int, max_chars: int, overlap: int) -> list[dict]:
    out: list[dict] = []
    buf = ""
    for sent in (s for s in _SENT_SPLIT.split(paragraph) if s.strip()):
        if len(sent) > max_chars:
            if buf:
                out.append({"text": buf, "start_page": page, "end_page": page})
                buf = ""
            step = max(max_chars - overlap, 1)
            for i in range(0, len(sent), step):
                piece = sent[i : i + max_chars]
                if len(piece) > overlap or i + max_chars >= len(sent):
                    out.append({"text": piece, "start_page": page, "end_page": page})
            continue
        if buf and len(buf) + len(sent) > max_chars:
            out.append({"text": buf, "start_page": page, "end_page": page})
            buf = buf[-overlap:] if overlap > 0 else ""
        buf += sent
    if buf.strip():
        out.append({"text": buf, "start_page": page, "end_page": page})
    return [c for c in out if c["text"].strip()]


def chunk_section(sec: Section, max_chars: int, overlap: int) -> list[dict]:
    """在单节范围内做 500 字聚合。缓冲不会跨越本节边界。"""
    chunks: list[dict] = []
    buf: list[str] = []
    buf_pages: list[int] = []
    length = 0

    def flush() -> None:
        nonlocal buf, buf_pages, length
        if buf:
            chunks.append(
                {
                    "text": "\n".join(buf).strip(),
                    "start_page": min(buf_pages),
                    "end_page": max(buf_pages),
                }
            )
            buf, buf_pages, length = [], [], 0

    for ln in sec.lines:
        para = ln.text.strip()
        if not para:
            continue
        if len(para) > max_chars:
            flush()
            chunks.extend(_split_long(para, ln.page, max_chars, overlap))
            continue
        if length + len(para) > max_chars and buf:
            flush()
        buf.append(para)
        buf_pages.append(ln.page)
        length += len(para)
    flush()
    return chunks


def validate(
    sections: list[Section],
    all_lines: list[Line],
    header_stats: dict,
    header_texts: set[str],
    body: float,
) -> dict:
    """四个硬指标 + 覆盖率，全部量化。"""
    l1 = [s for s in sections if s.section_id != "前置"]
    ids = [s.section_id for s in l1]
    # 招股书体例不统一：南岭用「第X节」、亚厦用「第X章」。只校验序数是否
    # 从 1 连续到 N，缺号即报（这样两种体例、任意节数都适用）
    ordinals = [o for o in (_ordinal(i) for i in ids) if o]
    missing = [
        f"第{n}个序号"
        for n in range(1, max(ordinals, default=0) + 1)
        if n not in ordinals
    ]
    order_ok = ordinals == sorted(ordinals) and not missing

    # 边界隔离：每片页码落在本节范围内 + 片内不含任何一级节标题
    cross_page: list[str] = []
    cross_heading: list[str] = []
    for sec in l1:
        for ch in sec.chunks:
            if ch["start_page"] < sec.start_page or ch["end_page"] > sec.end_page:
                cross_page.append(f"{sec.section_id}:{ch['start_page']}-{ch['end_page']}")
            if _SEC_L1.match(re.sub(r"\s+", "", ch["text"])):
                cross_heading.append(sec.section_id)

    # 页眉污染：片内是否残留被判定为页眉的行（按整行精确比对，
    # 避免正文引用「见第十节…」被误判）
    # 页眉污染：片内是否残留「第X节 …」类页眉（按整行精确比对，
    # 避免正文引用「见第十节…」被误判）。只统计正文节：前置区里的
    # 「目录」「释义」本身是内容标记，不算噪声
    noise_texts = {t for t in header_texts if _SEC_L1.match(t)}
    noisy = 0
    noisy_detail: list[dict] = []
    for sec in sections:
        if sec.section_id == "前置":
            for ch in sec.chunks:
                ch["has_header_noise"] = False
            continue
        for ch in sec.chunks:
            lines_in_chunk = {re.sub(r"\s+", "", s) for s in ch["text"].split("\n")}
            hit = lines_in_chunk & noise_texts
            ch["has_header_noise"] = bool(hit)
            noisy += int(bool(hit))
            if hit:
                noisy_detail.append(
                    {
                        "chunk_id": ch.get("chunk_id", "?"),
                        "pages": ch.get("page_range", ""),
                        "残留行": sorted(hit)[:3],
                    }
                )

    # 行级丢失：每个保留行的文本都必须能在本节的切片里找到。
    # 这是「章节栅栏没有吃掉正文」的硬保证
    lost_lines = 0
    for sec in sections:
        blob = re.sub(r"\s+", "", "".join(ch["text"] for ch in sec.chunks))
        lost_lines += sum(1 for ln in sec.lines if ln.norm not in blob)

    total_chars = sum(len(ln.text) for sec in sections for ln in sec.lines)
    kept_chars = sum(len(ch["text"]) for sec in sections for ch in sec.chunks)
    # 切片字符会比原始行字符多一点：行间补 \n、超长段落重叠 50 字。
    # 所以「覆盖率」不能用 切片/正文，否则永远 >1；这里给两个独立指标：
    #   · 丢弃字符占比：被页眉页脚规则删掉的正文占比（该是小的）
    #   · 切片膨胀率：切片字符 / 保留的行字符（应在 1.0~1.15）
    all_chars = sum(len(ln.text) for ln in all_lines)
    return {
        "一级节数量": len(l1),
        "一级节顺序正确": order_ok,
        "缺失的一级节": missing,
        "过滤掉的页眉页脚行": sum(header_stats.values()),
        "过滤明细": header_stats,
        "跨节切片数(页码越界)": len(cross_page),
        "片内含一级节标题的切片数": len(cross_heading),
        "含页眉噪声的切片数": noisy,
        "行级丢失数": lost_lines,
        "页眉残留明细": noisy_detail[:5],
        "丢弃字符占比(页眉页脚/目录)": round(1 - total_chars / all_chars, 4)
        if all_chars
        else 0,
        "切片膨胀率(换行+重叠)": round(kept_chars / total_chars, 4) if total_chars else 0,
        "字数最少的节": min(
            ((s.section_id, sum(len(ln.text) for ln in s.lines)) for s in l1),
            key=lambda kv: kv[1],
            default=("", 0),
        ),
        "正文字号": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="data/proto/section_preview.json")
    ap.add_argument("--max-chars", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    doc = pymupdf.open(args.pdf)
    lines = extract_lines(doc)
    body = body_size(lines)
    kept, header_stats, header_texts = drop_headers_footers(lines, body)
    sections = build_sections(kept, body)

    # 没有字号信息（扫描件/纯图片 PDF）时自动降级：原型只报告，不做假切分
    if len([s for s in sections if s.section_id != "前置"]) == 0:
        print("未能识别任何一级节标题：该 PDF 可能没有字号元信息，线上应降级回页面切分")

    for sec in sections:
        sec.chunks = chunk_section(sec, args.max_chars, args.overlap)
        sec.note = "" if sec.lines else "无文本层（扫描页，需 OCR 才能切块）"
        # 有正文的节按末行页码算结束页；纯扫描页章节没有行，就是标题所在页
        sec.end_page = max(ln.page for ln in sec.lines) if sec.lines else sec.start_page
        for i, ch in enumerate(sec.chunks, 1):
            ch["chunk_id"] = f"{sec.section_id}#{i:02d}"
            ch["section_path"] = (
                sec.section_id if sec.section_id == "前置" else f"{sec.section_id} {sec.title}"
            )
            ch["page_range"] = (
                f"{ch['start_page']}"
                if ch["start_page"] == ch["end_page"]
                else f"{ch['start_page']}-{ch['end_page']}"
            )

    report = validate(sections, lines, header_stats, header_texts, body)

    # 抽样：挑内容量差异大的三节，覆盖表格密集与文字密集
    pick = [s for s in sections if s.section_id in ("第三节", "第九节", "第十节")]
    pick = pick or [s for s in sections if s.section_id != "前置"][: args.samples]

    payload = {
        "source": Path(args.pdf).name,
        "params": {"max_chars": args.max_chars, "overlap": args.overlap},
        "validations": report,
        "sections": [
            {
                "section_id": s.section_id,
                "section_title": s.title,
                "start_page": s.start_page,
                "end_page": s.end_page,
                "heading_size": s.heading_size,
                "heading_y_ratio": round(s.heading_y, 3),
                "chunks_count": len(s.chunks),
                "note": getattr(s, "note", ""),
                "chars": sum(len(ln.text) for ln in s.lines),
                "sample_chunks": [
                    {
                        "chunk_id": ch["chunk_id"],
                        "content": ch["text"][:600],
                        "page_range": ch["page_range"],
                        "chars": len(ch["text"]),
                        "has_header_noise": ch["has_header_noise"],
                    }
                    for ch in s.chunks[:3]
                ],
            }
            for s in sections
        ],
        "sample_sections": [s.section_id for s in pick],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ——— 控制台报告 ———
    print(f"文件：{payload['source']}  总页数 {doc.page_count}  正文字号 {body}pt")
    print(f"\n{'节':<6}{'标题':<26}{'页码':<10}{'字数':>7}{'切片':>5}")
    for s in sections:
        print(
            f"{s.section_id:<6}{s.title[:24]:<26}{s.start_page}-{s.end_page:<6}"
            f"{sum(len(ln.text) for ln in s.lines):>7}{len(s.chunks):>5}"
        )
    print("\n校验：")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # 诊断：删掉的到底是不是页眉——按出现页数排前 10 的丢弃行
    kept_seq = {ln.seq for ln in kept}
    dropped = [ln for ln in lines if ln.seq not in kept_seq]
    by_text: dict[str, set[int]] = defaultdict(set)
    for ln in dropped:
        by_text[ln.norm].add(ln.page)
    top = sorted(by_text.items(), key=lambda kv: -len(kv[1]))[:10]
    print("\n删除最多的行（文本 → 出现页数）：")
    for text, pages in top:
        print(f"  {len(pages):>3}页  {text[:46]}")

    # 边界抽查：抽中节的「首片」与「末片」，肉眼确认没有越界
    print("\n边界抽查（首片/末片各截 120 字）：")
    for s in pick:
        if not s.chunks:
            continue
        first, last = s.chunks[0], s.chunks[-1]
        print(f"\n【{s.section_id} {s.title}】{len(s.chunks)} 片  p{s.start_page}-{s.end_page}")
        print(f"  首片 {first['chunk_id']} p{first['page_range']}: {first['text'][:120]}")
        print(f"  末片 {last['chunk_id']} p{last['page_range']}: {last['text'][-120:]}")
    print(f"\n预览已写入 {out}")


if __name__ == "__main__":
    main()
