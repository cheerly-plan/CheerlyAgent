"""章节识别与节内切片的单测。

都是纯函数级用例，用合成行数据构造「页眉与真标题文本一字不差」这种
真实场景——正是原型阶段踩坑最狠的地方，必须回归住。
"""

from ai_agent.kb.chunking import Chunk, chunk_segments, chunk_text
from ai_agent.kb.parsing import FIGURE, TEXT, Segment, _label_at, _merge_wrapped
from ai_agent.kb.sections import (
    Line,
    Section,
    body_size,
    group_by_section,
    is_l1_heading,
    strip_noise,
)

BODY = 12.0


def line(text: str, page: int = 1, size: float = BODY, y: float = 0.3) -> Line:
    return Line(text, page, size, y)


def test_body_size_picks_dominant_size():
    lines = [line("正文" * 50)] + [line("标题", size=16.0) for _ in range(3)]
    assert body_size(lines) == BODY


def test_heading_requires_font_scale():
    # 同样文本：页眉字号（10.5pt）不算标题，16pt 才算——这是识别成败的关键
    assert not is_l1_heading(line("第十节 管理层讨论与分析", size=10.5), BODY)
    assert is_l1_heading(line("第十节 管理层讨论与分析", size=16.0), BODY)


def test_heading_ignores_body_text_mentioning_section():
    # 正文里引用「见第十节」不应被当成标题（长度 + 字号双重约束）
    assert not is_l1_heading(line("详见第十节管理层讨论与分析相关内容", size=16.0), BODY)


def test_strip_noise_removes_repeated_top_header():
    lines = []
    for page in (1, 2, 3):
        lines.append(line("湖南南岭民用爆破器材股份有限公司 招股意向书", page, 10.5, 0.05))
        lines.append(line(f"第{page}页正文内容", page, BODY, 0.3))
    kept, stats = strip_noise(lines, BODY)
    assert len(kept) == 3
    assert all("招股意向书" not in ln.text for ln in kept)
    assert stats["页眉(贴顶小字号)"] == 3


def test_strip_noise_keeps_table_column_headers():
    # 财务表列头跨页重复，但在页面中部（y≈0.17）：不能当页眉删，否则表头没了
    lines = []
    for page in (1, 2, 3):
        lines.append(line("2005 年", page, 10.5, 0.17))
        lines.append(line("营业收入 1,234", page, BODY, 0.30))
    kept, _ = strip_noise(lines, BODY)
    assert sum(1 for ln in kept if ln.text == "2005 年") == 3


def test_strip_noise_removes_misplaced_header_copy():
    # 实测有个别页的页眉被排版挤到 y=0.117：纯位置规则会漏，靠「同文本」兜底
    lines = [
        line("第十二节 募股资金运用", 1, 10.5, 0.05),
        line("正文一", 1, BODY, 0.3),
        line("第十二节 募股资金运用", 2, 10.5, 0.117),
        line("正文二", 2, BODY, 0.3),
    ]
    kept, _ = strip_noise(lines, BODY)
    assert [ln.text for ln in kept] == ["正文一", "正文二"]


def test_group_by_section_splits_at_headings():
    lines = [
        line("封面文字"),
        line("第一节 释义", size=16.0, page=2),
        line("释义正文", page=2),
        line("第二节 概览", size=16.0, page=3),
        line("概览正文", page=3),
    ]
    sections = group_by_section(lines, BODY)
    assert [s.id for s in sections] == ["", "第一节", "第二节"]
    assert sections[1].title == "释义"
    assert sections[1].path == "第一节 释义"
    assert sections[1].start_page == 2


def test_group_by_section_keeps_scan_only_section():
    # 扫描页章节只有标题行（正文是图片），不能让它从章节树里消失
    lines = [
        line("第十四节 附录", size=16.0, page=10),
        line("附录正文", page=10),
        line("第十五节 签字页", size=16.0, page=11),
    ]
    sections = group_by_section(lines, BODY)
    assert [s.id for s in sections][-2:] == ["第十四节", "第十五节"]
    assert sections[-1].lines == []


def test_group_by_section_without_headings_returns_single_group():
    lines = [line("扫描件文字", 1), line("第二页", 2)]
    sections = group_by_section(lines, BODY)
    assert len(sections) == 1 and sections[0].id == ""


def test_cn_numbering_becomes_l1_when_no_section_headings():
    """季报没有「第X节」：一级就是「一、」，二级是「（一）」。

    实测三份季报的 206 个切片一个章节名都没有，图谱 prompt 只能给页码。
    """
    lines = [
        line("封面文字", 1),
        line("一、主要财务数据", 2),
        line("正文一", 2),
        line("（一）主要会计数据和财务指标", 2, size=10.4),
        line("正文二", 2),
        line("二、股东信息", 3),
    ]
    secs = group_by_section(lines, 10.4)
    labels = [s.label for s in secs if s.label]
    assert labels == [
        "一、主要财务数据",
        "一、主要财务数据 · （一）主要会计数据和财务指标",
        "二、股东信息",
    ]


def test_cn_numbering_becomes_sub_when_section_headings_exist():
    # 招股书里「第十节」是一级，「一、」是它下面的细分，不能提成同级
    lines = [
        line("第十节 管理层讨论与分析", page=2, size=16.0),
        line("一、主营业务", page=2, size=14.0),
        line("正文", page=2),
    ]
    secs = group_by_section(lines, BODY)
    assert secs[-1].path == "第十节 管理层讨论与分析"
    assert secs[-1].label == "第十节 管理层讨论与分析 · 一、主营业务"


def test_cn_heading_rejects_sentence_like_lines():
    # 报表行「一、经营活动产生的现金流量：」和带逗号的整句都不是标题
    lines = [
        line("一、经营活动产生的现金流量：", 1),
        line("二、公司实现营业收入 12 亿元，同比增长", 1),
    ]
    secs = group_by_section(lines, BODY)
    assert len(secs) == 1 and secs[0].label == ""


def test_label_keeps_narrow_space_after_cn_marker():
    sec = Section("一、", "主要财务数据", start_page=2, sub="（一）主要会计数据")
    assert sec.path == "一、主要财务数据"
    assert sec.label == "一、主要财务数据 · （一）主要会计数据"


def boxed(text: str, *, x0: float = 56.0, x1: float = 530.0, y0: float = 100.0) -> Line:
    return Line(text, 1, BODY, y0 / 800, (x0, y0, x1, y0 + 14))


def test_merge_wrapped_joins_line_broken_mid_sentence():
    # 「占总营」+「收94.84%」被版面折成两行：不接回去，切片边界就落在词中间
    lines = [
        boxed("公司境外实现营业收入247.91 亿元，同比增长21.99%，占总营"),
        boxed("收94.84%；境内实现营业收入13.50 亿元。"),
    ]
    assert _merge_wrapped(lines) == [
        "公司境外实现营业收入247.91 亿元，同比增长21.99%，占总营收94.84%；"
        "境内实现营业收入13.50 亿元。"
    ]


def test_merge_wrapped_keeps_paragraph_ending_with_punctuation():
    lines = [boxed("第一段结束。"), boxed("第二段开始，还没有结束")]
    assert _merge_wrapped(lines) == ["第一段结束。", "第二段开始，还没有结束"]


def test_merge_wrapped_does_not_join_two_columns():
    # 封面「证券代码：…」贴左、「公告编号：…」贴右，x0 不对就不能接
    lines = [
        boxed("证券代码：300866", x0=300.0, x1=530.0),
        boxed("债券代码：123257"),
    ]
    assert _merge_wrapped(lines) == ["证券代码：300866", "债券代码：123257"]


def test_merge_wrapped_does_not_glue_heading_to_body():
    lines = [
        boxed("（一）主要会计数据和财务指标"),
        boxed("本报告期营业收入 9,535,167,697.58 元。"),
    ]
    assert _merge_wrapped(lines) == [
        "（一）主要会计数据和财务指标",
        "本报告期营业收入 9,535,167,697.58 元。",
    ]


def test_label_at_picks_nearest_heading_and_carries_across_pages():
    markers = [(2, 100.0, "一、主要财务数据"), (2, 400.0, "一、主要财务数据 · （一）x")]
    assert _label_at(markers, 2, 350.0) == "一、主要财务数据"
    assert _label_at(markers, 2, 500.0) == "一、主要财务数据 · （一）x"
    # 表格页没有正文行：沿用上一页最后一个标题，而不是丢掉
    assert _label_at(markers, 3, 90.0) == "一、主要财务数据 · （一）x"


def test_chunk_segments_merges_pages_within_section():
    segments = [
        Segment("甲" * 300, 1, TEXT, "第一节 释义"),
        Segment("乙" * 300, 2, TEXT, "第一节 释义"),
    ]
    chunks = chunk_segments(segments, max_chars=700, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].page == 1 and chunks[0].page_end == 2
    assert chunks[0].section == "第一节 释义"


def test_chunk_segments_never_crosses_sections():
    segments = [
        Segment("甲" * 300, 1, TEXT, "第一节 释义"),
        Segment("乙" * 300, 2, TEXT, "第二节 概览"),
    ]
    chunks = chunk_segments(segments, max_chars=700, overlap=0)
    assert len(chunks) == 2
    assert all("甲" not in c.text or "乙" not in c.text for c in chunks)
    assert [c.section for c in chunks] == ["第一节 释义", "第二节 概览"]


def test_chunk_segments_keeps_table_out_of_text_flow():
    segments = [
        Segment("甲" * 100, 1, TEXT, "第一节 释义"),
        Segment("| 年 | 收入 |\n| --- | --- |\n| 2005 | 12 |", 1, "table", "第一节 释义"),
        Segment("乙" * 100, 1, TEXT, "第一节 释义"),
    ]
    chunks = chunk_segments(segments, max_chars=500, overlap=0)
    assert [c.kind for c in chunks] == ["text", "table", "text"]


def test_chunk_segments_without_section_keeps_page_isolation():
    # 无章节信息：段与段不合并，页码语义不变（老行为）
    segments = [Segment("甲" * 100, 1), Segment("乙" * 100, 2)]
    chunks = chunk_segments(segments, max_chars=700, overlap=0)
    assert len(chunks) == 2
    assert all(c.section is None for c in chunks)


def test_flow_chunk_respects_max_chars_with_newline():
    # 换行符也占长度：50 段 10 字拼起来是 50*10+49 字符，不能刚好卡在 500
    paras = "\n".join("字" * 10 for _ in range(60))
    pieces = chunk_text(paras, max_chars=100, overlap=0)
    assert all(len(p) <= 100 for p in pieces)


def test_flow_chunk_splits_long_paragraph_by_sentence():
    segments = [Segment("句子。" * 200, 1, TEXT, "第一节 释义")]
    chunks = chunk_segments(segments, max_chars=100, overlap=0)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(isinstance(c, Chunk) and c.section for c in chunks)


def test_within_page_mode_breaks_at_page_even_if_short():
    # 单页栅栏打开：宁可留一片 200 字的短片，也不跨页拼
    segments = [
        Segment("甲" * 200, 1, TEXT, "第一节 释义"),
        Segment("乙" * 200, 2, TEXT, "第一节 释义"),
    ]
    chunks = chunk_segments(segments, max_chars=500, overlap=50, within_page=True)
    assert [c.page for c in chunks] == [1, 2]
    assert all(c.page == c.page_end for c in chunks)


def test_within_page_mode_still_respects_section_fence():
    segments = [
        Segment("甲" * 200, 1, TEXT, "第一节 释义"),
        Segment("乙" * 200, 1, TEXT, "第二节 概览"),
    ]
    chunks = chunk_segments(segments, max_chars=500, overlap=0, within_page=True)
    assert [c.section for c in chunks] == ["第一节 释义", "第二节 概览"]



def test_noise_figure_description_is_dropped():
    # 视觉模型对「送审但没图」的页面会照格式编结论，这类片段不能进库
    segments = [
        Segment("无图表。", 195, FIGURE),
        Segment("无图表或图片", 196, FIGURE),
        Segment("图表类型：无图表 标题：无 坐标轴含义与单位：无", 199, FIGURE),
        Segment("图 1 装机容量趋势：2004-2006 年逐年上升。", 30, FIGURE),
    ]
    chunks = chunk_segments(segments, max_chars=500, overlap=0)
    assert len(chunks) == 1
    assert "装机容量" in chunks[0].text


def test_noise_filter_keeps_real_figure_text_mentioning_none():
    # 真描述里带了"无坐标轴"这类词不能误杀：只要不是整句否认有图就保留
    segments = [Segment("截图表；签字盖章页 无坐标轴 无关键数值", 192, FIGURE)]
    assert len(chunk_segments(segments, max_chars=500, overlap=0)) == 1


def test_toc_residue_chunk_is_dropped():
    segments = [
        Segment("目 录", 4),
        Segment("目\t录", 4),
        Segment("目录结构说明：本文档分为八节。", 4),
    ]
    chunks = chunk_segments(segments, max_chars=500, overlap=0)
    assert [c.text for c in chunks] == ["目录结构说明：本文档分为八节。"]
