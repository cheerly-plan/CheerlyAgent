"""文档切片（编排入口）。

按职责拆到本包的兄弟模块：common / text / html / caption /
table / preprocess。逻辑与拆分前逐字一致。

两级策略：先按段落聚合到目标长度；单段超长时按句子边界（中英文
标点）二次切分。相邻切片保留 overlap 字符，避免关键信息恰好落在
边界上被截断。纯字符数切分（中文 1 字符 ≈ 1 token 量级），不上
语义切分——收益配不上复杂度。

按内容类型分派：
- text：上面的常规策略
- table：一个行项目一片，每片重复表头、表题和单位。「其中：」、加减行和
  合计行跟所属科目走，不单独成片。正文里的 HTML `<table>` 先整张抽出，
  kind 记成 table。续页没印列名时补上一页的列名，表题同样往下带。
  向量和重排读的是单元格文字，不读标签。给模型的检索和抽取把 HTML 表
  换成 Markdown 表，库里仍留 HTML，前端合并单元格不用重切。
- figure：图像描述本身很短，直接沿用常规策略（一般不触发切分）

两级栅栏，谁窄听谁的：
- **节栅栏**：同节的连续正文段先合成一条流再聚合，一片不跨章节。
  章节是语义的自然边界（"第十节 管理层讨论与分析"里不会混进"第五节
  风险因素"的句子），跨节拼接只会让引用张冠李戴；
- **页边界**：无章节信息的文档（扫描件、MD/TXT、Excel）仍按段独立切，
  不做跨页合并——页码是溯源的最低要求。

页码溯源：解析层产出 Segment（带 page / section / kind），切片后每片
携带自己的页码区间与类型（PDF 页码、Excel sheet 序号、或 None）。
"""

from pathlib import Path

from ai_agent.kb.ingest.parsing import FIGURE, Segment, TABLE

from ai_agent.kb.ingest.chunking.caption import (
    _carry_md_table_captions,
    _scope_from_caption,
)

from ai_agent.kb.ingest.chunking.common import (
    Chunk,
)

from ai_agent.kb.ingest.chunking.html import (
    html_table_plain,
)

from ai_agent.kb.ingest.chunking.preprocess import (
    _carry_html_page_headers,
    _chunk_flow,
    _coalesce_segments,
    _drop_duplicate_chunks,
    _drop_noise,
    _drop_page_boilerplate,
    _expand_long_html_tables,
)

from ai_agent.kb.ingest.chunking.table import (
    _chunk_table,
)

from ai_agent.kb.ingest.chunking.text import (
    _chunk_plain_pages,
)


def chunk_segments(
    segments: list[Segment],
    *,
    max_chars: int = 500,
    overlap: int = 50,
    within_page: bool = False,
    table_group_mode: str = "row",
) -> list[Chunk]:
    """按节成流、按段兜底：同节连续正文合并后聚合，表与无章节段独立处理。

    `within_page=True` 时同节内也按页断（一片只落一页，溯源最精确，但
    跨页的长论述会被切断）；默认 False，允许同节内跨页聚合。
    """
    chunks: list[Chunk] = []
    flow: list[Segment] = []
    plain: list[Segment] = []

    def flush_flow() -> None:
        if flow:
            chunks.extend(
                _chunk_flow(flow, max_chars=max_chars, overlap=overlap)
            )
            flow.clear()

    def flush_plain() -> None:
        if plain:
            chunks.extend(
                _chunk_plain_pages(plain, max_chars=max_chars, overlap=overlap)
            )
            plain.clear()

    for seg in _carry_md_table_captions(
        _carry_html_page_headers(
            _drop_page_boilerplate(
                _coalesce_segments(_drop_noise(segments), max_chars=max_chars)
            )
        )
    ):
        if seg.kind == TABLE:
            # 表必须整表成片（超长时重复表头），不能揉进正文流
            flush_plain()
            flush_flow()
            chunks.extend(
                Chunk(
                    text=piece,
                    page=seg.page,
                    kind=seg.kind,
                    page_end=seg.page_end if seg.page_end is not None else seg.page,
                    section=seg.section,
                    # 解析阶段没标出 scope 的（表题原本在上一页、靠跨页补题才
                    # 补上来的），这里从切片首行补：母公司/合并口径直接决定
                    # 「问母公司某科目」时这个片会不会被 SCOPE_PARENT 过滤掉。
                    scope=seg.scope or _scope_from_caption(piece),
                )
                for piece in _chunk_table(
                    seg.text, max_chars=max_chars, group_mode=table_group_mode
                )
            )
        elif seg.section is None:
            # 无章节信息：页与页的正文不合并。紧挨着的 HTML 表在
            # _chunk_plain_pages 里按列数接成一张，并记上 page_end。
            flush_flow()
            plain.append(seg)
        else:
            # 两条栅栏：换节必断；开了单页栅栏时换页也断。
            # 断点处不保留重叠——重叠只发生在同一片流内部（超长段落二次
            # 切分时），所以重叠永远不会跨节、跨页（when within_page）。
            flush_plain()
            if flow and (
                flow[0].section != seg.section
                or (within_page and flow[-1].page != seg.page)
            ):
                flush_flow()
            flow.append(seg)
    flush_plain()
    flush_flow()
    return _drop_duplicate_chunks(
        _expand_long_html_tables(
            chunks, max_chars=max_chars, group_mode=table_group_mode
        )
    )

def context_prefix(
    filename: str, *, page: int | None, kind: str, section: str | None = None
) -> str:
    """切片的语境前缀：文档名 + 位置 + 章节。

    切片单看常常缺主语（"同比增长 56%" 是哪家公司的？哪一年的？），
    向量化时把这份语境一起嵌进去，召回才分得清同名指标；同时也让
    模型引用时有据可依。存储的正文保持干净——前缀只在嵌入时拼。
    """
    stem = Path(filename).stem if filename else "文档"
    suffix = Path(filename or "").suffix.lower()
    if kind == TABLE:
        if page is None:
            where = "表格"
        elif suffix in _SHEET_EXTS:
            where = f"表{page}"
        else:
            where = f"第{page}页"
    elif kind == FIGURE:
        where = f"第{page}页图表" if page is not None else "图表"
    else:
        where = f"第{page}页" if page is not None else ""
    base = f"《{stem}》{where}" if where else f"《{stem}》"
    return f"{base} · {section}" if section else base

def embedding_input(
    text: str,
    *,
    filename: str,
    page: int | None,
    kind: str,
    section: str | None = None,
) -> str:
    """送进向量模型的文本 = 语境前缀 + 正文。

    章节用完整标签（一级 + 二级）：试过只取一级，MRR 从 0.879 掉到 0.869
    ——二级标题里的「非经常性损益」「股东信息」恰恰是把数字表区分开的那几个
    字，砍掉反而更难找。

    HTML 表先换成单元格文字再嵌入。标签占掉的向量维度不区分「货币资金」
    和「应收账款」，整张表合成一个点时更是把要找的那一行稀释掉。
    """
    body = html_table_plain(text)
    return f"{context_prefix(filename, page=page, kind=kind, section=section)}\n{body}"

# 整页 OCR 的表是一整段 HTML。按 500 字硬切会切断 </td>，所以整表抽出，
# 再按行项目切开。这个数是重排读正文的窗口，不是一行组能装多少字。
TABLE_GROUP_CHARS = 800

# Excel 的 page 是 sheet 序号；PDF/Word 的 page 是页码。文档里的「表格16」
# 是作者自己编的表号，解析层不会去数，引用只能落到页。
_SHEET_EXTS = {".xlsx", ".xlsm", ".xls", ".csv"}
