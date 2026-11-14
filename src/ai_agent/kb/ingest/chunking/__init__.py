"""文档切片。

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

实现按职责拆到本包子模块，**依赖方向单一、无环**：

    common      叶子层：Chunk 数据类、HTML 表格正则、行项目判定、单元格转文本
      ↑
    html        行/列解析、rowspan 展开、HTML ↔ Markdown、超长表切分
      ↑
    text        纯文本聚合与二次切分        caption   表题识别与口径下带
      ↑                                        ↑
    table       行项目切片（表头重复、合计跟随）
      ↑
    preprocess  去噪声、合并小表、去页码样板、去重
      ↑
    core        chunk_segments 编排入口

调用方一律 ``from ai_agent.kb.ingest.chunking import X``。本文件把拆分前
模块里的**全部**符号原样再导出（含历史测试直接引用的私有符号），
因此导入路径与拆分前完全一致。
"""

from ai_agent.kb.ingest.chunking.caption import (
    _CAPTION_LINE,
    _UNIT_LINE,
    _caption_at_tail,
    _carry_md_table_captions,
    _scope_from_caption,
)
from ai_agent.kb.ingest.chunking.common import (
    _HTML_CELL,
    _HTML_COLSPAN,
    _HTML_ROW,
    _HTML_ROWSPAN,
    _HTML_TABLE,
    _MD_SEPARATOR,
    _NOT_HEADER_CELL,
    _NUM_ONLY_CELL,
    _TABLE_IFRS_SUB,
    _TABLE_SUB,
    _TABLE_TOTAL,
    Chunk,
    _cell_plain,
    _is_sub_row,
    _is_total_row,
    _markdown_table_plain,
    _md_cell,
    _md_cells,
    _md_row,
    _row_plain_line,
)
from ai_agent.kb.ingest.chunking.core import (
    TABLE_GROUP_CHARS,
    _SHEET_EXTS,
    chunk_segments,
    context_prefix,
    embedding_input,
)
from ai_agent.kb.ingest.chunking.html import (
    _buf_has_account,
    _can_merge_html_tables,
    _clamp_rowspans,
    _expand_html_grid,
    _html_chunk_kind,
    _html_header_row_count,
    _html_header_to_carry,
    _html_pieces,
    _html_row_is_clipped_header,
    _html_row_is_data,
    _html_row_is_header,
    _html_row_is_item,
    _html_tables_from_groups,
    _is_html_section_row,
    _merge_html_tables,
    _one_html_table_to_markdown,
    _row_texts,
    _row_width,
    _rows_len,
    _rowspan_continuations,
    _span_attr,
    _split_html_table,
    _split_html_table_grouped,
    _table_rows,
    html_table_plain,
    html_tables_to_markdown,
)
from ai_agent.kb.ingest.chunking.preprocess import (
    _BOILER_MAX_CHARS,
    _BOILER_MIN_CHARS,
    _BOILER_MIN_PAGES,
    _FOOTER_ONLY,
    _HEADING_LINE,
    _NO_CHART,
    _NO_CHART_MAX,
    _SIGNATURE_LINE,
    _SMALL_TABLE_PLAIN,
    _STANDALONE_ANSWER,
    _TABLE_LEAD_HEADING,
    _TABLE_LEAD_HINT,
    _TABLE_LEAD_MAX,
    _TOC_RESIDUE,
    _can_merge_small_tables,
    _carry_html_page_headers,
    _chunk_flow,
    _coalesce_segments,
    _drop_duplicate_chunks,
    _drop_noise,
    _drop_page_boilerplate,
    _expand_long_html_tables,
    _is_table_lead,
    _looks_like_boilerplate,
    _merge_table_segments,
)
from ai_agent.kb.ingest.chunking.table import (
    _INDICATOR_HEADER,
    _INDICATOR_LIMIT,
    _INDICATOR_NUM,
    _INDICATOR_SECTION,
    _INDICATOR_TOTAL,
    _chunk_html_table,
    _chunk_table,
    _clean_indicator,
    _demote_fake_md_header,
    _is_section_label,
    _lines_len,
    _md_first_cell,
    _md_row_cells,
    _merge_orphan_total_chunks,
    _pack_table_units,
    _pack_table_units_section,
    _table_label_cells,
    _table_row_units,
    extract_table_indicators,
)
from ai_agent.kb.ingest.chunking.text import (
    _PARA_SPLIT,
    _SENT_SPLIT,
    _chunk_plain_pages,
    _normalize,
    _plain_len,
    _same_page,
    _split_long,
    chunk_text,
)

__all__ = [
    "Chunk",
    "TABLE_GROUP_CHARS",
    "_BOILER_MAX_CHARS",
    "_BOILER_MIN_CHARS",
    "_BOILER_MIN_PAGES",
    "_CAPTION_LINE",
    "_FOOTER_ONLY",
    "_HEADING_LINE",
    "_HTML_CELL",
    "_HTML_COLSPAN",
    "_HTML_ROW",
    "_HTML_ROWSPAN",
    "_HTML_TABLE",
    "_INDICATOR_HEADER",
    "_INDICATOR_LIMIT",
    "_INDICATOR_NUM",
    "_INDICATOR_SECTION",
    "_INDICATOR_TOTAL",
    "_MD_SEPARATOR",
    "_NOT_HEADER_CELL",
    "_NO_CHART",
    "_NO_CHART_MAX",
    "_NUM_ONLY_CELL",
    "_PARA_SPLIT",
    "_SENT_SPLIT",
    "_SHEET_EXTS",
    "_SIGNATURE_LINE",
    "_SMALL_TABLE_PLAIN",
    "_STANDALONE_ANSWER",
    "_TABLE_IFRS_SUB",
    "_TABLE_LEAD_HEADING",
    "_TABLE_LEAD_HINT",
    "_TABLE_LEAD_MAX",
    "_TABLE_SUB",
    "_TABLE_TOTAL",
    "_TOC_RESIDUE",
    "_UNIT_LINE",
    "_buf_has_account",
    "_can_merge_html_tables",
    "_can_merge_small_tables",
    "_caption_at_tail",
    "_carry_html_page_headers",
    "_carry_md_table_captions",
    "_cell_plain",
    "_chunk_flow",
    "_chunk_html_table",
    "_chunk_plain_pages",
    "_chunk_table",
    "_clamp_rowspans",
    "_clean_indicator",
    "_coalesce_segments",
    "_demote_fake_md_header",
    "_drop_duplicate_chunks",
    "_drop_noise",
    "_drop_page_boilerplate",
    "_expand_html_grid",
    "_expand_long_html_tables",
    "_html_chunk_kind",
    "_html_header_row_count",
    "_html_header_to_carry",
    "_html_pieces",
    "_html_row_is_clipped_header",
    "_html_row_is_data",
    "_html_row_is_header",
    "_html_row_is_item",
    "_html_tables_from_groups",
    "_is_html_section_row",
    "_is_section_label",
    "_is_sub_row",
    "_is_table_lead",
    "_is_total_row",
    "_lines_len",
    "_looks_like_boilerplate",
    "_markdown_table_plain",
    "_md_cell",
    "_md_cells",
    "_md_first_cell",
    "_md_row",
    "_md_row_cells",
    "_merge_html_tables",
    "_merge_orphan_total_chunks",
    "_merge_table_segments",
    "_normalize",
    "_one_html_table_to_markdown",
    "_pack_table_units",
    "_pack_table_units_section",
    "_plain_len",
    "_row_plain_line",
    "_row_texts",
    "_row_width",
    "_rows_len",
    "_rowspan_continuations",
    "_same_page",
    "_scope_from_caption",
    "_span_attr",
    "_split_html_table",
    "_split_html_table_grouped",
    "_split_long",
    "_table_label_cells",
    "_table_row_units",
    "_table_rows",
    "chunk_segments",
    "chunk_text",
    "context_prefix",
    "embedding_input",
    "extract_table_indicators",
    "html_table_plain",
    "html_tables_to_markdown",
]
