"""知识库检索编排：向量召回 → 去冗余 → LLM 重排 → 邻接扩展。

只负责"取哪几段",不负责拼给模型的文案（那是 tools/kb_search 的事）——
图谱检索、对话工具、以后的评测可以共用同一套检索策略。

四级处理各自的必要性：
1. 召回放大到 candidates：直接取最近邻 3 条，常常全是同一段话的变体，
   真正回答问题的片段排在第四位；
2. 去冗余：切片有 overlap，相邻切片内容本来就高度重合，按三字组
   Jaccard 去重比引入 MMR + 向量再计算便宜得多；
3. 重排：向量相似度只看"像不像"，重排看"能不能回答问题"，后者更准；
4. 邻接扩展：切片按长度切，跨段落的结论会被切断，命中片前后各带一段。

实现按职责拆到 ``_impl/``（依赖方向单一、无环）：

    types      叶子层：Hit、format_where、_where、停用词
      ↑
    textutils  _clip / _trigrams / _jaccard
    subjects   主体与文件解析        keywords   关键词扩展
      ↑                                 ↑
    neighbors  邻接扩展              fusion      RRF 融合与去冗余
              ↘                     ↙
                 rerank  精排
                   ↑
                 search  search_knowledge / preview_search 编排

**本模块是对外唯一入口**，拆分前模块里的所有符号（含私有）都在这里原样
再导出，所以 ``from ai_agent.kb.retrieval.retrieval import X`` 与
``monkeypatch.setattr(retrieval, "X", ...)`` 的写法都不变。
"""

from __future__ import annotations

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb.ingest.chunking import TABLE_GROUP_CHARS, html_table_plain
from ai_agent.kb.ingest.lexicon import load_short_words, read_extra
from ai_agent.kb.ingest.meta import name_in_text
from ai_agent.kb.ingest.meta import ENTITY_SUFFIX as _ENTITY_SUFFIX
from ai_agent.kb.ingest.meta import EXT_RE as _EXT_RE
from ai_agent.kb.ingest.meta import FILENAME_NOISE as _FILENAME_NOISE
from ai_agent.kb.ingest.meta import NAME_PUNCT as _NAME_PUNCT
from ai_agent.kb.ingest.otsl import render_otsl
from ai_agent.kb.ingest.parsing import FIGURE, TABLE
from ai_agent.kb.ingest.tables import SCOPE_CONSOLIDATED, SCOPE_PARENT
from ai_agent.kb.retrieval._impl.fusion import (
    _DUP_THRESHOLD,
    _PARENT_INTENT,
    _RRF_K,
    _RRF_KEYWORD_WEIGHT,
    _RRF_KEYWORD_WEIGHT_FULL,
    _SCOPE_MATCH,
    _SCOPE_MISMATCH,
    _apply_scope_preference,
    _dedupe,
    _diversify,
    _diversify_by_file,
    _diversify_by_subject,
    _focused_file_ids,
    _fuse_with_keywords,
    _keep_distinctive_keywords,
    _prefer_scope,
    _rrf_merge,
    _to_hit,
    _vector_distance,
)
from ai_agent.kb.retrieval._impl.keywords import (
    _SHORT_WORD,
    _adjacent_short_phrases,
    _apply_short_phrases,
    _extract_keywords,
)
from ai_agent.kb.retrieval._impl.neighbors import (
    _CELL,
    _FIRST_ROW,
    _MERGE_MAX_CHARS,
    _MERGE_MAX_GAP,
    _attach_neighbors,
    _fetch_merged_gap,
    _join_table_texts,
    _merge_same_table_rows,
    _table_header,
)
from ai_agent.kb.retrieval._impl.rerank import (
    _JSON_ARRAY,
    _rerank_cache,
    _RERANK_CACHE_MAX,
    _RERANK_LIMIT,
    _boost_indicator_hits,
    _candidate_text,
    _llm_rerank_scores,
    _parse_order,
    _rerank,
    complete_chat_once,
)
from ai_agent.kb.retrieval._impl.search import (
    _core_search,
    _retrieve_candidates,
    preview_search,
    search_knowledge,
    search_knowledge_multi,
)
from ai_agent.kb.retrieval._impl.subjects import (
    _clean_filename_hint,
    _file_subject,
    _filename_hint_matches,
    _subject_core,
    _subject_token_matches,
    mentioned_entities,
    mentioned_subjects,
    query_looks_on_topic,
    resolve_files_by_name,
)
from ai_agent.kb.retrieval._impl.textutils import _RERANK_CLIP, _clip, _jaccard, _trigrams
from ai_agent.kb.retrieval._impl.types import (
    _SHEET_EXTS,
    _STOPWORDS,
    Hit,
    _where,
    format_where,
    is_sheet_file,
)
from ai_agent.kb.retrieval.rerank import get_reranker
from ai_agent.prompts.kb import build_rerank_user
from ai_agent.storage.embeddings import embed_query
from ai_agent.storage.store import store

logger = get_logger(__name__)

__all__ = [
    "FIGURE",
    "Hit",
    "SCOPE_CONSOLIDATED",
    "SCOPE_PARENT",
    "TABLE",
    "TABLE_GROUP_CHARS",
    "_CELL",
    "_DUP_THRESHOLD",
    "_ENTITY_SUFFIX",
    "_EXT_RE",
    "_FILENAME_NOISE",
    "_FIRST_ROW",
    "_JSON_ARRAY",
    "_MERGE_MAX_CHARS",
    "_MERGE_MAX_GAP",
    "_NAME_PUNCT",
    "_PARENT_INTENT",
    "_RERANK_CACHE_MAX",
    "_RERANK_CLIP",
    "_RERANK_LIMIT",
    "_RRF_K",
    "_RRF_KEYWORD_WEIGHT",
    "_RRF_KEYWORD_WEIGHT_FULL",
    "_SCOPE_MATCH",
    "_SCOPE_MISMATCH",
    "_SHEET_EXTS",
    "_SHORT_WORD",
    "_STOPWORDS",
    "_adjacent_short_phrases",
    "_apply_scope_preference",
    "_apply_short_phrases",
    "_attach_neighbors",
    "_boost_indicator_hits",
    "_candidate_text",
    "_clean_filename_hint",
    "_clip",
    "_core_search",
    "_dedupe",
    "_diversify",
    "_diversify_by_file",
    "_diversify_by_subject",
    "_extract_keywords",
    "_fetch_merged_gap",
    "_file_subject",
    "_filename_hint_matches",
    "_focused_file_ids",
    "_fuse_with_keywords",
    "_jaccard",
    "_join_table_texts",
    "_keep_distinctive_keywords",
    "_llm_rerank_scores",
    "_merge_same_table_rows",
    "_parse_order",
    "_prefer_scope",
    "_rerank",
    "_rerank_cache",
    "_retrieve_candidates",
    "_rrf_merge",
    "_subject_core",
    "_subject_token_matches",
    "_table_header",
    "_to_hit",
    "_trigrams",
    "_vector_distance",
    "_where",
    "build_rerank_user",
    "complete_chat_once",
    "embed_query",
    "format_where",
    "get_logger",
    "get_reranker",
    "html_table_plain",
    "is_sheet_file",
    "load_short_words",
    "mentioned_entities",
    "mentioned_subjects",
    "name_in_text",
    "preview_search",
    "query_looks_on_topic",
    "read_extra",
    "render_otsl",
    "resolve_files_by_name",
    "search_knowledge",
    "search_knowledge_multi",
    "settings",
    "store",
]
