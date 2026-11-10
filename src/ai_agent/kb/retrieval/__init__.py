"""检索：混合召回、精排、文档路由、范围裁定与引用组装。

``retrieval`` 是知识库检索的公开入口包。子模块按职责拆分：
``retrieval``（向量/关键词混合召回与融合）、``rerank``（精排）、
``router``（按文件名/主题路由到具体文件）、``scope``（会话级检索范围与
多轮改写上下文）、``citations``（引用收集与出处合并）。

包根只再导出对外公开的符号——调用方与测试应当 ``from ai_agent.kb.retrieval
import search_knowledge, mentioned_subjects``，而不是钻进内部子模块；这样内部
子模块重排不会破坏外部引用。再导出的都是函数/类，不会遮蔽同名子模块。
"""

from ai_agent.kb.retrieval.citations import (
    Citation,
    add_citations,
    collect_citations,
    merge_sources,
)
from ai_agent.kb.retrieval.retrieval import (
    Hit,
    format_where,
    mentioned_entities,
    mentioned_subjects,
    preview_search,
    query_looks_on_topic,
    resolve_files_by_name,
    search_knowledge,
    search_knowledge_multi,
)
from ai_agent.kb.retrieval.rerank import get_reranker, probe_reranker
from ai_agent.kb.retrieval.router import mentioned_file_ids, route_files
from ai_agent.kb.retrieval.scope import (
    effective_kb_query,
    resolve_tool_file_ids,
)

__all__ = [
    "Hit",
    "Citation",
    "search_knowledge",
    "search_knowledge_multi",
    "preview_search",
    "mentioned_subjects",
    "mentioned_entities",
    "mentioned_file_ids",
    "resolve_files_by_name",
    "route_files",
    "format_where",
    "query_looks_on_topic",
    "effective_kb_query",
    "resolve_tool_file_ids",
    "get_reranker",
    "probe_reranker",
    "collect_citations",
    "add_citations",
    "merge_sources",
]
