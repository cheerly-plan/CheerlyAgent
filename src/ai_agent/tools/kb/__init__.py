"""知识库工具：文件清单、混合检索、图谱检索。

``kb_list`` 列知识库文件；``kb_search`` 走混合检索（文档路由 + 向量/关键词
召回 + 精排 + 引用）；``kg_search`` 走知识图谱实体/关系查询。
巨潮补齐不进对话工具——由研报流水线在缺原文时自动触发。
"""

from ai_agent.tools.kb.kb_list import kb_list_files_tool
from ai_agent.tools.kb.kb_search import kb_search_tool
from ai_agent.tools.kb.kg_search import kg_search_tool

__all__ = ["kb_list_files_tool", "kb_search_tool", "kg_search_tool"]
