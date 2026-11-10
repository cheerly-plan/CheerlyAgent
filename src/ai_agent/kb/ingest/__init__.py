"""文档摄取：原始文件 → 结构化切片。

parsing / figures / office / ocr / vision 负责按文件类型抽出正文与版面；
tables / table_cells / otsl / sections 把表格转成可检索文本；chunking /
lexicon / meta 完成切分与元数据；doc_profiles / summary 生成文档画像与
路由摘要。本子包是检索与图谱的公共底座，不反向依赖它们。
"""

from ai_agent.kb.ingest.chunking import extract_table_indicators

__all__ = ["extract_table_indicators"]
