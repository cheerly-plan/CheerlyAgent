"""知识图谱：实体/关系抽取、存储、对齐与查询。

graph_extract 按文档画像配置抽三元组；graph_store / graph_write 负责
存储与写合并；align 做实体对齐；kg_match 把问句实体匹配进图谱；
graph_names / graph_schema / graph_metrics / community 是命名规范、
schema 与社区检测辅助。图谱建在 ingest 的切片之上，不依赖 retrieval。
"""
