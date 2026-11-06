"""存储与外呼基础设施：连接池、全局 store 门面、认证、向量化。

从 memory 包析出的原因：`memory` 曾经同时装着「全局 DB 基础设施
（db / store 门面，被 kb、report、tools、api 跨 9 个包引用）」「用户
认证」「记忆功能」三种角色，导致 kb ↔ memory 在包级互相依赖，包名也
名不副实。析出后各包职责：

* storage —— 数据库连接池与 schema（db）、全局 store 单例（store，
  组装根，装配 kb 与 memory 的各 Store 类）、认证（auth_store）、
  向量化（embeddings）
* memory —— 只剩记忆功能：会话、画像、情景记忆、断点续跑、摘要
* kb / report / tools —— 只 import storage，不再 import memory

刻意不在 ``__init__`` 里 re-export 子模块符号（如 ``store`` 单例）：
导出名会遮蔽同名子模块，monkeypatch 按点分路径解析属性时会把
``ai_agent.storage.store`` 解析成单例对象而不是模块，路径再深一层
就断。调用方一律用完整路径 ``from ai_agent.storage.store import store``。
"""
