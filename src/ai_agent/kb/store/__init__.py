"""知识库文件与切片的 DB 存取。

与 CheckpointStore/EpisodeStore 同为 mixin，挂到 MemoryStore 上
共享连接池。

权限模型（P1）：知识库是**全站共享**的文档库——管理（列表/上传/删除）
只对管理员开放（路由层 require_admin 把关），普通用户只能在对话里通过
kb_search 工具检索内容。因此这里的读写方法都不按 user_id 过滤；
chunk.user_id 只作为「谁上传的」溯源信息保留。update_kb_status 供后台
管道用（不在任何 user_scope 里，按主键更新）。

实现按表域拆到本包子模块，KbStore 用多继承拼回来（与 storage/store.py
的 MemoryStore 同一套组合式写法）：

    files      KbFileStore      文件元数据、状态机、上传管道领取/恢复
    chunks     KbChunkStore     切片写入、向量回填、正文更新、按页定位
    search     KbSearchStore    文件/切片检索、关键词命中、词频
    neighbors  KbNeighborStore  按位置/seq 取片、前后邻域

40 个方法之间**没有 self 互调**（唯一共享的 self._get_pool() 来自
Database），所以拆分只是搬移，KbStore 的名字与方法对外不变；
``from ai_agent.kb.store import KbStore, _chunk_meta_where`` 与拆分前一致。
"""

from ai_agent.kb.store._helpers import (
    _INDICATOR_LIST,
    _SUBJECT_BUCKET,
    _chunk_indicators,
    _chunk_meta_where,
    _keyword_hit,
)
from ai_agent.kb.store.chunks import KbChunkStore
from ai_agent.kb.store.files import KbFileStore
from ai_agent.kb.store.neighbors import KbNeighborStore
from ai_agent.kb.store.search import KbSearchStore


class KbStore(
    KbFileStore,
    KbChunkStore,
    KbSearchStore,
    KbNeighborStore,
):
    """知识库存储（组合门面）。方法实现分散在四个 mixin 里。"""


__all__ = [
    "_INDICATOR_LIST",
    "_SUBJECT_BUCKET",
    "_chunk_indicators",
    "_chunk_meta_where",
    "_keyword_hit",
    "KbChunkStore",
    "KbFileStore",
    "KbNeighborStore",
    "KbSearchStore",
    "KbStore",
]
