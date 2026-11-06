"""全局 store 门面：装配各域 Store，提供唯一单例。

位置说明：这里是「组装根」（composition root）——import kb 与 memory 的
Store 类是装配行为，不是底层依赖；kb / memory 的内部模块 import 本模块
的单例时不会形成模块级环（被 import 的 kb.store / graph_store 等类模块
自身不回头 import 这里）。历史上它住在 memory 包里，导致 kb ↔ memory
在包级互相依赖、且「记忆」包名不副实（认证、连接池都在里面），故移入
storage 与 db / auth_store / embeddings 同层。
"""

from ai_agent.kb.graph.graph_store import KgStore
from ai_agent.kb.graph.graph_write import KgWriteStore
from ai_agent.kb.store import KbStore
from ai_agent.memory.checkpoint_store import CheckpointStore
from ai_agent.memory.episode_store import EpisodeStore
from ai_agent.memory.profile_store import ProfileStore
from ai_agent.memory.session_store import SessionStore
from ai_agent.storage.auth_store import AuthStore
from ai_agent.storage.db import (
    DEFAULT_SESSION_TITLE,
    ChatHistoryPage,
    MemoryItem,
    SessionRecord,
    UserRecord,
    current_user_id,
    user_scope,
)

__all__ = [
    "DEFAULT_SESSION_TITLE",
    "ChatHistoryPage",
    "MemoryItem",
    "MemoryStore",
    "SessionRecord",
    "UserRecord",
    "current_user_id",
    "store",
    "user_scope",
]


class MemoryStore(
    AuthStore,
    EpisodeStore,
    ProfileStore,
    SessionStore,
    CheckpointStore,
    KbStore,
    KgStore,
    KgWriteStore,
):
    def clear_long_term(self, user_id: str | None = None) -> None:
        with self._get_pool().connection() as conn:
            if user_id:
                conn.execute("DELETE FROM agent_episode WHERE user_id = %s", (user_id,))
                conn.execute(
                    "DELETE FROM agent_user_profile WHERE user_id = %s", (user_id,)
                )
            else:
                conn.execute("DELETE FROM agent_episode")
                conn.execute("DELETE FROM agent_user_profile")
            conn.commit()


store = MemoryStore()
