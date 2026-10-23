from ai_agent.memory.db import (
    DEFAULT_SESSION_TITLE,
    ChatHistoryPage,
    MemoryItem,
    SessionRecord,
    current_user_id,
    user_scope,
)
from ai_agent.memory.episode_store import EpisodeStore
from ai_agent.memory.profile_store import ProfileStore
from ai_agent.memory.sessions import SessionStore

__all__ = [
    "DEFAULT_SESSION_TITLE",
    "ChatHistoryPage",
    "MemoryItem",
    "MemoryStore",
    "SessionRecord",
    "current_user_id",
    "store",
    "user_scope",
]


class MemoryStore(EpisodeStore, ProfileStore, SessionStore):
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
