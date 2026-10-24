from threading import Event, Thread

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.memory.store import store

logger = get_logger(__name__)


class MaintenanceCleaner:
    def __init__(self) -> None:
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        interval = settings.session_cleanup_interval_seconds
        if interval <= 0:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(
            target=self._loop,
            name="empty-session-cleanup",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1)
        self._thread = None

    def _loop(self) -> None:
        interval = settings.session_cleanup_interval_seconds
        while not self._stop.wait(interval):
            try:
                deleted = store.cleanup_empty_unnamed_sessions(
                    max_age_seconds=settings.session_empty_max_age_seconds
                )
                if deleted:
                    logger.info("清理空会话 %s 条", deleted)
            except Exception:
                logger.exception("清理空会话失败")
            try:
                stale = store.cleanup_stale_checkpoints(stale_seconds=1800)
                if stale:
                    logger.info("清理僵尸 checkpoint %s 个", stale)
            except Exception:
                logger.exception("清理僵尸 checkpoint 失败")
            try:
                # failed/interrupted 只为断点续跑服务，7 天后删除防累积
                expired = store.cleanup_old_checkpoints(max_age_seconds=7 * 24 * 3600)
                if expired:
                    logger.info("清理过期 checkpoint %s 个", expired)
            except Exception:
                logger.exception("清理过期 checkpoint 失败")


maintenance_cleaner = MaintenanceCleaner()
