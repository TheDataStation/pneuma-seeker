from collections import OrderedDict
from logging import Logger
from threading import Lock

from anyio import to_thread

from pneuma_seeker.chat_session import ChatSession
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.shared.config import Config


class SessionManager:
    """Manages chat sessions."""

    def __init__(self, config: Config, logger: Logger, pneuma_db: PneumaDB):
        """Initializes SessionManager."""
        self.config = config
        self.logger = logger
        self.pneuma_db = pneuma_db
        self.chat_sessions: OrderedDict[tuple[str, str], ChatSession] = OrderedDict()
        self._creation_lock = Lock()

    def get_chat_session(
        self, user_id: str, chat_id: str, dataset_name: str | None = None
    ) -> ChatSession:
        """
        Retrieves or creates a ChatSession, evicting the LRU entry if at capacity.

        Thread-safe via double-checked locking: the fast path (cache hit) avoids
        the lock entirely; the slow path (session creation) serialises under the
        lock to prevent double-loading when concurrent requests race for the same
        uncached session.  Call get_chat_session_async from async endpoints so
        that the potentially slow session-creation I/O runs in a thread pool
        rather than on the event loop.

        `dataset_name` only matters when creating a brand-new session — it's
        ignored on a cache hit, since an existing session's dataset is already
        fixed (see ChatSession.__init__).
        """
        key = (user_id, chat_id)

        # Fast path — GIL makes the individual dict operations atomic.
        if key in self.chat_sessions:
            self.chat_sessions.move_to_end(key)
            return self.chat_sessions[key]

        # Slow path — serialise creation so concurrent requests don't double-load.
        with self._creation_lock:
            if key in self.chat_sessions:
                self.chat_sessions.move_to_end(key)
                return self.chat_sessions[key]

            if len(self.chat_sessions) >= self.config.SESSION_MANAGER_MAX_SESSIONS:
                self.chat_sessions.popitem(last=False)

            self.chat_sessions[key] = ChatSession(
                user_id,
                chat_id,
                self.config,
                self.logger,
                DBAPI(self.config, self.logger, pneuma_db=self.pneuma_db),
                LanguageModelAPI(self.config, self.logger),
                dataset_name,
            )
            return self.chat_sessions[key]

    async def get_chat_session_async(
        self, user_id: str, chat_id: str, dataset_name: str | None = None
    ) -> ChatSession:
        """
        Event-loop-safe session retrieval for use inside async endpoint handlers.

        Cache hits return immediately on the event loop.  Cache misses delegate to
        a thread-pool worker so the heavy session-restore I/O (DuckDB reads, table
        loading) does not block the event loop and starve other in-flight requests.
        """
        key = (user_id, chat_id)
        if key in self.chat_sessions:
            return self.get_chat_session(user_id, chat_id, dataset_name)
        return await to_thread.run_sync(
            lambda: self.get_chat_session(user_id, chat_id, dataset_name)
        )

    def evict_chat_session(self, user_id: str, chat_id: str) -> None:
        """Removes a ChatSession from the in-memory cache, if present."""
        self.chat_sessions.pop((user_id, chat_id), None)
