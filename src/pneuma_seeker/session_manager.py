# src/pneuma_seeker/session_manager.py
from collections import OrderedDict
from logging import Logger

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

    def get_chat_session(self, user_id: str, chat_id: str) -> ChatSession:
        """Retrieves or creates a ChatSession, evicting the LRU entry if at capacity."""
        key = (user_id, chat_id)
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
        )
        return self.chat_sessions[key]

    def evict_chat_session(self, user_id: str, chat_id: str) -> None:
        """Removes a ChatSession from the in-memory cache, if present."""
        self.chat_sessions.pop((user_id, chat_id), None)
