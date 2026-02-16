# backend/services/core-service/src/core_service/chat_session.py
from logging import Logger

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.conductor.main import Conductor
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.conductor import UserConductorInteraction
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.role import Role


class ChatSession:
    """Represents a chat session for a user."""

    def __init__(
        self,
        user_id: str,
        chat_id: str,
        config: Config,
        logger: Logger,
        db_api: DBAPI,
        language_model_api: LanguageModelAPI,
    ):
        """Initializes the ChatSession with user and chat IDs, configuration, logger, and APIs."""
        self.user_id = user_id
        self.chat_id = chat_id
        self.config = config
        self.logger = logger
        self.db_api = db_api
        self.language_model_api = language_model_api
        self.messages: list[LLMMessage] = []
        self.__last_user_input: str = ""
        self.__last_system_response: str = ""

        self.conductor = Conductor(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            ProvenanceGraph(self.logger),
            self.db_api,
            self.language_model_api,
        )

        if self.config.PERSIST_CHAT_SESSION:
            (
                messages,
                conductor_state,
                provenance_graph,
                retrieved_tables,
                enumerated_tables,
                web_search_result,
                web_crawl_result,
                join_paths,
            ) = self.db_api.load_session(
                self.user_id,
                self.chat_id,
            )
            self.messages = messages
            self.conductor.state = conductor_state
            self.conductor.set_prov_graph(provenance_graph)
            self.conductor.retrieved_tables = retrieved_tables
            self.conductor.enumerated_tables = enumerated_tables
            self.conductor.web_search_result = web_search_result
            self.conductor.web_crawl_result = web_crawl_result
            self.conductor.join_paths = join_paths

    def chat(
        self,
        messages: list[LLMMessage] | None = None,
        external_data_paths: list[str] | None = None,
    ):
        """Processes chat messages and yields responses from the Conductor."""
        external_data_paths = external_data_paths or []
        interaction_history: list[UserConductorInteraction] = []

        if not messages and len(self.messages) == 0:
            raise ValueError("No messages provided for chat session.")
        if not messages:
            messages = self.messages
        else:
            self.messages = messages

        self.__last_user_input = messages[-1]["content"]
        for i in range(0, len(messages) - 1, 2):
            if (
                messages[i]["role"] == Role.USER.value
                and messages[i + 1]["role"] == Role.ASSISTANT.value
            ):
                interaction_history.append(
                    UserConductorInteraction(
                        messages[i]["content"],
                        messages[i + 1]["content"],
                    )
                )

        for conductor_response in self.conductor.chat(
            messages[-1]["content"],
            interaction_history,
            external_data_paths,
        ):
            self.__last_system_response = conductor_response
            yield conductor_response

        yield "DONE"

    def persist_session(self):
        """Callback to persist the current state of Provenance Graph."""
        try:
            self.__log(f"Persisting session...")
            self.__log(f"=> Number of provenance nodes: {len(self.conductor.prov_graph.nodes)}")
            self.db_api.persist_session(
                self.user_id,
                self.chat_id,
                self.__last_user_input,
                self.__last_system_response,
                self.conductor.state,
                self.conductor.prov_graph,
                self.conductor.retrieved_tables,
                self.conductor.enumerated_tables,
                self.conductor.web_search_result,
                self.conductor.web_crawl_result,
                self.conductor.join_paths,
            )
            self.__log("Session persisted successfully.")
        except Exception as e:
            self.__log(f"Failed to persist session: {e}")

    def __log(self, message: str):
        self.logger.info(f"[ChatSession] {message}")
