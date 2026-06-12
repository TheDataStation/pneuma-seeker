# src/pneuma_seeker/chat_session.py
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
        self.conductor = Conductor(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            ProvenanceGraph(self.logger),
            self.db_api,
            self.language_model_api,
        )

        (
            messages,
            conductor_state,
            provenance_graph,
            retrieved_tables,
            enumerated_tables,
            web_search_result,
            web_crawl_result,
            join_paths,
            dataset_name,
        ) = self.db_api.load_session(
            self.user_id,
            self.chat_id,
        )
        self.messages = messages
        self.dataset_name = dataset_name
        self.conductor.state = conductor_state
        self.conductor.set_prov_graph(provenance_graph)
        self.conductor.retrieved_tables = retrieved_tables
        self.conductor.enumerated_tables = enumerated_tables
        self.conductor.web_search_result = web_search_result
        self.conductor.web_crawl_result = web_crawl_result
        self.conductor.join_paths = join_paths

    def chat(
        self,
        latest_user_message: str,
        external_data_paths: list[str] | None = None,
    ):
        """Processes chat messages and yields responses from the Conductor."""
        external_data_paths = external_data_paths or []
        interaction_history: list[UserConductorInteraction] = []

        latest_user_message = latest_user_message.strip()
        if not latest_user_message:
            raise ValueError("No user message provided for chat session.")

        self.messages.append(
            LLMMessage(
                role=Role.USER.value,
                content=latest_user_message,
            )
        )

        for i in range(0, len(self.messages) - 1, 2):
            if (
                self.messages[i]["role"] == Role.USER.value
                and self.messages[i + 1]["role"] == Role.ASSISTANT.value
            ):
                interaction_history.append(
                    UserConductorInteraction(
                        self.messages[i]["content"],
                        self.messages[i + 1]["content"],
                    )
                )

        full_response = ""
        for conductor_response in self.conductor.chat(
            latest_user_message,
            interaction_history,
            external_data_paths,
        ):
            full_response = conductor_response
            yield conductor_response

        if full_response:
            self.messages.append(
                LLMMessage(
                    role=Role.ASSISTANT.value,
                    content=full_response,
                )
            )
        yield "DONE"

    def persist_session(self, dataset_name: str):
        """Callback to persist the current state of Provenance Graph."""
        try:
            self.__log(f"Persisting session...")
            self.__log(
                f"=> Number of provenance nodes: {len(self.conductor.prov_graph.nodes)}"
            )

            last_user_input = ""
            last_system_response = ""
            if self.messages:
                # If the last message is from the assistant, the one before it was the user prompt
                if self.messages[-1]["role"] == Role.ASSISTANT.value:
                    last_system_response = self.messages[-1]["content"]
                    if len(self.messages) > 1:
                        last_user_input = self.messages[-2]["content"]
                # If it cut off right after the user sent something, but before assistant finished
                elif self.messages[-1]["role"] == Role.USER.value:
                    last_user_input = self.messages[-1]["content"]

            self.dataset_name = dataset_name
            self.db_api.persist_session(
                self.user_id,
                self.chat_id,
                dataset_name,
                last_user_input,
                last_system_response,
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
