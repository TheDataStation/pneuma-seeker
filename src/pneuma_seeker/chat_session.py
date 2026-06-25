# src/pneuma_seeker/chat_session.py
from logging import Logger

from pneuma_seeker.provenance.graph import ProvenanceGraph
from pneuma_seeker.services.core.conductor.models import (
    ConductorResponse,
    ConductorResponseType,
)
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.services.core.conductor.main import Conductor
from pneuma_seeker.services.skill_based_core.agent import SkillsAgent
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import formatted_log
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
        """Initializes the ChatSession with user and chat IDs, config, logger, and APIs."""
        self.user_id = user_id
        self.chat_id = chat_id
        self.config = config
        self.logger = logger
        self.db_api = db_api
        self.language_model_api = language_model_api

        _agent_cls = SkillsAgent if config.USE_SKILLS_AGENT else Conductor
        self.conductor = _agent_cls(
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
        user_message: str,
        external_table_paths: list[str] | None = None,
    ):
        """Processes a user message and yields responses from Conductor."""
        external_table_paths = external_table_paths or []

        user_message = user_message.strip()
        if not user_message:
            raise ValueError("No user message provided for chat session.")

        self.messages.append(
            LLMMessage(
                role=Role.USER.value,
                content=user_message,
            )
        )

        final_response = ""
        for conductor_response in self.conductor.chat(
            user_message,
            self.messages[:-1],
            external_table_paths,
        ):
            if conductor_response.type == ConductorResponseType.FINAL_RESPONSE:
                final_response = conductor_response.message
            yield conductor_response

        if final_response:
            self.messages.append(
                LLMMessage(
                    role=Role.ASSISTANT.value,
                    content=final_response,
                )
            )
        yield ConductorResponse(ConductorResponseType.DONE, "")

    def persist_session(self, dataset_name: str):
        """Persists the current state of the chat session to the database."""
        try:
            self.__log(f"Persisting session...")
            last_user_message = ""
            last_conductor_response = ""
            if self.messages:
                # If the last message is from the assistant, the one before it was the user prompt
                if self.messages[-1]["role"] == Role.ASSISTANT.value:
                    last_conductor_response = self.messages[-1]["content"]
                    if len(self.messages) > 1:
                        last_user_message = self.messages[-2]["content"]
                # If it cut off right after the user sent something, but before assistant finished
                elif self.messages[-1]["role"] == Role.USER.value:
                    last_user_message = self.messages[-1]["content"]

            self.dataset_name = dataset_name
            self.db_api.persist_session(
                self.user_id,
                self.chat_id,
                dataset_name,
                last_user_message,
                last_conductor_response,
                self.conductor.state,
                self.conductor.prov_graph,
                self.conductor.retrieved_tables,
                self.conductor.enumerated_tables,
                self.conductor.web_search_result,
                self.conductor.web_crawl_result,
                self.conductor.join_paths,
            )
            self.__log("Session persisted successfully!")
        except Exception as e:
            self.__log(f"Failed to persist session: {e}")

    def __log(self, message: str):
        formatted_log(self.logger, "ChatSession", message)
