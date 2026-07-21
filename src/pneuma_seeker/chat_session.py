# src/pneuma_seeker/chat_session.py
from logging import Logger
from typing import Callable

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
        dataset_name: str | None = None,
    ):
        """Initializes the ChatSession with user and chat IDs, config, logger, and APIs.

        `dataset_name` only seeds a brand-new chat — an existing chat's persisted
        dataset always takes precedence, since a chat's dataset is fixed once established.
        """
        self.user_id = user_id
        self.chat_id = chat_id
        self.config = config
        self.logger = logger
        self.db_api = db_api
        self.language_model_api = language_model_api

        (
            messages,
            conductor_state,
            provenance_graph,
            retrieved_tables,
            enumerated_tables,
            web_search_result,
            web_crawl_result,
            join_paths,
            persisted_dataset_name,
        ) = self.db_api.load_session(
            self.user_id,
            self.chat_id,
        )

        self.messages = messages
        self.dataset_name = persisted_dataset_name or dataset_name or ""

        _agent_cls = SkillsAgent if config.USE_SKILLS_AGENT else Conductor
        self.conductor = _agent_cls(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            ProvenanceGraph(self.logger),
            self.db_api,
            self.language_model_api,
            frontend_callback=lambda _: None,
        )

        self.conductor.state = conductor_state
        self.conductor.set_prov_graph(provenance_graph)
        self.conductor.retrieved_tables = retrieved_tables
        self.conductor.enumerated_tables = enumerated_tables
        self.conductor.web_search_result = web_search_result
        self.conductor.web_crawl_result = web_crawl_result
        self.conductor.join_paths = join_paths

    def set_dataset_name(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    def chat(
        self,
        user_message: str,
        external_table_paths: list[str] | None = None,
        frontend_callback: Callable[[ConductorResponse], None] = lambda _: None,
        plan_mode: bool = False,
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

        self.conductor.frontend_callback = frontend_callback
        final_response = ""
        final_response_is_plan_proposal = False
        for conductor_response in self.conductor.chat(
            user_message,
            self.messages[:-1],
            external_table_paths,
            plan_mode=plan_mode,
            dataset_name=self.dataset_name,
        ):
            if conductor_response.type in (
                ConductorResponseType.FINAL_RESPONSE,
                ConductorResponseType.PLAN_PROPOSAL,
            ):
                final_response = conductor_response.message
                final_response_is_plan_proposal = (
                    conductor_response.type == ConductorResponseType.PLAN_PROPOSAL
                )
            yield conductor_response

        if final_response:
            self.messages.append(
                LLMMessage(
                    role=Role.ASSISTANT.value,
                    content=final_response,
                    is_plan_proposal=final_response_is_plan_proposal,
                )
            )
        yield ConductorResponse(ConductorResponseType.DONE, "")

    def persist_session(self, dataset_name: str):
        """Persists the current state of the chat session to the database."""
        try:
            self.__log(f"Persisting session...")
            last_user_message = ""
            last_conductor_response = ""
            last_response_is_plan_proposal = False
            if self.messages:
                # If the last message is from the assistant, the one before it was the user prompt
                if self.messages[-1]["role"] == Role.ASSISTANT.value:
                    last_conductor_response = self.messages[-1]["content"]
                    last_response_is_plan_proposal = self.messages[-1].get(
                        "is_plan_proposal", False
                    )
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
                last_response_is_plan_proposal,
            )
            self.__log("Session persisted successfully!")
        except Exception as e:
            self.__log(f"Failed to persist session: {e}")

    def __log(self, message: str):
        formatted_log(self.logger, "ChatSession", message)
