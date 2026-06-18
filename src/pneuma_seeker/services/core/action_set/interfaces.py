from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger
from typing import TYPE_CHECKING, Any

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.shared.config import Config

if TYPE_CHECKING:
    from pneuma_seeker.shared.schemas.core.action import ActionNames
    from pneuma_seeker.shared.schemas.core.agent import AgentType


class Action(ABC):
    """Base class for all actions.

    Subclasses must declare:
        action_name: ActionNames   — the enum member for this action
        agents: frozenset[AgentType] — which agents may invoke this action

    Subclasses may override:
        flag: str | None    — Config attribute that must be True for the action to be enabled
        order: int          — sort key used when building agent prompts (lower = earlier)
        show_in_prompt: bool — whether the action description should appear in agent prompts
    """

    # --- class-level declarations (override in every subclass) ---
    action_name: "ActionNames"
    agents: "frozenset[AgentType]"

    # --- optional class-level overrides ---
    flag: str | None = None
    order: int = 99
    show_in_prompt: bool = True

    def __init__(
        self,
        user_id: str,
        chat_id: str,
        config: Config,
        logger: Logger,
        db_api: DBAPI,
        language_model_api: LanguageModelAPI,
        **kwargs: Any,
    ) -> None:
        self.user_id = user_id
        self.chat_id = chat_id
        self.config = config
        self.logger = logger
        self.db_api = db_api
        self.language_model_api = language_model_api

    @abstractmethod
    def get_description(self, agent: "AgentType | None" = None) -> str:
        """Return the prompt description for this action.

        Override with agent-specific text when the description differs between
        Conductor and Materializer (e.g. different result-table names).
        """

    def get_name(self) -> str:
        return self.action_name.value

    def get_input_schema(self) -> dict[str, str]:
        return {}

    def get_notes(self) -> str:
        return ""


class Applicable(ABC):
    @abstractmethod
    def apply(self, input: dict[str, Any]) -> Any:
        """Apply the action and return its output."""


class Executable(ABC):
    @abstractmethod
    def execute(self, input: dict[str, Any]) -> Any:
        """Execute the action and return its output."""
