from abc import ABC, abstractmethod
from logging import Logger
from typing import Any

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.shared.config import Config


class Action(ABC):
    def __init__(
        self,
        user_id: str,
        chat_id: str,
        config: Config,
        logger: Logger,
        db_api: DBAPI,
        language_model_api: LanguageModelAPI,
        **kwargs
    ):
        self.user_id = user_id
        self.chat_id = chat_id
        self.config = config
        self.logger = logger
        self.db_api = db_api
        self.language_model_api = language_model_api

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of the action."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Returns the description of the action."""
        pass

    @abstractmethod
    def get_input_schema(self) -> dict[str, str]:
        """Returns the input schema of the action."""
        pass

    @abstractmethod
    def get_notes(self) -> str:
        """Returns additional notes about the action."""
        pass


class Applicable(ABC):
    @abstractmethod
    def apply(self, input: dict[str, Any]) -> Any:
        """Applies the action with the given input and returns the output."""
        pass


class Executable(ABC):
    @abstractmethod
    def execute(self, input: dict[str, Any]) -> Any:
        """Executes the action with the given input and returns the output."""
        pass
