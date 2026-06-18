from __future__ import annotations

import importlib
import inspect
import pkgutil
from logging import Logger
from typing import TYPE_CHECKING

from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType

if TYPE_CHECKING:
    pass

_IMPL_PACKAGE = "pneuma_seeker.services.core.action_set.impl"


class ActionRegistry:
    """Auto-discovers and instantiates every Action subclass in the impl package.

    To add a new action:
      1. Add its name to ActionNames.
      2. Create impl/<action_name>.py with a class that declares `action_name` and `agents`.
      3. Done — the registry picks it up automatically.
    """

    def __init__(
        self,
        user_id: str,
        chat_id: str,
        config: Config,
        logger: Logger,
        db_api: DBAPI,
        language_model_api: LanguageModelAPI,
    ) -> None:
        self._config = config
        self._instances: dict[ActionNames, Action] = {}
        self._discover_and_instantiate(
            user_id, chat_id, config, logger, db_api, language_model_api
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_and_instantiate(
        self,
        user_id: str,
        chat_id: str,
        config: Config,
        logger: Logger,
        db_api: DBAPI,
        language_model_api: LanguageModelAPI,
    ) -> None:
        import pneuma_seeker.services.core.action_set.impl as impl_pkg

        for _, module_name, _ in pkgutil.iter_modules(impl_pkg.__path__):
            full_name = f"{_IMPL_PACKAGE}.{module_name}"
            try:
                module = importlib.import_module(full_name)
            except Exception as exc:
                logger.warning(f"ActionRegistry: could not import {full_name}: {exc}")
                continue

            for _, cls in inspect.getmembers(module, inspect.isclass):
                if (
                    cls is Action
                    or not issubclass(cls, Action)
                    or not hasattr(cls, "action_name")
                    or not hasattr(cls, "agents")
                    or cls.action_name in self._instances  # already registered
                ):
                    continue
                try:
                    instance = cls(
                        user_id, chat_id, config, logger, db_api, language_model_api
                    )
                    self._instances[cls.action_name] = instance
                except Exception as exc:
                    logger.warning(
                        f"ActionRegistry: could not instantiate {cls.__name__}: {exc}"
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: ActionNames) -> Action:
        if name not in self._instances:
            raise KeyError(f"Action {name!r} is not registered")
        return self._instances[name]

    def get_for_agent(self, agent: AgentType, prompt_only: bool = True) -> list[Action]:
        """Return actions available to *agent*, sorted by their `order` attribute.

        Args:
            agent: The agent type to filter by.
            prompt_only: When True (default) only returns actions with
                ``show_in_prompt = True``.  Pass False to get all valid
                actions regardless (e.g. for action validation).
        """
        result: list[Action] = []
        for action in self._instances.values():
            if agent not in action.agents:
                continue
            if action.flag is not None and not getattr(
                self._config, action.flag, False
            ):
                continue
            if prompt_only and not action.show_in_prompt:
                continue
            result.append(action)
        return sorted(result, key=lambda a: a.order)

    def get_description(self, name: ActionNames, agent: AgentType | None = None) -> str:
        return self.get(name).get_description(agent)

    def is_valid_action(self, action_name: str, agent: AgentType) -> bool:
        """Return True when *action_name* is a registered action available to *agent*."""
        try:
            name = ActionNames(action_name)
        except ValueError:
            return False
        if name not in self._instances:
            return False
        action = self._instances[name]
        if agent not in action.agents:
            return False
        if action.flag is not None and not getattr(self._config, action.flag, False):
            return False
        return True
