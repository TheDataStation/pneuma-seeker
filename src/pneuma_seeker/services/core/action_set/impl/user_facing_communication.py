from typing import Any

from pneuma_seeker.services.core.action_set.interfaces import Action, Executable
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


class UserFacingCommunication(Action, Executable):
    action_name = ActionNames.USER_FACING_COMMUNICATION
    agents = frozenset({AgentType.CONDUCTOR})
    flag = None
    order = 99
    show_in_prompt = False

    def get_description(self, agent: AgentType | None = None) -> str:
        return (
            f"**{ActionNames.USER_FACING_COMMUNICATION.value}**: Send a message to the user."
            '\n  - **Args**: {"message": "<your message to the user>"}'
        )

    def execute(self, input: dict[str, Any]) -> str:
        message = input.get("message")
        if not isinstance(message, str):
            raise ValueError("`args` must be an object with a `message` property")
        if not message.strip():
            raise ValueError("`message` must be a non-empty string")
        return message
