from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


class UserFacingCommunication(Action):
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
