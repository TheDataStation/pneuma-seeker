from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


class StateManipulation(Action):
    action_name = ActionNames.STATE_MANIPULATION
    agents = frozenset({AgentType.CONDUCTOR})
    flag = None
    order = 3
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        return (
            f"**{ActionNames.STATE_MANIPULATION.value}**:\n"
            "  Update T, S, or both.\n"
            "  - **Args**:\n"
            '  {"T": {...}, "column_descriptions": {...}}\n'
            '  OR { "S": "..." }\n'
            '  OR {"T": {...}, "column_descriptions": {...}, "S": "..."}.\n'
            "  - **Notes**:\n"
            f"    - A `{ActionNames.STATE_MANIPULATION.value}` call resets previous T rather than appending."
        )
