from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


class SituationalAnalysis(Action):
    action_name = ActionNames.SITUATIONAL_ANALYSIS
    agents = frozenset({AgentType.CONDUCTOR, AgentType.MATERIALIZER})
    flag = None
    order = 99
    show_in_prompt = False

    def get_description(self, agent: AgentType | None = None) -> str:
        return (
            f"**{ActionNames.SITUATIONAL_ANALYSIS.value}**: Analyze the current environment "
            "and plan the next sequence of actions."
            '\n  - **Args**: {"message": "<your analysis>"}'
        )
