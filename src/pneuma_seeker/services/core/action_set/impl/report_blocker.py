from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


class ReportBlocker(Action):
    """Lets the Materializer stop when T cannot be built from the tables it was given."""

    action_name = ActionNames.REPORT_BLOCKER
    agents = frozenset({AgentType.MATERIALIZER})
    order = 98

    def get_description(self, agent: AgentType | None = None) -> str:
        return (
            f"**{ActionNames.REPORT_BLOCKER.value}**\n"
            "    - Stops materialization when T cannot be built from the provided tables, e.g., a column's source "
            "table is missing, or the provided tables lack the values a column needs.\n"
            "    - Use it only after checking the provided tables; the caller will revise T and its column sources.\n"
            '    - Args: {"reason": "<what is missing, and which T columns it affects>"}'
        )
