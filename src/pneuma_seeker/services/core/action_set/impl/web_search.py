from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


class WebSearch(Action):
    action_name = ActionNames.WEB_SEARCH
    agents = frozenset({AgentType.CONDUCTOR, AgentType.MATERIALIZER})
    flag = "ENABLE_WEB_SEARCH"
    order = 14
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        return f"""**{ActionNames.WEB_SEARCH.value}**
    - Retrieves information from the web when internal and external tables are insufficient.
    - Args: {{"prompt": "<query describing what data to retrieve>"}}
    - Usage notes:
        - Use only when no reliable internal/external source exists for the required column(s).
        - Avoid repetitive or redundant queries."""

    def get_input_schema(self) -> dict[str, str]:
        return {
            "prompt": "The input query to search the web.",
            "k": "The number of results to retrieve.",
            "sample_only": "Whether to sample the results only.",
            "sample_size": "The size of the sample if sample_only is True.",
        }

    def get_notes(self) -> str:
        return "This action uses the IR system to search the web."
