from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


class TableRetrieve(Action):
    action_name = ActionNames.TABLE_RETRIEVE
    agents = frozenset({AgentType.CONDUCTOR, AgentType.MATERIALIZER})
    flag = None
    order = 1
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        return (
            f"**{ActionNames.TABLE_RETRIEVE.value}**:\n"
            "    Retrieve internal tables.\n"
            '    - **Args**: {"prompts": ["<retrieval query 1>", "<retrieval query 2>", ...]}\n'
            "    - **Notes**:\n"
            f"        - You may provide multiple retrieval queries in a single call to retrieve tables on different topics (at most {self.config.TABLE_RETRIEVE_MAX_TOPICS} topics).\n"
            "        - If available, include specific keywords or entities in each query to improve retrieval precision.\n"
            "        - Previously retrieved tables will be replaced with new retrievals.\n"
            "        - Potential join paths between retrieved tables will be provided for reference."
        )
