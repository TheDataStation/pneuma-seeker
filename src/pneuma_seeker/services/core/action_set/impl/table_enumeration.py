from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


class TableEnumeration(Action):
    action_name = ActionNames.TABLE_ENUMERATION
    agents = frozenset({AgentType.CONDUCTOR, AgentType.MATERIALIZER})
    flag = None
    order = 2
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        if agent == AgentType.MATERIALIZER:
            return self._materializer_description()
        return self._conductor_description()

    def _conductor_description(self) -> str:
        return (
            f"**{ActionNames.TABLE_ENUMERATION.value}**:\n"
            "  List other available internal tables in the database whose names match given regex patterns.\n"
            f'  - **Args**: {{"patterns": ["<regex pattern 1>", "<regex pattern 2>", ...]}}\n'
            "  - **Notes**:\n"
            "    - **Precondition — MUST NOT be called unless there is at least one internal table already retrieved.**\n"
            f"    - Each pattern in `patterns` **must be derived from the names of existing internal tables** (or obvious common tokens in them).\n"
            f"    - Enumerated tables will be unioned with internal tables retrieved via `{ActionNames.TABLE_RETRIEVE.value}`.\n"
            "    - Returns names only (not data), but `{ActionNames.MATERIALIZER.value}` will access the actual data.\n"
            "    - This is useful when you retrieve one table (e.g., `topic_2020`) but suspect there are other related tables (`topic_2021`, `topic_2022`, etc.)\n"
            f"    - You may provide multiple patterns in a single call (at most {self.config.TABLE_RETRIEVE_MAX_TOPICS} patterns).\n"
            '    - Example: {"patterns": ["^sales_\\\\d{4}$", "^revenue_\\\\d{4}$"]} will match all tables named like `sales_2020`, `sales_2021`, etc., and `revenue_2020`, `revenue_2021`, etc.'
        )

    def _materializer_description(self) -> str:
        return (
            f"**{ActionNames.TABLE_ENUMERATION.value}**\n"
            "    - **Precondition — MUST NOT be called unless there is at least one internal table already retrieved.**\n"
            "    - Each pattern in `patterns` **must be derived from the names of existing internal tables** (or obvious common tokens in them).\n"
            "    - Lists other available internal tables in the database whose names match given regex patterns.\n"
            "    - This is useful when you retrieve one table (e.g., `topic_2020`) but suspect there are other related tables (`topic_2021`, `topic_2022`, etc.)\n"
            f"    - Enumerated tables will be unioned with internal tables retrieved via Table Retrieve.\n"
            "    - Does not affect user-provided external data.\n"
            f'    - Args: {{"patterns": ["<regex pattern 1>", "<regex pattern 2>", ...]}}\n'
            "    - Notes:\n"
            f"        - You may provide multiple patterns in a single call (at most {self.config.TABLE_RETRIEVE_MAX_TOPICS} patterns).\n"
            '    - Example: {"patterns": ["^sales_\\\\d{4}$", "^revenue_\\\\d{4}$"]} will match all tables named like `sales_2020`, `sales_2021`, etc., and `revenue_2020`, `revenue_2021`, etc.'
        )
