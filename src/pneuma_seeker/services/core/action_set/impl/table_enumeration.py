from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.services.core.dataset_context import MULTI_DATASET_MODE
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


class TableEnumeration(Action):
    action_name = ActionNames.TABLE_ENUMERATION
    agents = frozenset({AgentType.CONDUCTOR})
    flag = "ENABLE_TABLE_ENUMERATION"
    order = 2
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        description = self._base_description()
        if MULTI_DATASET_MODE in self.active_modes:
            description += self._multi_dataset_note()
        return description

    def _multi_dataset_note(self) -> str:
        return (
            "\n    - Multi-dataset: optionally add "
            '`"datasets": ["<dataset>", ...]` to the args to choose where to enumerate; '
            "it defaults to the datasets of the tables already retrieved."
        )

    def _base_description(self) -> str:
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
