from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.services.core.dataset_context import MULTI_DATASET_MODE
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


class TableRetrieve(Action):
    action_name = ActionNames.TABLE_RETRIEVE
    agents = frozenset({AgentType.CONDUCTOR})
    flag = None
    order = 1
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        multi = MULTI_DATASET_MODE in self.active_modes
        datasets_arg = ', "datasets": ["<dataset 1>", ...]' if multi else ""
        multi_notes = (
            "        - Every query is run against every dataset in `datasets`. `datasets` is optional and defaults to "
            f"the chat's default dataset (if any). If you don't know which dataset holds the data, call "
            f"`{ActionNames.DATASET_LISTING.value}` first.\n"
            '        - Each table is shown with its dataset. In SQL, refer to it as `<dataset>."<table>"`.\n'
            if multi
            else ""
        )
        return (
            f"**{ActionNames.TABLE_RETRIEVE.value}**:\n"
            "    Retrieve internal tables, by search and/or by name.\n"
            f'    - **Args**: {{"prompts": ["<retrieval query 1>", ...], "tables": ["<table name>", ...]{datasets_arg}}} (at least one of `prompts` and `tables`)\n'
            "    - **Notes**:\n"
            f"        - You may provide multiple retrieval queries in a single call to retrieve tables on different topics (at most {self.config.TABLE_RETRIEVE_MAX_TOPICS} topics).\n"
            "        - If available, include specific keywords or entities in each query to improve retrieval precision.\n"
            "        - `tables` loads tables by name, without searching; `*` matches any characters (e.g., `\"sales_*\"` loads every table whose name starts with `sales_`). Use it to load tables whose names you know, such as all tables of a family, or to see again the sample rows of a table you retrieved earlier.\n"
            "        - Retrieved tables accumulate: earlier tables stay available, listed on one line unless they are in the latest retrieval or used by T's column sources.\n"
            f"{multi_notes}"
            "        - Potential join paths between retrieved tables will be provided for reference."
        )
