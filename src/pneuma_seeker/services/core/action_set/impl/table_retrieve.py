from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.interfaces import Action


class TableRetrieve(Action):
    def get_name(self) -> str:
        return ActionNames.TABLE_RETRIEVE.value

    def get_description(self) -> str:
        return f"""**{ActionNames.TABLE_RETRIEVE.value}**:
    Retrieve internal tables.
    - **Args**: {{"prompts": ["<retrieval query 1>", "<retrieval query 2>", ...]}}
    - **Notes**:
        - You may provide multiple retrieval queries in a single call to retrieve tables on different topics (at most {self.config.TABLE_RETRIEVE_MAX_TOPICS} topics).
        - If available, include specific keywords or entities in each query to improve retrieval precision.
        - Previously retrieved tables will be replaced with new retrievals.
        - Potential join paths between retrieved tables will be provided for reference."""

    def get_input_schema(self) -> dict[str, str]:
        return {
            "prompt": "The input query to retrieve relevant tables.",
            "k": "The number of tables to retrieve.",
            "sample_only": "Whether to sample the tables only.",
            "sample_size": "The size of the sample if sample_only is True.",
        }

    def get_notes(self) -> str:
        return "This action uses the IR system to retrieve tables."
