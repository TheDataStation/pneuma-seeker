from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.interfaces.abstract_action import Action


class TableRetrieve(Action):
    def get_name(self) -> str:
        return ActionNames.TABLE_RETRIEVE.value

    def get_description(self) -> str:
        return "Retrieves relevant tables based on the input query."

    def get_input_schema(self) -> dict[str, str]:
        return {
            "prompt": "The input query to retrieve relevant tables.",
            "k": "The number of tables to retrieve.",
            "sample_only": "Whether to sample the tables only.",
            "sample_size": "The size of the sample if sample_only is True.",
        }

    def get_notes(self) -> str:
        return "This action uses the IR system to retrieve tables."
