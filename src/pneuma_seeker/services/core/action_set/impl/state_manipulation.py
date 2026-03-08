from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.interfaces import Action


class StateManipulation(Action):
    def get_name(self) -> str:
        return ActionNames.STATE_MANIPULATION.value

    def get_description(self) -> str:
        return "Manipulates the state of the system to achieve a desired outcome."

    def get_input_schema(self) -> dict[str, str]:
        return {
            "prompt": "The input query to extract relevant context.",
            "k": "The number of context items to extract.",
            "sample_only": "Whether to sample the context only.",
            "sample_size": "The size of the sample if sample_only is True.",
        }

    def get_notes(self) -> str:
        return "This action uses the IR system to retrieve tables."
