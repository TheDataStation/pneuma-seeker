from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.interfaces import Action


class SituationalAnalysis(Action):
    def get_name(self) -> str:
        return ActionNames.SITUATIONAL_ANALYSIS.value

    def get_description(self) -> str:
        return "Analyzes the situational context to provide more informed decision-making capabilities."

    def get_input_schema(self) -> dict[str, str]:
        return {
            "prompt": "The input query to extract relevant context.",
            "k": "The number of context items to extract.",
            "sample_only": "Whether to sample the context only.",
            "sample_size": "The size of the sample if sample_only is True.",
        }

    def get_notes(self) -> str:
        return "This action uses the IR system to retrieve tables."
