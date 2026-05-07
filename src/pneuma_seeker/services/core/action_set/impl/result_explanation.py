from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.interfaces import Action


class ResultExplanation(Action):
    def get_name(self) -> str:
        return ActionNames.RESULT_EXPLANATION.value

    def get_description(self) -> str:
        return (
            "Reads the full step-by-step derivation of the current result: which source tables "
            "were used, how they were integrated (materialization steps with code), the processing "
            "script, and the result schema with per-column annotations. Use this when the user asks "
            "how the result was produced, wants an explanation of the pipeline, or asks follow-up "
            "questions about methodology."
        )

    def get_input_schema(self) -> dict[str, str]:
        return {}

    def get_notes(self) -> str:
        return "Returns pipeline steps and code without raw data rows to keep token usage low."
