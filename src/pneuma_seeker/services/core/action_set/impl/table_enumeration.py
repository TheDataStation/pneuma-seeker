from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.interfaces import Action


class TableEnumeration(Action):
    def get_name(self) -> str:
        return ActionNames.TABLE_ENUMERATION.value

    def get_description(self) -> str:
        return "Enumerates tables based on the given regex pattern."

    def get_input_schema(self) -> dict[str, str]:
        return {
            "regex_pattern": "The regex pattern to enumerate tables.",
        }

    def get_notes(self) -> str:
        return "This action uses the IR system to enumerate tables."
