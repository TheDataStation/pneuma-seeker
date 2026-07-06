from typing import Any

from pneuma_seeker.services.core.action_set.interfaces import Action, Executable
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import LLMOption
from pneuma_seeker.shared.schemas.language_model.role import Role


class UserFacingCommunication(Action, Executable):
    action_name = ActionNames.USER_FACING_COMMUNICATION
    agents = frozenset({AgentType.CONDUCTOR})
    flag = None
    order = 99
    show_in_prompt = False

    def get_description(self, agent: AgentType | None = None) -> str:
        return (
            f"**{ActionNames.USER_FACING_COMMUNICATION.value}**: Signals that you are "
            "ready to respond to the user now. The system generates the actual response "
            "text from the current context — this action takes no arguments."
            "\n  - **Args**: {}"
        )

    def execute(self, input: dict[str, Any]) -> str:
        planning_messages: list[LLMMessage] = input.get("planning_messages") or []
        user_message: str = input.get("user_message", "")
        interaction_history: list[LLMMessage] = input.get("interaction_history") or []
        forced: bool = bool(input.get("forced", False))

        is_follow_up = len(interaction_history) > 0
        messages = list(planning_messages) + [
            LLMMessage(
                role=Role.SYSTEM.value,
                content=self._get_response_system_prompt(
                    user_message, interaction_history, is_follow_up, forced
                ),
            )
        ]
        response = "".join(
            self.language_model_api.chat(messages, LLMOption(stream=False))
        )
        return response.strip()

    def _get_response_system_prompt(
        self,
        user_message: str,
        interaction_history: list[LLMMessage],
        is_follow_up: bool,
        forced: bool,
    ) -> str:
        follow_up_clause = (
            f"This is a follow-up in an ongoing conversation with the user "
            f"(prior turns below). Frame your response as a delta/update relative to "
            f"what you already told them — highlight only what's new or changed, don't "
            f"restate the full analysis from scratch.\n\nPrior turns:\n"
            f"{self._format_interaction_history(interaction_history)}"
            if is_follow_up
            else "This is the first response you are giving in this conversation."
        )
        forced_clause = (
            "\nThe step budget was exhausted before the task could be fully resolved. "
            "Clearly state what you were able to accomplish and what remains uncertain "
            "or incomplete — do not present a partial result as if it were final and complete."
            if forced
            else ""
        )
        return f"""You have just finished working through the plan above (retrieving tables, materializing data, running analysis, etc.) to address the user's request below. Respond to the user directly now.

User's request: {user_message}

Guidelines:
- Briefly state what was done (data used, transformation/analysis performed) and give the concrete answer.
- Directly answer the user's actual question — don't just narrate your process or restate the plan.
- If you relied on a proxy metric, made assumptions, or hit data limitations, disclose them concisely.
- Be concise: not too long (avoid dumping full tables or verbose step-by-step narration), not too brief (the user should understand both the result and enough to trust it).
- {follow_up_clause}{forced_clause}
- Do not mention internal action/tool names, JSON, or system mechanics — write as a natural, direct response."""

    def _format_interaction_history(self, messages: list[LLMMessage]) -> str:
        lines = []
        for i in range(0, len(messages) - 1, 2):
            if (
                messages[i]["role"] == Role.USER.value
                and messages[i + 1]["role"] == Role.ASSISTANT.value
            ):
                lines.append(
                    f'- {{"user input": {messages[i]["content"]}, "your response": {messages[i + 1]["content"]}}}'
                )
        return "\n".join(lines)
