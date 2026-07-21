from typing import Any

from pneuma_seeker.services.core.action_set.interfaces import Action, Executable
from pneuma_seeker.shared.parser import parse_json
from pneuma_seeker.shared.schemas.core.action import PLAN_INSTRUCTION, ActionNames
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
        plan_mode: bool = bool(input.get("plan_mode", False))

        is_follow_up = len(interaction_history) > 0
        planning_transcript = self._format_planning_transcript(planning_messages)
        system_prompt = (
            self._get_plan_mode_response_system_prompt(
                user_message,
                interaction_history,
                is_follow_up,
                forced,
                planning_transcript,
            )
            if plan_mode
            else self._get_response_system_prompt(
                user_message,
                interaction_history,
                is_follow_up,
                forced,
                planning_transcript,
            )
        )
        # Single system message, not planning_messages' raw turns — replaying those
        # verbatim conditions the model to keep answering in the same JSON plan
        # format instead of switching to prose.
        messages = [LLMMessage(role=Role.SYSTEM.value, content=system_prompt)]
        response = self._generate(messages)

        if self._looks_like_plan_json(response):
            correction_messages = messages + [
                LLMMessage(role=Role.ASSISTANT.value, content=response),
                LLMMessage(
                    role=Role.SYSTEM.value,
                    content=(
                        "That reply was formatted as an internal action-plan (JSON "
                        "with 'action'/'plan' keys) instead of a direct message to "
                        "the user. Rewrite it as plain natural-language prose only "
                        "— no JSON, no action names, no braces/brackets."
                    ),
                ),
            ]
            response = self._generate(correction_messages)

        return response

    def _generate(self, messages: list[LLMMessage]) -> str:
        response = "".join(
            self.language_model_api.chat(messages, LLMOption(stream=False))
        )
        return response.strip()

    def _looks_like_plan_json(self, text: str) -> bool:
        """Lightweight guard against the model responding with an internal
        action-plan JSON blob instead of a natural-language reply."""
        if '"action":' not in text and '"plan":' not in text:
            return False
        stripped = text.strip()
        if not stripped or stripped[0] not in "{[":
            # Plan-shaped key(s) present but the text isn't even well-formed
            # JSON to begin with (e.g. a leak truncated at/missing the opening
            # brace) — still a leak.
            return True
        try:
            parsed = parse_json(stripped)
        except ValueError:
            return True  # garbled near-JSON dump, e.g. mismatched brackets
        return isinstance(parsed, dict) and ("plan" in parsed or "action" in parsed)

    def _get_response_system_prompt(
        self,
        user_message: str,
        interaction_history: list[LLMMessage],
        is_follow_up: bool,
        forced: bool,
        planning_transcript: str,
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
        return f"""You have just finished working through the following plan (retrieving tables, materializing data, running analysis, etc.) to address the user's request below. Respond to the user directly now.

What you did, step by step:
{planning_transcript or "(no planning steps were recorded)"}

User's request: {user_message}

Guidelines:
- Briefly state what was done (data used, transformation/analysis performed) and give the concrete answer.
- Directly answer the user's actual question — don't just narrate your process or restate the plan.
- If you relied on a proxy metric, made assumptions, or hit data limitations, disclose them concisely.
- Be concise: not too long (avoid dumping full tables or verbose step-by-step narration), not too brief (the user should understand both the result and enough to trust it).
- {follow_up_clause}{forced_clause}
- Do not mention internal action/tool names, JSON, or system mechanics — write as a natural, direct response."""

    def _get_plan_mode_response_system_prompt(
        self,
        user_message: str,
        interaction_history: list[LLMMessage],
        is_follow_up: bool,
        forced: bool,
        planning_transcript: str,
    ) -> str:
        follow_up_clause = (
            f"This is a follow-up in an ongoing conversation with the user "
            f"(prior turns below). Frame your response as a delta/update relative to "
            f"what you already told them — highlight only what's new or changed, don't "
            f"restate the full proposal from scratch.\n\nPrior turns:\n"
            f"{self._format_interaction_history(interaction_history)}"
            if is_follow_up
            else "This is the first response you are giving in this conversation."
        )
        forced_clause = (
            "\nThe step budget was exhausted before you settled on a confident "
            "proposal. Present your best-effort proposal so far and be explicit "
            "about what's still unresolved — do not present it as a finished plan."
            if forced
            else ""
        )
        return f"""You have explored the available data (retrieving tables, checking values, etc.) to address the user's request below. You have NOT materialized any tables or executed any computation — do not state or imply that any result has been computed, run, or found.

What you did while exploring, step by step:
{planning_transcript or "(no planning steps were recorded)"}

User's request: {user_message}

First, decide which of these applies, based on what you actually found — not on what would be convenient to assume:

**A. You have a grounded proposal.** You found data that plausibly relates to the request and settled on a (T,S) you're reasonably confident in. Present it now, using this structure:
1. **Proposed data**: What table(s)/columns you intend to use and why they're relevant — described in plain language, not as a raw schema dump.
2. **Proposed analysis**: What the script will compute, filter, or aggregate — described narratively (what it does and why), not as code.
3. **Assumptions & interpretations**: Any proxy metrics, ambiguous terms, or scope decisions you resolved on your own, so the user can correct them if wrong.
4. **Open questions** (if any): Clarifying questions that would materially change the analysis if answered differently.
Close by explicitly inviting the user to confirm or tell you what to change so you can proceed.

**B. The available data does not support this request.** After exploring, nothing you found plausibly relates to the question — a genuine subject-matter mismatch, not just an imperfect proxy (e.g., the dataset covers a different domain than what's being asked). In this case, do **not** force a proposal into the structure above. Say directly and honestly that this doesn't look answerable with the available data, briefly note what you checked and why it didn't fit, and ask whether the user has different data in mind or wants to rephrase.

Guidelines:
- Be concise: a busy reader should be able to skim and respond "looks good" or correct one thing.
- {follow_up_clause}{forced_clause}
- Do not mention internal action/tool names, JSON, or system mechanics — write as a natural, direct response."""

    def _format_planning_transcript(self, planning_messages: list[LLMMessage]) -> str:
        # Narrative summary ("Step N plan/output"), not literal turns — see execute().
        steps: list[dict[str, Any]] = []
        for msg in planning_messages:
            role = msg["role"]
            content = msg["content"]
            if role == Role.SYSTEM.value or PLAN_INSTRUCTION in content:
                continue
            if role == Role.ASSISTANT.value:
                steps.append({"plan": content, "results": []})
            elif role == Role.USER.value and steps:
                steps[-1]["results"].append(content)

        lines = []
        for i, step in enumerate(steps, start=1):
            lines.append(f"Step {i} plan: {step['plan']}")
            for result in step["results"]:
                lines.append(f"Step {i} output: {result}")
        return "\n".join(lines)

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
