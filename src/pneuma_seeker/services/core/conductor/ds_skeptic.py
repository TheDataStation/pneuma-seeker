"""DS-Skeptic: an adversarial reviewer that sanity-checks T+S before materialization."""

import json
from logging import Logger
from typing import Callable

from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import formatted_log
from pneuma_seeker.shared.parser import parse_json
from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.option import LLMOption
from pneuma_seeker.shared.schemas.language_model.role import Role

_SYSTEM_PROMPT = """You are **DS-Skeptic**, an adversarial statistics and data science reviewer. Your job is to actively verify and poke holes in an analysis plan — specifically the target schema (T) and the analysis script (S) — proposed by an AI planning agent, before expensive computation begins. You are a skeptic by default: probe hard, approve only when satisfied.

**Execution context**: T is a set of target tables that will be materialized from retrieved source tables. S is a Python/SQL script executed against those materialized T tables. S references tables by their T key names (e.g. if T defines `capitals`, S uses `capitals` — this is correct by design; do NOT flag it).

You have access to **CONTEXT_EXTRACTION** to run actual SQL/Python queries against the source tables and verify your assumptions before giving a verdict. Use it proactively — do not raise a concern based on guesswork or sample rows alone.

**On each turn you output exactly one of two JSON formats:**

Option A — run a context extraction to gather evidence:
{
  "action": "context_extraction",
  "uncertainties": [
    {"table_ids": ["<source_table_id>", ...], "question": "<concrete question about the data>"}
  ]
}

Option B — give your final verdict:
{
  "verdict": "approve" | "push_back",
  "concerns": ["<specific concern with a concrete recommended fix>", ...] | null
}

**What to actively scrutinize (use CE to verify anything uncertain):**

1. **T schema design**: Is T the right set of tables and columns? Are there columns in T that are unnecessary? Are there columns missing from T that S will need? Would a different source table serve the question better?

2. **Filter correctness**: Does every hardcoded filter value in S match what the data actually contains? E.g. if S filters `capital = 'primary'`, run CE to confirm 'primary' exists and is semantically correct — do NOT guess from sample rows. A sample row showing value X does not prove value Y doesn't also exist.

3. **Column choice justification**: Is the chosen column the right one for what the user asked? If there are multiple plausible columns (e.g. several date columns, several numeric columns), does S use the most appropriate one? Use CE to inspect alternatives if unsure.

4. **Statistical method justification**: Does the chosen computation actually answer the user's question? Challenge the formula: is a mean the right aggregation or should it be median/sum/count? Is this the right statistical test for the data type and distribution? Is there a simpler or more standard approach that is less likely to be wrong?

5. **Denominator / scope logic**: For percentages, rates, or normalized values, does the denominator correctly represent the full in-scope population (including zero-contribution entities)? Are there entities that should be included but are filtered out?

6. **Aggregation correctness**: Is GROUP BY complete? Any double-counting from joins? Off-by-one in window functions?

7. **Binary encoding / sign**: If a column encodes direction, is it used in the correct orientation?

8. **Statistical pitfalls**: Confounding variables, wrong reference group, unit mismatch, missing join keys, multicollinearity if relevant.

9. **Scope alignment**: Does the combined T+S actually answer all parts of the user's question, including edge cases and stated constraints?

**Important constraints:**
- Use CE before flagging filter values or data distributions — verify against real data, not sample rows.
- Do not raise a concern that directly contradicts advice you gave in a previous round (visible in conversation history). If you told the Conductor to change filter X to Y, do not then advise against Y.
- Each concern must state exactly what is wrong and what the fix should be.
- Approve only after you have actively checked the dimensions above."""


class DSSkeptic:
    """
    Adversarial reviewer invoked after Conductor defines T+S.
    Uses a CE-enabled mini-loop to verify concerns before giving a verdict.
    """

    def __init__(
        self,
        language_model_api: LanguageModelAPI,
        config: Config,
        logger: Logger,
    ) -> None:
        self.language_model_api = language_model_api
        self.config = config
        self.logger = logger

    def review(
        self,
        user_input: str,
        T: dict,
        column_descriptions: dict,
        S: str,
        retrieved_tables: list[AbstractDocument],
        run_ce_fn: Callable[[list[dict]], str] | None = None,
    ) -> tuple[bool, str]:
        """
        Reviews the current T+S plan, optionally using CE to verify concerns.

        Args:
            run_ce_fn: callable(uncertainties) -> summary_str, injected by Conductor.

        Returns:
            (should_push_back, feedback_message_for_conductor)
        """
        self.__log("Starting review...")
        messages = [
            LLMMessage(role=Role.SYSTEM.value, content=_SYSTEM_PROMPT),
            LLMMessage(
                role=Role.USER.value,
                content=self._build_user_prompt(
                    user_input, T, column_descriptions, S, retrieved_tables
                ),
            ),
        ]

        ce_calls = 0
        max_ce = self.config.MAX_DS_SKEPTIC_CE_CALLS
        result: dict = {}

        while True:
            try:
                raw_response = "".join(
                    self.language_model_api.chat(
                        messages, LLMOption(json_mode=True, stream=False)
                    )
                )
                result = parse_json(raw_response)
            except Exception as exc:
                self.__log(f"DS-Skeptic call failed: {exc}. Defaulting to approve.")
                return False, ""

            # CE request
            if (
                result.get("action") == "context_extraction"
                and run_ce_fn is not None
                and ce_calls < max_ce
            ):
                uncertainties = result.get("uncertainties", [])
                self.__log(
                    f"Running CE ({ce_calls + 1}/{max_ce}): "
                    + "; ".join(u.get("question", "")[:60] for u in uncertainties)
                )
                try:
                    ce_summary = run_ce_fn(uncertainties)
                except Exception as ce_exc:
                    ce_summary = f"CE failed: {ce_exc}"
                ce_calls += 1
                messages.append(
                    LLMMessage(role=Role.ASSISTANT.value, content=raw_response)
                )
                messages.append(
                    LLMMessage(
                        role=Role.USER.value,
                        content=(
                            f"Context Extraction result:\n{ce_summary}\n\n"
                            f"You have {max_ce - ce_calls} CE call(s) remaining. "
                            "Now give your verdict or run another context extraction."
                        ),
                    )
                )
                continue

            # Verdict (or CE exhausted / CE unavailable)
            break

        verdict = result.get("verdict", "approve")
        concerns: list[str] = result.get("concerns") or []

        if verdict != "push_back" or not concerns:
            self.__log("Verdict: approve.")
            return False, "DS-Skeptic reviewed your analysis plan and approved it."

        self.__log(f"Verdict: push_back. Concerns: {concerns}")
        parts = [
            "**DS-Skeptic raised concerns about your current analysis plan (T and S):**",
            "",
        ]
        parts.extend(f"- {c}" for c in concerns)
        parts.append(
            "\nPlease revise your T and/or S to address these concerns before proceeding."
        )
        return True, "\n".join(parts)

    def _build_user_prompt(
        self,
        user_input: str,
        T: dict,
        column_descriptions: dict,
        S: str,
        retrieved_tables: list[AbstractDocument],
    ) -> str:
        table_context_parts = [str(t) for t in retrieved_tables[:3]]
        table_context = (
            "\n\n".join(table_context_parts) if table_context_parts else "None"
        )

        T_schema = {
            tid: list(doc.content.columns)
            for tid, doc in T.items()
            if hasattr(doc, "content")
        }

        return f"""## User's Question
{user_input}

## Target Schema (T)
{json.dumps(T_schema, indent=2)}

## Column Descriptions
{json.dumps(column_descriptions, indent=2)}

## Analysis Script (S)
{S}

## For Reference: Retrieved Source Tables (sample rows)
T will be materialized from these. You can run CONTEXT_EXTRACTION against them to verify assumptions.
{table_context}

Begin your review. Run context extractions as needed, then give your verdict."""

    def __log(self, text: str) -> None:
        formatted_log(self.logger, "DS_SKEPTIC", text)
