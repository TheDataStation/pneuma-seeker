"""Prompt factory baseline: retrieve → compute → answer.

This prompt factory instructs the model to:
1) retrieve relevant internal tables (optionally enumerate related tables),
2) run a single compute step to create a small preview result table, and
3) answer the user in a subsequent step.

The compute step uses the `assumption_check` action, which executes inline code
and must create or replace a table named "conductor_assumption_check" in the
workspace database.
"""

from __future__ import annotations

from pneuma_seeker.services.core.conductor.state import ConductorState
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.conductor import UserConductorInteraction
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    convert_retrieval_results_to_str,
)


class SimplePromptFactory:
    """Factory for prompts used by Conductor (direct retrieve + compute baseline)."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def get_sys_prompt(self) -> str:
        """Gets the system prompt for the baseline."""
        return f"""
# Role
You are **Planner**, an iterative agent in **Pneuma-Seeker**.

# Goal
Answer the user's question by retrieving relevant tables and producing a **small, preview-sized** result table via one compute step.

You operate iteratively, and the total number of steps must not exceed **{self.config.MAX_CONDUCTOR_STEPS}**.

# Planning Structure (important)
In each step, follow this structure:
1. Start with **{ActionNames.SITUATIONAL_ANALYSIS.value}**.
2. Use **{ActionNames.TABLE_RETRIEVE.value}** (and optionally **{ActionNames.TABLE_ENUMERATION.value}**) to discover tables.
3. Use **{ActionNames.CONTEXT_EXTRACTION.value}** as the **compute** tool to query/transform tables and materialize a preview result table.
4. Communicate the final answer in a **subsequent step** via **{ActionNames.USER_FACING_COMMUNICATION.value}**.

Important runtime constraint:
- If your plan includes `{ActionNames.TABLE_RETRIEVE.value}` or `{ActionNames.CONTEXT_EXTRACTION.value}`, do **not** include `{ActionNames.USER_FACING_COMMUNICATION.value}` in the same plan.
  Do retrieval/compute first, then answer on the next step.

# Actions
{self.__get_table_retrieve_description()}
{self.__get_table_enumeration_description()}
{self.__get_assumption_check_as_compute_description()}
{self.__get_web_search_description() if self.config.ENABLE_WEB_SEARCH else ""}
{self.__get_web_crawl_description() if self.config.ENABLE_WEB_CRAWL else ""}

# Available Data
- **Internal Tables**: retrievable via `{ActionNames.TABLE_RETRIEVE.value}`.
- **External Tables**: user-uploaded tables if any; already visible in the environment.
{("- **Web Search Results**: Relevant information from the web.\n" if self.config.ENABLE_WEB_SEARCH else "")}

# Guidelines on Table Retrieval
- Each `{ActionNames.TABLE_RETRIEVE.value}` prompt should preserve semantic constraints in the user input, not just entity keywords.
  Include: entities + at least one qualifier/constraint (time, geography, policy, etc.).
- `{ActionNames.TABLE_RETRIEVE.value}` can be noisy; use a quick compute step to sample/check columns before committing to heavy logic.

# Output
Return **one JSON object** describing your planned actions for this step:
{{
  "plan": [
    {{"action": "{ActionNames.SITUATIONAL_ANALYSIS.value}", "args": {{"message": "..."}}}},
    {{"action": "{ActionNames.TABLE_RETRIEVE.value}", "args": {{"prompts": ["..."]}}}},
    {{"action": "{ActionNames.CONTEXT_EXTRACTION.value}", "args": {{"code": "..."}}}}
  ]
}}
""".strip()

    def __get_table_retrieve_description(self) -> str:
        return f"""- **{ActionNames.TABLE_RETRIEVE.value}**:
  Retrieve internal tables.
  - **Args**: {{"prompts": ["<retrieval query 1>", "<retrieval query 2>", ...]}}
  - **Notes**:
    - You may provide multiple retrieval queries in a single call (at most {self.config.TABLE_RETRIEVE_MAX_TOPICS} topics).
    - Previously retrieved tables will be replaced with new retrievals.
    - Potential join paths between retrieved tables will be provided for reference.\n"""

    def __get_table_enumeration_description(self) -> str:
        return f"""- **{ActionNames.TABLE_ENUMERATION.value}**:
  List other available internal tables in the database whose names match given regex patterns.
  - **Args**: {{"patterns": ["<regex pattern 1>", "<regex pattern 2>", ...]}}
  - **Notes**:
    - **Precondition — MUST NOT be called unless there is at least one internal table already retrieved.**
    - Each pattern must be derived from names/tokens of existing retrieved tables.
    - You may provide multiple patterns in a single call (at most {self.config.TABLE_RETRIEVE_MAX_TOPICS} patterns).\n"""

    def __get_assumption_check_as_compute_description(self) -> str:
        return f"""- **{ActionNames.CONTEXT_EXTRACTION.value}** (used as compute):
  Execute inline code to query/transform tables and produce a **preview-sized** result.
  - **Args**: {{"code": "<code string>"}}
  - **Execution context**:
    - Tables live in a DuckDB-backed workspace database.
    - Access via: `db_api.execute_query(user_id, chat_id, "<SQL>")`.
    - `db_api`, `user_id`, and `chat_id` are available variables. Do NOT import `db_api`.
    - When referencing retrieved/enumerated internal tables, use dataset-qualified names:
      - Table x (dataset: y) → `y."x"`
    - External/workspace tables can be referenced directly by their table name.
  - **Output contract (required)**:
    - Your code MUST create or replace exactly one table or view named **"conductor_assumption_check"**.
      Example:
      `db_api.execute_query(user_id, chat_id, '''
      CREATE OR REPLACE TABLE conductor_assumption_check AS
      SELECT ...
      ''' )`
    - Keep it small (aggregates/samples/few rows). The system reads at most the first 10 rows.
    - Do NOT create other persistent tables.
  - **Safety/performance**:
    - Prefer SQL over Pandas; do not load full tables into memory.
    - Do NOT create a new DuckDB connection (no `duckdb.connect()`).
    - Use triple-quoted strings for multi-line SQL (real newlines, no escaped `\\n`).\n"""

    def __get_web_search_description(self) -> str:
        return f"""\n- **{ActionNames.WEB_SEARCH.value}**:
  Find a piece of information from the web.
  - **Args**: {{"prompt": "<retrieval query>"}}
  - **Returns**: A summarized textual snippet from relevant web sources.
  - **Notes**:
    - Avoid retrying the same or slightly modified queries repeatedly.\n"""

    def __get_web_crawl_description(self) -> str:
        return f"""\n- **{ActionNames.WEB_CRAWL.value}**:
  Fetch/raw-crawl a specific web page (URL) and return extracted text content.
  - **Args**: {{"url": "<page_url>"}}
  - **Returns**: The textual content (possibly truncated) of the requested page.
  - **Notes**:
    - Returned content is raw extracted text from the page (no summarization).\n"""

    def get_env_state_prompt(
        self,
        current_step: int,
        info_need_state: ConductorState,
        interaction_history: list[UserConductorInteraction],
        actions_taken: list[str],
        retrieved_tables: list[AbstractDocument],
        user_input: str,
        enumerated_tables: list[AbstractDocument],
        external_tables: list[AbstractDocument],
        web_search_result: AbstractDocument | None = None,
        web_crawl_result: AbstractDocument | None = None,
        join_paths: str | None = None,
    ) -> str:
        """Gets the environment state prompt.

        Note: `info_need_state` is ignored (kept for interface compatibility).
        """
        _ = info_need_state
        return f"""
Step {current_step} (out of maximum {self.config.MAX_CONDUCTOR_STEPS} steps)

Current user input: {user_input}

Recent actions:
{actions_taken}

Recent user interactions:
{self.__convert_interactions_to_str(interaction_history)}

Retrieved Tables:
{convert_retrieval_results_to_str(retrieved_tables)}
{"- Potential join paths between retrieved tables:\n" + join_paths if join_paths else ""}
{f"\nOther table IDs with similar naming patterns (for reference):\n{[i.doc_id for i in enumerated_tables]}\n" if len(enumerated_tables) > 0 else ""}
{f"\nExternal tables:\n{convert_retrieval_results_to_str(external_tables)}\n" if len(external_tables) > 0 else ""}
{f"\nWeb search result:\n{web_search_result}\n" if self.config.ENABLE_WEB_SEARCH and web_search_result else ""}
{f"\nWeb crawl result:\n{web_crawl_result}\n" if self.config.ENABLE_WEB_CRAWL and web_crawl_result else ""}

Decide your next plan and output a JSON object of one or more actions.
""".strip()

    def get_skeleton_env_state_prompt(self, current_step: int) -> str:
        """Gets a truncated environment prompt for logging/debug."""
        return f"""
Step {current_step} (out of maximum {self.config.MAX_CONDUCTOR_STEPS} steps)
... (truncated for brevity)
Decide your next plan and output a JSON object of one or more actions.
""".strip()

    def get_knowledge_extraction_prompt(self, user_input: str) -> str:
        """Gets the knowledge extraction prompt."""
        return f"""You are very talented in inferring knowledge from a text.
You are given a human input to a question-answering system: ```{user_input}```
Please consider whether it consists domain knowledge that will be helpful for other people using the system. Make sure you only extract general knowledge that does not just apply to a specific user. If there is none, then do not force for there to be any.

When you find multiple pieces of related information, combine them into a single comprehensive knowledge statement rather than splitting them into separate points. The goal is to capture the complete context and relationships in one cohesive statement.

Output JSON:
{{
  "knowledge": "<one concise knowledge statement, or empty string>"
}}
""".strip()

    def get_direct_response_anyway_prompt(self) -> str:
        return (
            "You have reached the maximum number of steps. "
            f"Please answer the current user input. You are essentially asked to produce a `{ActionNames.USER_FACING_COMMUNICATION.value}` response but without the JSON format requirements. Simply output the response answering the current user input."
        )

    def __convert_interactions_to_str(
        self, interactions: list[UserConductorInteraction]
    ) -> str:
        interaction_repr = ""
        for interaction in interactions:
            interaction_repr += f"- {interaction}\n"
        return interaction_repr.strip() or "(no prior interactions)"
