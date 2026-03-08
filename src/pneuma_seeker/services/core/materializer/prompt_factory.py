import json

from pandas import DataFrame

from pneuma_seeker.services.core.action_set.main import ActionSet
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.materializer.action_descriptions import (
    get_materializer_actions,
)
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    convert_retrieval_results_to_str,
)


class MaterializerPromptFactory:
    """Generates prompts for the Materializer LLM agent."""

    def __init__(self, config: Config, action_set: ActionSet) -> None:
        self.config = config
        self.action_set = action_set

    def get_planning_prompt(
        self,
        T: dict[str, DataFrame],
        column_descriptions: dict[str, dict[str, str]],
        S: str,
    ) -> str:
        """Generates the planning prompt for Materializer."""
        return f"""
You are **Materializer**. Your task is to materialize tuples for target tables (T) using allowed actions that operate on retrieved internal tables and user-uploaded external tables (if any).
You operate through iterative steps. In each step, you may select one or more actions based on the current environment (intermediate tables, previous actions, etc.).
The total number of steps must not exceed **{self.config.MAX_MATERIALIZER_STEPS}**.

When forming a sequence of actions for a step, you must follow this **reactive planning structure**:
1. Begin with **{ActionNames.SITUATIONAL_ANALYSIS.value}** to analyze the current environment, evaluate what information is missing, and determine what action(s) are necessary.
2. Perform one or more actions (`{ActionNames.TABLE_PROJECTION.value}`, `{ActionNames.EQUALITY_JOIN.value}`, etc.) to progress toward materializing tuples for T.
3. An action may modify the environment, so actions that depend on previous action outputs must be in separate steps. For example, **{ActionNames.PYTHON_EXECUTOR.value}** that depends on results from `{ActionNames.TABLE_PROJECTION.value}` must occur in a subsequent step after those actions have executed and their outputs are reflected in the environment.

# Target Tables (T)
{json.dumps({k: list(df.columns) for k, df in T.items()}, indent=2)}

## Column Descriptions of T
{column_descriptions}

# Script (S) to be run on (materialized) T later (just for reference, not for you to execute)
{S}

# Available Actions
{get_materializer_actions(self.config, self.action_set)}

# Notes

## Table Handling Guidelines
- Treat external tables just like internal tables, except they are fixed and will never be replaced by calling {ActionNames.TABLE_RETRIEVE.value}.
- Use external tables (if available) and internal tables; call {ActionNames.TABLE_RETRIEVE.value} to retrieve or re-retrieve internal tables (if necessary).
- Internal tables are reset each time {ActionNames.TABLE_RETRIEVE.value} is used; external tables persist.
- You may already see some internal tables provided at the start (pre-fetched by the caller). Treat it the same as if you had retrieved it yourself — use them if useful, or call {ActionNames.TABLE_RETRIEVE.value} again if needed. These pre-fetched tables are not guaranteed to be complete or sufficient.
{f"- You may already see a web search result provided at the start (pre-fetched by the caller). Treat it the same as if you had performed the web search yourself — use it if useful. This pre-fetched web search result is not guaranteed to be complete or sufficient.\n" if self.config.ENABLE_WEB_SEARCH else ""}
{f"- You may already see a web crawl result provided at the start (pre-fetched by the caller). Treat it the same as if you had performed the web crawl yourself — use it if useful. This pre-fetched web crawl result is not guaranteed to be complete or sufficient.\n" if self.config.ENABLE_WEB_CRAWL else ""}

# Action-Related Guidelines
- Always include an "args" object (use {{}} if the action has no args).
- If an action has an `assign_to` argument, set the argument to the correct target table IDs or intermediate table IDs exactly (case-sensitive).
{f"- Use {ActionNames.CONTEXT_EXTRACTION.value} to validate assumptions (e.g., about the existence of values) in the tables prior to determining how best to integrate them.\n" if self.config.ENABLE_CONTEXT_EXTRACTION else ""}
- If you need to integrate (e.g., union) tables of certain names or patterns, the pre-provided tables may not be comprehensive. Call {ActionNames.TABLE_ENUMERATION.value} to discover all matching tables.
- Operator outputs (e.g., from {ActionNames.TABLE_PROJECTION.value} and {ActionNames.EQUALITY_JOIN.value}) are persisted immediately into the workspace database. You can use those output tables in subsequent operator calls.
- You may compose operators arbitrarily (e.g., projection -> join -> projection).
    - Prefer using operators first (e.g., {ActionNames.TABLE_PROJECTION.value}, {ActionNames.EQUALITY_JOIN.value}, {ActionNames.TABLE_UNION.value}) whenever they can express the transformation.
    - If operators are insufficient but the transformation is clean in standard SQL, use {ActionNames.QUERY_EXECUTOR.value} to execute SQL and persist the result into an intermediate table.
    - Use {ActionNames.PYTHON_EXECUTOR.value} only when you need more degrees of freedom than operators/SQL can provide.

# Output
Return **one JSON object** describing your planned actions for this step, e.g.,
{{
  "plan": [
    {{"action": "{ActionNames.SITUATIONAL_ANALYSIS.value}", "args": {{"message": "..."}}}},
    {{"action": "<one of the available actions>", "args": {{...}}}},
    ...
  ]
}}
""".strip()

    def get_context_prompt(
        self,
        retrieved_tables: list[AbstractDocument],
        intermediate_tables: list[AbstractDocument],
        recent_actions: list[str],
        step_count: int,
        user_side_note: str,
        external_tables: list[AbstractDocument],
        web_search_result: AbstractDocument | None,
        web_crawl_result: AbstractDocument | None,
        join_paths: str | None,
    ) -> str:
        """Generates the context prompt for each iteration of the Materializer."""
        return f"""
Step {step_count} (out of maximum {self.config.MAX_MATERIALIZER_STEPS} steps)
{f"\nUser note: {user_side_note}\n" if len(user_side_note) > 0 else ""}
Recent actions:
{recent_actions}

Intermediate tables created so far:
{convert_retrieval_results_to_str(intermediate_tables)}

Retrieved tables:
{convert_retrieval_results_to_str(retrieved_tables)}
{"- Potential join paths between retrieved tables:\n" + join_paths if join_paths else ""}
{f"\nExternal tables:\n{convert_retrieval_results_to_str(external_tables)}\n" if len(external_tables) > 0 else ""}
{f"\nWeb search result:\n{web_search_result}\n" if self.config.ENABLE_WEB_SEARCH and web_search_result else ""}
{f"\nWeb crawl result:\n{web_crawl_result}\n" if self.config.ENABLE_WEB_CRAWL and web_crawl_result else ""}
Decide your next plan and output a JSON object of one or more actions.""".strip()

    def get_skeleton_context_prompt(
        self,
        step_count: int,
    ) -> str:
        """Generates the context prompt for each iteration of the Materializer."""
        return f"""
Step {step_count} (out of maximum {self.config.MAX_MATERIALIZER_STEPS} steps)
... (truncated for brevity)
Decide your next plan and output a JSON object of one or more actions.""".strip()
