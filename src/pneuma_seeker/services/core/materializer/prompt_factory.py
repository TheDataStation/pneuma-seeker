import json

from pandas import DataFrame

from pneuma_seeker.services.core.action_set.main import ActionSet
from pneuma_seeker.services.core.retrieved_tables import render_retrieved_tables
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType
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
        is_update_mode: bool = False,
        prior_intermediate_table_ids: list[str] | None = None,
        column_sources: dict[str, dict[str, list[str]]] | None = None,
    ) -> str:
        """Generates the planning prompt for Materializer."""
        return f"""
You are **Materializer**. Your task is to materialize tuples for target tables (T) by performing all necessary data integration — joins, unions, source-level filtering, and any other transformation needed to populate T from the available source tables. You are the integration layer: Conductor defines what the output schema should look like; you figure out how to produce it.
You operate through iterative steps. In each step, you may select one or more actions based on the current environment (intermediate tables, previous actions, etc.).
The total number of steps must not exceed **{self.config.MAX_MATERIALIZER_STEPS}**.

**Key constraint**: Each table you produce must have exactly the columns specified in T — no more, no fewer. Do not add extra columns not listed in T.

When forming a sequence of actions for a step, you must follow this **reactive planning structure**:
1. Begin with **{ActionNames.SITUATIONAL_ANALYSIS.value}** to analyze the current environment, evaluate what information is missing, and determine what action(s) are necessary.
2. Perform one or more actions (`{ActionNames.TABLE_PROJECTION.value}`, `{ActionNames.EQUALITY_JOIN.value}`, etc.) to progress toward materializing tuples for T.
3. An action may modify the environment, so actions that depend on previous action outputs must be in separate steps. For example, **{ActionNames.PYTHON_EXECUTOR.value}** that depends on results from `{ActionNames.TABLE_PROJECTION.value}` must occur in a subsequent step after those actions have executed and their outputs are reflected in the environment.

# Target Tables (T)
{json.dumps({k: list(df.columns) for k, df in T.items()}, indent=2)}

## Column Descriptions of T
{column_descriptions}

## Column Sources of T
Where Conductor expects each column to come from (`<dataset>.<table>.<column>`; `*` matches a family of tables). They tell you which source tables to use, not how to combine them.
{json.dumps(column_sources or {}, indent=2)}

# Script (S) to be run on (materialized) T later (just for reference, not for you to execute)
{S}

# Available Actions
{self.__get_actions_section()}

# Notes

## Table Handling Guidelines
- You are given the tables named by T's column sources (internal tables) and the user's external tables. Build T from these only.
- If these tables cannot produce T — a source table is missing, or the provided tables lack the values a column needs — do not work around it: call {ActionNames.REPORT_BLOCKER.value} with what is missing, and Conductor will revise T and its column sources.
{f"- You may already see a web search result provided at the start (pre-fetched by the caller). Treat it the same as if you had performed the web search yourself — use it if useful. This pre-fetched web search result is not guaranteed to be complete or sufficient.\n" if self.config.ENABLE_WEB_SEARCH else ""}
{f"- You may already see a web crawl result provided at the start (pre-fetched by the caller). Treat it the same as if you had performed the web crawl yourself — use it if useful. This pre-fetched web crawl result is not guaranteed to be complete or sufficient.\n" if self.config.ENABLE_WEB_CRAWL else ""}

# Action-Related Guidelines
- Always include an "args" object (use {{}} if the action has no args).
- If an action has an `assign_to` argument, set the argument to the correct target table IDs or intermediate table IDs exactly (case-sensitive).
{f"- Use {ActionNames.CONTEXT_EXTRACTION.value} to validate assumptions (e.g., about the existence of values) in the tables prior to determining how best to integrate them.\n" if self.config.ENABLE_CONTEXT_EXTRACTION else ""}
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
{self.__get_update_mode_section(is_update_mode, prior_intermediate_table_ids or [])}""".strip()

    def get_integration_plan_sys_prompt(self) -> str:
        """Gets the system prompt for the Materializer's up-front integration plan."""
        return f"""
# Role
You are the planning stage of **Materializer** in **Pneuma-Seeker**. Before any step runs, outline how to build the target tables (T) from the provided source tables.

# What to produce
For each table in T, a short numbered list of integration steps: which source tables to read (as T's column sources name them), how they combine (joins on which keys, unions of which tables), the filters the column descriptions imply, and how each column is derived. Note anything you need to check in the data first, and any column the provided tables cannot produce (that would call for {ActionNames.REPORT_BLOCKER.value}).

# Rules
- Use only the provided tables. Do not write code; describe the steps.
- Keep it concise. The plan is provisional and will be revised when the data shows something different.

# Output
Plain text: the plan only.
""".strip()

    def get_integration_plan_user_prompt(
        self,
        T: dict[str, DataFrame],
        column_descriptions: dict[str, dict[str, str]],
        column_sources: dict[str, dict[str, list[str]]],
        S: str,
        note: str,
        tables: list[AbstractDocument],
    ) -> str:
        """Gets the user message for the Materializer's up-front integration plan."""
        return f"""
Target tables (T):
{json.dumps({k: list(df.columns) for k, df in T.items()}, indent=2)}

Column descriptions of T:
{column_descriptions}

Column sources of T:
{json.dumps(column_sources, indent=2)}

Script (S) to be run on T later (for reference):
{S}
{f"\nNote from Conductor: {note}\n" if note else ""}
Provided tables:
{render_retrieved_tables(tables)}
""".strip()

    def __get_update_mode_section(
        self, is_update_mode: bool, prior_intermediate_table_ids: list[str]
    ) -> str:
        if not is_update_mode:
            return ""
        ids_str = (
            ", ".join(f'"{tid}"' for tid in prior_intermediate_table_ids)
            if prior_intermediate_table_ids
            else "(none)"
        )
        return f"""

# Update Mode
T was previously materialized. The following intermediate tables from the prior run are
already present in the database and can be reused directly: {ids_str}

Your job is to apply the **delta only** — update or extend existing tables to satisfy
the new T schema rather than re-deriving everything from scratch.
- Use {ActionNames.CONTEXT_EXTRACTION.value} to inspect the pre-loaded intermediates and
  confirm what columns they already have before deciding what work remains.
- If a pre-loaded table already satisfies the new T schema for that table ID, no further
  action is needed for it.
- Only redo work that is genuinely no longer valid given the updated T."""

    def __get_actions_section(self) -> str:
        actions = self.action_set.registry.get_for_agent(AgentType.MATERIALIZER)
        return "\n\n".join(
            f"- {a.get_description(AgentType.MATERIALIZER)}" for a in actions
        )

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

Provided tables (named by T's column sources):
{render_retrieved_tables(retrieved_tables)}
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
