import re
from typing import Any

import pandas as pd

from pneuma_seeker.services.core.action_set.interfaces import Action, Executable
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType

# Matches a leading "CREATE [OR REPLACE] TABLE <ident> AS" that the LLM may
# mistakenly include even though execute() already adds that wrapper itself.
_CREATE_TABLE_AS_RE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+\S+\s+AS\s+",
    re.IGNORECASE | re.DOTALL,
)


def _quote_ident(identifier: str) -> str:
    """Safely quotes a DuckDB identifier using double-quotes."""
    return '"' + identifier.replace('"', '""') + '"'


def _strip_create_wrapper(sql: str) -> str:
    """Remove an accidental CREATE TABLE … AS prefix, returning just the query body."""
    return _CREATE_TABLE_AS_RE.sub("", sql).strip().rstrip(";").strip()


def _assert_single_statement(sql: str) -> None:
    """Reject obvious multi-statement SQL to reduce foot-guns."""
    stripped = sql.strip()
    if stripped == "":
        raise ValueError("Input 'query' must be a non-empty string.")

    # Allow at most one trailing semicolon.
    if ";" in stripped[:-1]:
        raise ValueError(
            "Input 'query' must be a single SQL statement (no embedded ';')."
        )


class QueryExecutor(Action, Executable):
    """Executes a SQL query and materializes the result into a workspace table."""

    action_name = ActionNames.QUERY_EXECUTOR
    agents = frozenset({AgentType.MATERIALIZER})
    flag = None
    order = 5
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        return f"""**{ActionNames.QUERY_EXECUTOR.value}**
    - Executes a single SQL query and materializes the result into a workspace table.
    - The system will run: `CREATE OR REPLACE TABLE "<assign_to>" AS <query>`, then returns a preview with `SELECT * FROM "<assign_to>" LIMIT {self.config.MAX_RESULT_PREVIEW_ROWS}`.
    - Input guidelines:
        - `query` should generally start with `SELECT ...` or `WITH ... SELECT ...`.
        - Provide only one SQL statement (no embedded `;`).
        - When referencing retrieved/enumerated tables, use dataset-qualified names like `y."x"`.
        - When referencing intermediate or external tables created in the workspace, use the table name directly.
    - Args: {{"query": "<SQL query>", "assign_to": "<ID of the resulting intermediate table>"}}\n"""

    def get_input_schema(self) -> dict[str, str]:
        return {
            "query": "SQL query string (typically a SELECT/CTE) to materialize.",
            "result_table_id": "Workspace table name to CREATE OR REPLACE.",
        }

    def get_notes(self) -> str:
        return (
            "This action runs: CREATE OR REPLACE TABLE <result_table_id> AS <query>. "
            f"Then it returns SELECT * FROM <result_table_id> LIMIT {self.config.MAX_RESULT_PREVIEW_ROWS}."
        )

    def execute(self, input: dict[str, Any]) -> pd.DataFrame:
        query = input.get("query")
        result_table_id = input.get("result_table_id")

        if not isinstance(query, str):
            raise ValueError("Input 'query' must be a string.")
        if not isinstance(result_table_id, str):
            raise ValueError("Input 'result_table_id' must be a string.")

        # Strip any accidental CREATE TABLE … AS wrapper before we add our own.
        query = _strip_create_wrapper(query)
        _assert_single_statement(query)
        if result_table_id.strip() == "":
            raise ValueError("Input 'result_table_id' must be a non-empty string.")

        quoted_table = _quote_ident(result_table_id)
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f"CREATE OR REPLACE TABLE {quoted_table} AS {query}",
        )
        return self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f"SELECT * FROM {quoted_table} LIMIT {self.config.MAX_RESULT_PREVIEW_ROWS};",
        )
