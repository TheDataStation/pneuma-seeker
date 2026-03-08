from typing import Any

import pandas as pd

from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.services.core.action_set.interfaces import Executable
from pneuma_seeker.shared.schemas.core.action import ActionNames


def _quote_ident(identifier: str) -> str:
	"""Safely quotes a DuckDB identifier using double-quotes."""
	return '"' + identifier.replace('"', '""') + '"'


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

	def get_name(self) -> str:
		return ActionNames.QUERY_EXECUTOR.value

	def get_description(self) -> str:
		return f"""**{ActionNames.QUERY_EXECUTOR.value}**
    - Executes a single SQL query and materializes the result into a workspace table.
    - The system will run: `CREATE OR REPLACE TABLE "<assign_to>" AS <query>`, then returns a preview with `SELECT * FROM "<assign_to>" LIMIT 10`.
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
			"Then it returns SELECT * FROM <result_table_id> LIMIT 10."
		)

	def execute(self, input: dict[str, Any]) -> pd.DataFrame:
		query = input.get("query")
		result_table_id = input.get("result_table_id")

		if not isinstance(query, str):
			raise ValueError("Input 'query' must be a string.")
		if not isinstance(result_table_id, str):
			raise ValueError("Input 'result_table_id' must be a string.")

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
			f"SELECT * FROM {quoted_table} LIMIT 10;",
		)
