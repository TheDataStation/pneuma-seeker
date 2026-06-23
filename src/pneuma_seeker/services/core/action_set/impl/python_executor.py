from typing import Any

import numpy as np
import pandas as pd
import scipy

from pneuma_seeker.services.core.action_set.interfaces import Action, Executable
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType


class PythonExecutor(Action, Executable):
    action_name = ActionNames.PYTHON_EXECUTOR
    agents = frozenset({AgentType.CONDUCTOR, AgentType.MATERIALIZER})
    flag = None
    order = 6
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        if agent == AgentType.CONDUCTOR:
            return self._conductor_description()
        return self._materializer_description()

    def _conductor_description(self) -> str:
        return (
            f"**{ActionNames.PYTHON_EXECUTOR.value}**:\n"
            f"  Execute `S` on `T` to produce the final information that will be communicated to the user via `{ActionNames.USER_FACING_COMMUNICATION.value}`.\n"
            "  - **Args**: {}"
        )

    def _materializer_description(self) -> str:
        return f"""**{ActionNames.PYTHON_EXECUTOR.value}**
    - Executes Python code to transform and/or combine data.
    - Tables are stored in the DuckDB-based workspace database and are NOT guaranteed to fit in memory.
    - To access tables, use the provided database API:
        - `db_api.execute_query(user_id, chat_id, "<SQL query>")`
        - `db_api.register_temporary_df(user_id, chat_id, df, "<table_name>")` can be used to register a small temporary Pandas DataFrame in the workspace for SQL queries.
        - `db_api`, `chat_id`, and `user_id` are available as variables in the environment. Do NOT import `db_api`; use `db_api.<function_name>`.
        - These APIs return a Pandas DataFrame containing the relational query result.
    - Note:
        - When referencing retrieved or enumerated tables, use the dataset-qualified name as provided by the system. For example, Table x (dataset: y) -> y."x".
            - Internally, `db_api` uses DuckDB ATTACH DATABASE to connect datasets to the workspace database. The attached dataset name (e.g., y) is a *catalog* (database), not a schema. Tables live under y.main.<table>, but DuckDB allows shorthand access as y.<table>.
            Do NOT treat the dataset name as a schema when inspecting metadata.
        - When referencing intermediate or external tables created in the workspace, use the table name directly. For example, Table x -> x.
    - Prefer performing transformations directly in standard SQL whenever possible instead of loading tables using Pandas.
    - If Python processing is necessary, process data in batches and never load full tables into memory. For example:
        - Offset pagination:
            - SELECT * FROM "<table_id>" ORDER BY rowid LIMIT 100 OFFSET 0;
            - SELECT * FROM "<table_id>" ORDER BY rowid LIMIT 100 OFFSET 100;
        - Keyset pagination (preferred for large tables):
            - SELECT * FROM "<table_id>" WHERE rowid > last_seen_rowid ORDER BY rowid LIMIT 100;
    - When writing Python code:
        - Never use escaped newlines (\\n) inside strings.
        - Use triple-quoted strings for multi-line SQL with real newlines.
        - The value of `code` MUST be raw Python source code, not a quoted string.
        - Never wrap Python code in quotes.
        - Do not construct Python code as strings for later execution (no exec-style indirection).
        - Do NOT create a new DuckDB connection (no `duckdb.connect()`). Any tables created on a standalone connection will not be visible to the workspace DB, and the system will fail when it tries to read the expected output table.
    - When using CTEs (WITH ...), attach them directly to the SELECT of a CREATE TABLE AS statement:
        CREATE OR REPLACE TABLE <name> AS
        WITH ...
        SELECT ...
    - Pandas, NumPy, and SciPy are available for small, intermediate, in-memory batches only (remember to add relevant import statements if needed).
    - You can perform operations such as renaming or reordering columns, transforming column values, normalizing formats (e.g., "Month Date, Year" -> "yyyy-mm-dd"), etc.
    - You may create intermediate tables during processing (e.g., to store batches or results of sub-steps; don't forget to clean them up), but the final output must be materialized as a single table in the database using SQL (for example, `CREATE OR REPLACE TABLE "<assign_to>" AS SELECT ...`).
    - Do NOT return large tables as Pandas DataFrames. Any Pandas DataFrame should only be used for small previews or intermediate batch processing.
    - If your code creates the final output table using SQL (e.g., `CREATE OR REPLACE TABLE "<assign_to>" AS ...`), do **NOT** call `db_api.register_temporary_df(..., "<assign_to>")` with a preview DataFrame.
        - DuckDB's registration can shadow the persistent table with the same name, which would silently truncate downstream queries.
        - If you need a preview, query `SELECT * FROM "<assign_to>" LIMIT 10` and (optionally) register it under a different temporary name like `"<assign_to>__preview"`.
    - Args: {{"code": "<Python code string>", "assign_to": "<ID of the resulting intermediate table>"}}"""

    def execute(
        self,
        input: dict[str, Any],
    ) -> pd.DataFrame:
        """Executes the tool with the given input and returns the output."""
        code = input.get("code")
        result_table_id = input.get("result_table_id")
        if not isinstance(code, str):
            raise ValueError("Input 'code' must be a string.")
        if not isinstance(result_table_id, str):
            raise ValueError("Input 'result_table_id' must be a string.")

        # Basic sanitization/fixes for model-produced code to reduce SyntaxError
        def _strip_code_fence(s: str) -> str:
            s = s.strip()
            # extract content from triple-backtick blocks
            import re

            m = re.search(
                r"```(?:python)?\n(.+?)```", s, flags=re.DOTALL | re.IGNORECASE
            )
            if m:
                return m.group(1).strip()
            # sometimes models return JSON like {"code": "..."}
            try:
                import json

                parsed = json.loads(s)
                if (
                    isinstance(parsed, dict)
                    and "code" in parsed
                    and isinstance(parsed["code"], str)
                ):
                    return parsed["code"].strip()
            except Exception:
                pass
            return s

        def _fix_common_unicode_quotes(s: str) -> str:
            return (
                s.replace("“", '"')
                .replace("”", '"')
                .replace("‘", "'")
                .replace("’", "'")
            )

        code = _strip_code_fence(code)
        code = _fix_common_unicode_quotes(code)

        # Try compiling first to give a clearer error and attempt minimal fixes
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            # As a fallback, try removing leading/trailing lines that often include
            # assistant messages like 'S:' or 'RESULT:'
            lines = code.splitlines()
            # drop leading non-indented short prefixes
            while lines and (
                lines[0].strip().endswith(":")
                or len(lines[0].strip()) <= 3
                and not lines[0]
                .lstrip()
                .startswith(("def ", "import ", "from ", "result", "pd", "np"))
            ):
                lines.pop(0)
            new_code = "\n".join(lines)
            new_code = _fix_common_unicode_quotes(new_code)
            try:
                compile(new_code, "<string>", "exec")
                code = new_code
            except SyntaxError:
                # surface a richer error message including the sanitized code
                raise SyntaxError(
                    f"Code compilation failed after sanitization: {e}; sanitized code:\n{new_code}"
                )

        env = {
            "pd": pd,
            "np": np,
            "scipy": scipy,
            "db_api": self.db_api,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
        }
        exec(code, env)

        return self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f"SELECT * FROM {result_table_id} LIMIT {self.config.MAX_RESULT_PREVIEW_ROWS};",
        )
