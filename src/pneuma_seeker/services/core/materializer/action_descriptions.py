from pneuma_seeker.services.core.action_set.main import ActionSet
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.config import Config


def get_materializer_actions(
    config: Config, action_set: ActionSet,
) -> str:
    return f"""
- {action_set.get_action_description(ActionNames.TABLE_RETRIEVE)}

- {get_table_enumeration_description(config)}

- {action_set.get_action_description(ActionNames.QUERY_EXECUTOR)}

- {get_python_executor_description()}

- {get_context_extraction_description() if config.ENABLE_CONTEXT_EXTRACTION else ""}

- {action_set.get_action_description(ActionNames.TABLE_PROJECTION)}

- {action_set.get_action_description(ActionNames.EQUALITY_JOIN)}

- {action_set.get_action_description(ActionNames.TABLE_UNION)}

{"- " + action_set.get_action_description(ActionNames.SEMANTIC_JOIN) + "\n" if config.ENABLE_SEMANTIC_JOIN else ""}
{"- " + action_set.get_action_description(ActionNames.SEMANTIC_COLUMN_GENERATION) + "\n" if config.ENABLE_SEMANTIC_COL_GEN else ""}
{"- " + action_set.get_action_description(ActionNames.WEB_SEARCH) + "\n" if config.ENABLE_WEB_SEARCH else ""}
{"- " + action_set.get_action_description(ActionNames.WEB_CRAWL) + "\n" if config.ENABLE_WEB_CRAWL else ""}""".strip()


def get_table_enumeration_description(config: Config) -> str:
    return f"""**{ActionNames.TABLE_ENUMERATION.value}**
    - **Precondition — MUST NOT be called unless there is at least one internal table already retrieved.**
    - Each pattern in `patterns` **must be derived from the names of existing internal tables** (or obvious common tokens in them).
    - Lists other available internal tables in the database whose names match given regex patterns.
    - This is useful when you retrieve one table (e.g., `topic_2020`) but suspect there are other related tables (`topic_2021`, `topic_2022`, etc.)
    - Enumerated tables will be unioned with internal tables retrieved via Table Retrieve.
    - Does not affect user-provided external data.
    - Args: {{"patterns": ["<regex pattern 1>", "<regex pattern 2>", ...]}}
    - Notes:
        - You may provide multiple patterns in a single call (at most {config.TABLE_RETRIEVE_MAX_TOPICS} patterns).
    - Example: {{"patterns": ["^sales_\\d{4}$", "^revenue_\\d{4}$"]}} will match all tables named like `sales_2020`, `sales_2021`, etc., and `revenue_2020`, `revenue_2021`, etc."""


def get_python_executor_description() -> str:
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
        - Never use escaped newlines (\n) inside strings.
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
    - You can perform operations such as renaming or reordering columns, transforming column values, normalizing formats (e.g., "Month Date, Year" → "yyyy-mm-dd"), etc.
    - You may create intermediate tables during processing (e.g., to store batches or results of sub-steps; don't forget to clean them up), but the final output must be materialized as a single table in the database using SQL (for example, `CREATE OR REPLACE TABLE "<assign_to>" AS SELECT ...`).
    - Do NOT return large tables as Pandas DataFrames. Any Pandas DataFrame should only be used for small previews or intermediate batch processing.
    - If your code creates the final output table using SQL (e.g., `CREATE OR REPLACE TABLE "<assign_to>" AS ...`), do **NOT** call `db_api.register_temporary_df(..., "<assign_to>")` with a preview DataFrame.
        - DuckDB's registration can shadow the persistent table with the same name, which would silently truncate downstream queries.
        - If you need a preview, query `SELECT * FROM "<assign_to>" LIMIT 10` and (optionally) register it under a different temporary name like `"<assign_to>__preview"`.
    - Args: {{"code": "<Python code string>", "assign_to": "<ID of the resulting intermediate table>"}}"""


def get_context_extraction_description():
    return f"""**{ActionNames.CONTEXT_EXTRACTION.value}**
    - Executes Python code to explore, inspect, or test assumptions about the data.
    - This action is used ONLY to gather evidence, perform sanity checks, or confirm suspicions. It has no lasting side effects.
    - It MUST NOT be used to construct final outputs or pipeline tables.
    - Tables are stored in the DuckDB-based workspace database and are NOT guaranteed to fit in memory.
    - To access tables, use the provided database API:
        - `db_api.execute_query(user_id, chat_id, "<SQL query>")`
        - `db_api.register_temporary_df(user_id, chat_id, df, "<table_name>")` can be used to register a small temporary Pandas DataFrame in the workspace for SQL queries.
        - `db_api`, `chat_id`, and `user_id` are available as variables in the environment. Do NOT import `db_api`; use `db_api.<function_name>`.
        - This returns a Pandas DataFrame containing the relational query result.
        - Note:
            - When referencing retrieved or enumerated tables, use the dataset-qualified name as provided by the system. For example, Table x (dataset: y) -> y."x".
                - Internally, `db_api` uses DuckDB ATTACH DATABASE to connect datasets to the workspace database. The attached dataset name (e.g., y) is a *catalog* (database), not a schema. Tables live under y.main.<table>, but DuckDB allows shorthand access as y.<table>.
              Do NOT treat the dataset name as a schema when inspecting metadata.
            - When referencing intermediate or external tables created in the workspace, use the table name directly. For example, Table x -> x.
    - Prefer performing inspection and checks directly in standard SQL whenever possible (counts, filters, group-bys, aggregates, sampling, detecting nulls, checking ranges, distributions, uniqueness, etc.) instead of loading tables into Pandas.
    - **Common pitfalls to avoid (important for reliability):**
        - Do not rely only on column-name substring matching. If a companion dictionary/variable-description table exists, query it directly for semantic clues and return candidate columns even if none of the column names match a simple regex.
        - When combining evidence from multiple sources (e.g., PRAGMA column list + dictionary rows), do not structure the result as a LEFT JOIN from a potentially empty base set. Prefer a UNION of candidate column names from both sources so you do not accidentally return an empty result.
        - Avoid reserved SQL keywords (e.g., "desc") as CTE names or aliases.
    - If Python processing is necessary, process data in small batches and never load full tables into memory.
    - When writing Python code:
        - Never use escaped newlines (\n) inside strings.
        - Use triple-quoted strings for multi-line SQL with real newlines.
        - The value of `code` MUST be raw Python source code, not a quoted string.
        - Never wrap Python code in quotes.
        - Do not construct Python code as strings for later execution (no exec-style indirection).
    - When using CTEs (WITH ...), attach them directly to the SELECT of a CREATE TABLE AS statement:
        CREATE OR REPLACE TABLE <name> AS
        WITH ...
        SELECT ...
    - Typical uses:
        - Checking whether a condition holds
        - Inspecting column value distributions or edge cases
        - Counting, filtering, sampling, or summarizing to confirm a belief
    - The final result of the inspection must be materialized into a temporary table named "materializer_assumption_check" using SQL (e.g., CREATE OR REPLACE TABLE "materializer_assumption_check" AS SELECT ...).
    - The content of "materializer_assumption_check" should be small and preview-sized (for example, aggregated statistics, samples, or at most a few rows).
    - The system will automatically read from "materializer_assumption_check", return at most the first 10 rows, and clean up the table after use.
    - Do NOT create any other persistent tables.
    - Args: {{\"code\": \"<Python code string>\"}}"""
