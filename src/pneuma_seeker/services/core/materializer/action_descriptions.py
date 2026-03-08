from pneuma_seeker.services.core.action_set.main import ActionSet
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.config import Config


def get_materializer_actions(
    config: Config, action_set: ActionSet,
) -> str:
    return f"""
{get_table_retrieve_description(config)}
{get_table_enumeration_description(config)}
{get_query_executor_description()}
{get_python_executor_description()}
{get_context_extraction_description() if config.ENABLE_CONTEXT_EXTRACTION else ""}
{get_table_projection_description()}
{get_equality_join_description()}
{get_table_union_description()}
{get_semantic_join_description() if config.ENABLE_SEMANTIC_JOIN else ""}
{get_semantic_col_gen_description() if config.ENABLE_SEMANTIC_COL_GEN else ""}
{get_web_search_description() if config.ENABLE_WEB_SEARCH else ""}
{get_web_crawl_description() if config.ENABLE_WEB_CRAWL else ""}""".strip()


def get_semantic_join_description():
    return f"""- **{ActionNames.SEMANTIC_JOIN.value}**
        - Joins two tables (internal, external, or intermediate) by computing semantic similarity between specified columns.
        - Similarity uses a weighted combination of embedding cosine similarity and normalized Damerau-Levenshtein edit similarity.
        - Produces a new joined table containing matched rows and a similarity_score column.
        - Use case: when the user explicitly asks for it, when two tables contain related entities that do not match exactly by key or text (e.g., "Intl Business Machines" vs. "IBM"), or when there are no potential join paths.
            Even if both tables share a key column (e.g., "product_id"), the user may prefer semantic matching — for instance, comparing product descriptions between catalogs from different years to detect essentially identical products that were renumbered but now sold at different prices.
        - Args: {{
                "left_table_id": "<ID of left table (must exist in retrieved or intermediate tables)>",
                "right_table_id": "<ID of right table (must exist in retrieved or intermediate tables)>",
                "relevant_left_cols": ["<list of columns from left table used for semantic comparison>"],
                "relevant_right_cols": ["<list of columns from right table used for semantic comparison>"],
                "joined_table_id": "<ID to store the resulting joined table>"
            }}
        - Example: {{
                "left_table_id": "companies_2024",
                "right_table_id": "clients_2024",
                "relevant_left_cols": ["company_name", "headquarters_city"],
                "relevant_right_cols": ["client_name", "hq_location"],
                "joined_table_id": "company_client_matches"
            }}\n"""


def get_semantic_col_gen_description():
    return f"""- **{ActionNames.SEMANTIC_COLUMN_GENERATION.value}**
        - Adds a new column to an *intermediate* table using an LLM.
        - The column is derived from specified `relevant_columns` only — no other columns are used.
        - External and internal tables should first be transformed into intermediate tables if new columns are needed, because retrieved internal tables can be replaced.
        - Args: {{
                "table_id": "<intermediate_table_id>",
                "new_column_name": "<column to add>",
                "relevant_columns": ["<list of source columns for generation>"],
                "instruction": "<instruction describing how to generate the new column values>"
            }}
        - Example: {{
                "table_id": "products_2024",
                "new_column_name": "category",
                "relevant_columns": ["product_name", "description"],
                "instruction": "Classify each product into 'Electronics', 'Furniture', or 'Clothing'."
            }}\n"""


def get_table_enumeration_description(config: Config) -> str:
    return f"""- **{ActionNames.TABLE_ENUMERATION.value}**
    - **Precondition — MUST NOT be called unless there is at least one internal table already retrieved.**
    - Each pattern in `patterns` **must be derived from the names of existing internal tables** (or obvious common tokens in them).
    - Lists other available internal tables in the database whose names match given regex patterns.
    - This is useful when you retrieve one table (e.g., `topic_2020`) but suspect there are other related tables (`topic_2021`, `topic_2022`, etc.)
    - Enumerated tables will be unioned with internal tables retrieved via Table Retrieve.
    - Does not affect user-provided external data.
    - Args: {{"patterns": ["<regex pattern 1>", "<regex pattern 2>", ...]}}
    - Notes:
        - You may provide multiple patterns in a single call (at most {config.TABLE_RETRIEVE_MAX_TOPICS} patterns).
    - Example: {{"patterns": ["^sales_\\d{4}$", "^revenue_\\d{4}$"]}} will match all tables named like `sales_2020`, `sales_2021`, etc., and `revenue_2020`, `revenue_2021`, etc.\n"""


def get_query_executor_description() -> str:
    return f"""- **{ActionNames.QUERY_EXECUTOR.value}**
    - Executes a single SQL query and materializes the result into a workspace table.
    - The system will run: `CREATE OR REPLACE TABLE "<assign_to>" AS <query>`, then returns a preview with `SELECT * FROM "<assign_to>" LIMIT 10`.
    - Input guidelines:
        - `query` should generally start with `SELECT ...` or `WITH ... SELECT ...`.
        - Provide only one SQL statement (no embedded `;`).
        - When referencing retrieved/enumerated tables, use dataset-qualified names like `y."x"`.
        - When referencing intermediate or external tables created in the workspace, use the table name directly.
    - Args: {{"query": "<SQL query>", "assign_to": "<ID of the resulting intermediate table>"}}\n"""


def get_python_executor_description() -> str:
    return f"""- **{ActionNames.PYTHON_EXECUTOR.value}**
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
    - Args: {{"code": "<Python code string>", "assign_to": "<ID of the resulting intermediate table>"}}\n"""


def get_table_projection_description() -> str:
    return f"""- **{ActionNames.TABLE_PROJECTION.value}**
    - Projects a table (internal, external, or intermediate) to a subset of its columns, optionally renaming columns at the same time.
    - The output table is materialized into the workspace database.
    - Args:
    {{
        "<target_table_id>": {{
            "id": "<source_table_id>",
            "columns": {{"<output_col_1>": "<source_col_1>", "<output_col_2>": "<source_col_2>"}}
        }}
    }}
    - Notes:
        - If the source table is an internal table that was retrieved (i.e., the table has an ID like "Table x (dataset: y)"), reference it as a dataset-qualified name like `y."x"`.
        - If the source table is an intermediate/external table created in the workspace, reference it by its workspace name directly (e.g., `my_intermediate_table`).
        - The `columns` mapping is **output_column_name -> source_column_name** (use this to rename columns during projection).
        - The output table (`target_table_id`) is always created/overwritten in the workspace.\n"""


def get_equality_join_description() -> str:
    return f"""- **{ActionNames.EQUALITY_JOIN.value}**
    - Joins two tables (internal, external, or intermediate) by exact equality on specified key columns.
    - The output table is materialized into the workspace database.
    - Args:
    {{
        "left_table_id": "<left table reference>",
        "right_table_id": "<right table reference>",
        "left_table_column_keys": ["<left join key col 1>", "<left join key col 2>", ...],
        "right_table_column_keys": ["<right join key col 1>", "<right join key col 2>", ...],
        "result_table_id": "<intermediate/target table name for the join result>"
    }}
    - Notes:
        - If a table is an internal table that was retrieved (i.e., the table has an ID like "Table x (dataset: y)"), reference it as a dataset-qualified name like `y."x"`.
        - If a table is an intermediate/external table created during processing, reference it by its workspace name directly (e.g., `my_intermediate_table`).
        - The key lists must be non-empty and the same length; keys are matched positionally."""


def get_table_union_description() -> str:
    return f"""- **{ActionNames.TABLE_UNION.value}**
    - Unions multiple tables (internal, external, or intermediate) into a single workspace table.
    - Each entry in `table_ids` may be either:
        - an explicit table reference (e.g., `my_intermediate_table` or `y."x"`), OR
        - a regex selector prefixed with `re:` (e.g., `re:^data_\\d{4}$` or `re:^y\\.data_\\d{4}$`).
    - The output includes a provenance column derived from each source table ID.
    - Args: {{
        "table_ids": ["<table ref or re:<pattern>>", ...],
        "result_table_id": "<intermediate/target table name for the union result>",
        "provenance_column_name": "<output column name for provenance>",
        "provenance_regex": "<regex used to extract provenance from each table id/name>"
      }}
    - Notes:
        - Regex patterns are matched against both the full display name (e.g., `y.data_2021`) and the bare table name (e.g., `data_2021`).
        - If tables have different schemas, missing columns are filled with NULL.
        - The output table (`result_table_id`) is always created/overwritten in the workspace."""


def get_table_retrieve_description(config: Config) -> str:
    return f"""- **{ActionNames.TABLE_RETRIEVE.value}**:
    Retrieve internal tables.
    - **Args**: {{"prompts": ["<retrieval query 1>", "<retrieval query 2>", ...]}}
    - **Notes**:
        - You may provide multiple retrieval queries in a single call to retrieve tables on different topics (at most {config.TABLE_RETRIEVE_MAX_TOPICS} topics).
        - Previously retrieved tables will be replaced with new retrievals; does not affect user-provided external tables.
        - Potential join paths between retrieved tables will be provided for reference.\n"""


def get_context_extraction_description():
    return f"""\n- **{ActionNames.CONTEXT_EXTRACTION.value}**
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
    - Args: {{\"code\": \"<Python code string>\"}}\n"""


def get_web_search_description():
    """Gets the optional web search description for the Materializer."""
    return f"""\n- **{ActionNames.WEB_SEARCH.value}**
    - Retrieves information from the web to assist in filling tables when internal and external data are insufficient.
    - Args: {{"prompt": "<query describing what data to retrieve or clarify>"}}
    - Usage notes:
        - Use web_search only when no reliable internal/external source exists for the required column(s).
        - Avoid repetitive or redundant queries.\n"""


def get_web_crawl_description():
    """Gets the optional web crawl description for the Materializer."""
    return f"""\n- **{ActionNames.WEB_CRAWL.value}**
    - Crawls a specified web page to extract textual content for table materialization.
    - Args: {{"url": "<URL of the web page to crawl>"}}
    - Usage notes:
        - Use this when the user specifically requests information from a particular URL.
        - The crawler respects robots.txt and will not fetch disallowed paths.
        - Returned content is raw extracted text from the page (no summarization).\n"""
