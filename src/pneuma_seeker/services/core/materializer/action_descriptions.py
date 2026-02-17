from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.config import Config


def get_materializer_actions(
    config: Config,
) -> str:
    return (
        f"""
{get_table_retrieve_description(config)}

- **{ActionNames.TABLE_ENUMERATION.value}**
    - **Precondition — MUST NOT be called unless there is at least one internal table already retrieved.**
    - The `pattern` argument **must be derived from the names of existing internal tables** (or obvious common tokens in them).
    - Lists other available internal tables in the database whose names match a given regex pattern.
    - This is useful when you retrieve one table (e.g., `topic_2020`) but suspect there are other related tables (`topic_2021`, `topic_2022`, etc.)
    - Does not affect user-provided external data.
    - Args: {{"pattern": "<regex pattern to match table names>"}}
    - Example: {{"pattern": "^sales_\\d{4}$"}} will match all tables named like `sales_2020`, `sales_2021`, etc.

- **{ActionNames.PYTHON_EXECUTOR.value}**
    - Executes Python code to transform and/or combine data.
    - Tables are stored in the DuckDB-based workspace database and are NOT guaranteed to fit in memory.
    - To access tables, use the provided database API:
        - `db_api.execute_query(user_id, chat_id, "<SQL query>")`
        - `db_api.register_temporary_df(user_id, chat_id, df, "<table_name>")` can be used to register a small temporary Pandas DataFrame in the workspace for SQL queries.
        - `db_api`, `chat_id`, and `user_id` are available as variables in the environment.
        - This returns a Pandas DataFrame containing the relational query result.
        - NOTE: Solely for the purpose of referencing tables in SQL queries, if a retrieved table has an ID like "Table x (dataset: y)", treat "y" as the schema and reference the table in SQL as SELECT * FROM y."x".
    - Prefer performing transformations directly in standard SQL whenever possible (filtering, projection, joins, aggregation, casting, renaming columns, value normalization, date parsing, etc.) instead of loading tables into Pandas.
    - If Python processing is necessary, process data in batches and never load full tables into memory. For example:
        - Offset pagination:
            - SELECT * FROM "<table_id>" ORDER BY rowid LIMIT 100 OFFSET 0;
            - SELECT * FROM "<table_id>" ORDER BY rowid LIMIT 100 OFFSET 100;
        - Keyset pagination (preferred for large tables):
            - SELECT * FROM "<table_id>" WHERE rowid > last_seen_rowid ORDER BY rowid LIMIT 100;
    - Pandas, NumPy, and SciPy are available for small, intermediate, in-memory batches only (remember to add relevant import statements if needed).
    - You can perform operations such as renaming or reordering columns, transforming column values, normalizing formats (e.g., "Month Date, Year" → "yyyy-mm-dd"), etc.
    - You may create intermediate tables during processing (e.g., to store batches or results of sub-steps; don't forget to clean them up), but the final output must be materialized as a single table in the database using SQL (for example, `CREATE OR REPLACE TABLE "<assign_to>" AS SELECT ...`).
    - Do NOT return large tables as Pandas DataFrames. Any Pandas DataFrame should only be used for small previews or intermediate batch processing.
    - Args: {{"code": "<Python code string>", "assign_to": "<ID of the resulting intermediate table>"}}
{get_assumption_check_description() if config.ENABLE_ASSUMPTION_CHECK else ""}

- **{ActionNames.TABLE_PROJECTION.value}**
    - Directly maps an existing table (internal, external, or intermediate) to a target table (or a subset of its columns).
    - Args: {{"<target_table_id>": {{
                    {{
                        "id": "<source_table_id>",
                        "columns": ["<subset of columns from source table to use>"]
                    }}
                }}
            }}
    - Note: If the source table is an internal table that was retrieved (i.e., the table has an ID like "Table x (dataset: y)"), reference it by "y.\"x\"" in the `id` field.
    - Example use case: If table A has columns that match some columns of target table B, you can select it directly instead of creating Python code.

- **{ActionNames.SEMANTIC_JOIN.value}**
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
      }}

- **{ActionNames.SEMANTIC_COLUMN_GENERATION.value}**
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
      }}
""".strip()
        + (get_web_search_description() if config.ENABLE_WEB_SEARCH else "")
        + (get_web_crawl_description() if config.ENABLE_WEB_CRAWL else "")
    )


def get_table_retrieve_description(config: Config) -> str:
    if not config.ENABLE_MULTI_TOPIC_TABLE_RETRIEVE:
        return f"""- **{ActionNames.TABLE_RETRIEVE.value}**:
    Retrieve internal tables.
    - **Args**: {{"prompt": "<retrieval query>"}}
    - **Notes**:
        - Avoid retrying the same or slightly modified queries repeatedly.
        - However, for different topics or aspects of an information need, feel free to call multiple times.
        - Previously retrieved tables will be replaced with new retrievals; does not affect user-provided external tables.
        {"- Potential join paths between retrieved tables will be provided for reference." if config.ENABLE_JOIN_PATH_EXTRACTION else ""}"""
    else:
        return f"""- **{ActionNames.TABLE_RETRIEVE.value}**:
    Retrieve internal tables.
    - **Args**: {{"prompts": "[<retrieval query 1>, <retrieval query 2>, ...]"}}
    - **Notes**:
        - You may provide multiple retrieval queries in a single call to retrieve tables on different topics (at most {config.TABLE_RETRIEVE_MAX_TOPICS} topics).
        - Previously retrieved tables will be replaced with new retrievals; does not affect user-provided external tables.
        {"- Potential join paths between retrieved tables will be provided for reference." if config.ENABLE_JOIN_PATH_EXTRACTION else ""}"""


def get_assumption_check_description():
    return f"""\n- **{ActionNames.ASSUMPTION_CHECK.value}**
    - Executes Python code to explore, inspect, or test assumptions about the data.
    - This tool is used ONLY to gather evidence, perform sanity checks, or confirm suspicions. It has no lasting side effects.
    - It MUST NOT be used to construct final outputs or pipeline tables.
    - Tables are stored in the DuckDB-based workspace database and are NOT guaranteed to fit in memory.
    - To access tables, use the provided database API:
        - `db_api.execute_query(user_id, chat_id, "<SQL query>")`
        - `db_api.register_temporary_df(user_id, chat_id, df, "<table_name>")` can be used to register a small temporary Pandas DataFrame in the workspace for SQL queries.
        - `db_api`, `chat_id`, and `user_id` are available as variables in the environment.
        - This returns a Pandas DataFrame containing the relational query result.
        - NOTE: Solely for the purpose of referencing tables in SQL queries, if a retrieved table has an ID like "Table x (dataset: y)", treat "y" as the schema and reference the table in SQL as SELECT * FROM y."x".
    - Prefer performing inspection and checks directly in standard SQL whenever possible (counts, filters, group-bys, aggregates, sampling, detecting nulls, checking ranges, distributions, uniqueness, etc.) instead of loading tables into Pandas.
    - If Python processing is necessary, process data in small batches and never load full tables into memory.
    - Typical uses:
        - Checking whether a condition holds
        - Inspecting column value distributions or edge cases
        - Counting, filtering, sampling, or summarizing to confirm a belief
    - The final result of the inspection must be materialized into a temporary table named "materializer_assumption_check" using SQL (e.g., CREATE OR REPLACE TABLE "materializer_assumption_check" AS SELECT ...).
    - The content of "materializer_assumption_check" should be small and preview-sized (for example, aggregated statistics, samples, or at most a few rows).
    - The system will automatically read from "materializer_assumption_check", return at most the first 5 rows, and clean up the table after use.
    - Do NOT create any other persistent tables.
    - Args: {{\"code\": \"<Python code string>\"}}"""


def get_web_search_description():
    """Gets the optional web search description for the Materializer."""
    return f"""\n- **{ActionNames.WEB_SEARCH.value}**
    - Retrieves information from the web to assist in filling tables when internal and external data are insufficient.
    - Args: {{"prompt": "<query describing what data to retrieve or clarify>"}}
    - Usage notes:
        - Use web_search only when no reliable internal/external source exists for the required column(s).
        - Avoid repetitive or redundant queries."""


def get_web_crawl_description():
    """Gets the optional web crawl description for the Materializer."""
    return f"""\n- **{ActionNames.WEB_CRAWL.value}**
    - Crawls a specified web page to extract textual content for table materialization.
    - Args: {{"url": "<URL of the web page to crawl>"}}
    - Usage notes:
        - Use this when the user specifically requests information from a particular URL.
        - The crawler respects robots.txt and will not fetch disallowed paths.
        - Returned content is raw extracted text from the page (no summarization)."""
