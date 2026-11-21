"""
Holds the rules every prompt asking for SQL or Python on the workspace shares, written
once so the prompts stay consistent. Prompts insert them next to their own rules.

- DuckDB semantics that differ from the dialects language models mostly learned
  (PostgreSQL, SQLite, SQL Server).
- The workspace: where tables live and how code reaches them.
"""

# DuckDB semantics
DUCKDB_DIVISION = (
    "DuckDB's `/` is decimal division even between integers (`7 / 2` is `3.5`); "
    "use `//` for integer division."
)

# The workspace
WORKSPACE_TABLES = (
    "Tables are stored in the DuckDB-based workspace database and are NOT guaranteed to fit in memory."
)
DB_API_VARIABLES = (
    "`db_api`, `user_id`, and `chat_id` are available as variables. "
    "Do NOT import `db_api`; use `db_api.<function_name>`."
)
DATASET_QUALIFIED = (
    "When referencing retrieved or enumerated tables, use the dataset-qualified name as provided. "
    'For example, Table x (dataset: y) -> y."x".'
)
PREFER_SQL = "Prefer performing transformations directly in SQL instead of loading tables with Pandas."
BATCHES = "If Python processing is necessary, process data in small batches and never load full tables into memory."
NO_NEW_CONNECTION = (
    "Do NOT create a new DuckDB connection (no `duckdb.connect()`): it is not linked to the "
    "workspace database, so tables created on it are not visible to the system."
)
NO_ESCAPED_NEWLINES = (
    "Never use escaped newlines (\\n) inside strings; use triple-quoted strings for multi-line SQL "
    "with real newlines."
)
CTE_IN_CREATE = (
    "When using CTEs (WITH ...), attach them directly to the SELECT of a CREATE TABLE AS statement: "
    "`CREATE OR REPLACE TABLE <name> AS WITH ... SELECT ...`."
)
