from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.shared.schemas.core.agent import AgentType

_SHARED_PREAMBLE = f"""**{ActionNames.CONTEXT_EXTRACTION.value}**
  - Executes Python code to explore, inspect, or test assumptions or relevance of the retrieved or external tables.
  - This action is used ONLY to gather evidence, perform sanity checks, or confirm suspicions. It has no lasting side effects.
  - It MUST NOT be used to construct final outputs or pipeline tables.
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
        - When referencing target tables in T or external tables, use the table name directly. For example, Table x -> x.
  - Prefer performing inspection and checks directly in standard SQL whenever possible instead of loading tables into Pandas.
  - **Common pitfalls to avoid (important for reliability):**
    - Do not rely only on column-name substring matching. If a companion dictionary/variable-description table exists, query it directly for semantic clues and return candidate columns even if none of the column names match a simple regex.
    - When combining evidence from multiple sources (e.g., PRAGMA column list + dictionary rows), do not structure the result as a LEFT JOIN from a potentially empty base set. Prefer a UNION of candidate column names from both sources so you do not accidentally return an empty result.
    - Avoid reserved SQL keywords (e.g., "desc") as CTE names or aliases.
  - If Python processing is necessary, process data in small batches and never load full tables into memory.
  - When writing Python code:
    - Never use escaped newlines (\\n) inside strings.
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
    - Counting, filtering, sampling, or summarizing to confirm a belief"""

_CONDUCTOR_SUFFIX = """
  - The final result of the inspection must be materialized into a temporary table named "conductor_assumption_check" using SQL (e.g., CREATE OR REPLACE TABLE "conductor_assumption_check" AS SELECT ...).
  - The content of "conductor_assumption_check" should be small and preview-sized (for example, aggregated statistics, samples, or at most a few rows).
  - The system will automatically read from "conductor_assumption_check", return at most the first 10 rows, and clean up the table after use.
  - Do NOT create any other persistent tables.
  - Args: {"code": "<Python code string>"}"""

_MATERIALIZER_SUFFIX = """
  - The final result of the inspection must be materialized into a temporary table named "materializer_assumption_check" using SQL (e.g., CREATE OR REPLACE TABLE "materializer_assumption_check" AS SELECT ...).
  - The content of "materializer_assumption_check" should be small and preview-sized (for example, aggregated statistics, samples, or at most a few rows).
  - The system will automatically read from "materializer_assumption_check", return at most the first 10 rows, and clean up the table after use.
  - Do NOT create any other persistent tables.
  - Args: {"code": "<Python code string>"}"""


class ContextExtraction(Action):
    action_name = ActionNames.CONTEXT_EXTRACTION
    agents = frozenset({AgentType.CONDUCTOR, AgentType.MATERIALIZER})
    flag = "ENABLE_CONTEXT_EXTRACTION"
    order = 7
    show_in_prompt = True

    def get_description(self, agent: AgentType | None = None) -> str:
        if agent == AgentType.MATERIALIZER:
            return _SHARED_PREAMBLE + _MATERIALIZER_SUFFIX
        return _SHARED_PREAMBLE + _CONDUCTOR_SUFFIX
