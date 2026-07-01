# Execution Skills

Use execution skills to run code against workspace tables. Prefer `run_sql` when standard SQL
can express the transformation; fall back to `run_python` for logic SQL cannot express cleanly.

## run_sql

Execute a SQL SELECT query in the workspace DuckDB database. The result is automatically
persisted as `result_table_id` and can be used by later skills.

**Args:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | str | yes | A SQL SELECT (or WITH…SELECT). All retrieved and workspace table IDs are available as-is. |
| `result_table_id` | str | no | Name for the output table. Default: `sql_result`. |

**Returns:** Row count + first 10 rows preview.

---

## run_python

Execute Python code in the workspace. The code must create a DuckDB table named
`result_table_id` via `db_api.execute_query(user_id, chat_id, "CREATE OR REPLACE TABLE ...")`.

**Args:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | str | yes | Python source. Available variables: `db_api`, `user_id`, `chat_id`. Use `db_api.execute_query(user_id, chat_id, sql)` to read/write tables. |
| `result_table_id` | str | no | Name for the output table the code must create. Default: `python_result`. |

**Returns:** Row count + first 10 rows preview.

**Notes:**
- Allowed libraries: Pandas, NumPy, SciPy.
- Do NOT call `duckdb.connect()` — workspace tables are only visible through `db_api`.
- Never load large tables fully into memory; process in batches.
