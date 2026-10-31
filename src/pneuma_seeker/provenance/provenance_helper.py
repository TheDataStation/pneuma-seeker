"""Code generators for the provenance graph.

Every provenance node carries a self-contained snippet of Python that, when the
snippets are concatenated in execution order after `SCRIPT_PREAMBLE`, replays
the materialization outside of Pneuma-Seeker:

- Source tables are loaded from CSV files placed next to the script into an
  in-memory DuckDB database that mirrors the chat's workspace (dataset tables
  live under a catalog named after the dataset, so `dataset."table"` refs in
  the replayed SQL resolve exactly as they did in Pneuma-Seeker).
- Deterministic steps (projection, joins, unions, SQL, Python) are re-executed
  verbatim.
- Model-driven steps (semantic column generation, semantic join, entity
  resolution) cannot be recomputed without the same models, so the decisions
  the model made are recorded in the snippet and replayed.
"""

import inspect
from typing import Any

from pneuma_seeker.shared.str_processor import clean_column_table_name


def quote_ident(identifier: str) -> str:
    """Quotes a DuckDB identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def table_ref(table_name: str, dataset_alias: str | None = None) -> str:
    """Returns the SQL reference of a table, dataset-qualified when applicable."""
    if dataset_alias:
        return f"{dataset_alias}.{quote_ident(table_name)}"
    return quote_ident(table_name)


def python_str_literal(text: str) -> str:
    """Returns a readable Python literal for `text` (raw triple-quoted when safe)."""
    for quote in ('"""', "'''"):
        if (
            quote not in text
            and not text.endswith(quote[0])
            and not text.endswith("\\")
            and "\r" not in text
        ):
            return f"r{quote}{text}{quote}"
    return repr(text)


def _rows_literal(rows: list[tuple[Any, ...]], indent: str = "        ") -> str:
    if not rows:
        return "[]"
    lines = "\n".join(f"{indent}{row!r}," for row in rows)
    return f"[\n{lines}\n{indent[:-4]}]"


def generate_load_table_code(table_name: str, dataset_alias: str | None) -> str:
    """Loads a source table (`<table_name>.csv`) into the workspace."""
    if dataset_alias:
        return f"load_csv_table({table_name!r}, dataset={dataset_alias!r})"
    return f"load_csv_table({table_name!r})"


def generate_sql_step_code(sql: str) -> str:
    """Re-executes a SQL statement against the workspace."""
    return f"run_sql(\n    {python_str_literal(sql.strip())}\n)"


def generate_view_textual_document_code(var_name: str, content: Any) -> str:
    """Keeps a textual document (web search/crawl result) as a Python variable."""
    return f"{var_name} = {python_str_literal(str(content))}"


def generate_semantic_column_replay_code(
    table_id: str,
    source_columns: list[str],
    new_column: str,
    generated_values: list[tuple[Any, ...]],
) -> str:
    """Replays an LLM-generated column from the values the LLM produced."""
    return (
        "replay_generated_column(\n"
        f"    table={table_id!r},\n"
        f"    source_columns={source_columns!r},\n"
        f"    new_column={new_column!r},\n"
        f"    # (value of each source column cast to text, ..., generated value)\n"
        f"    generated_values={_rows_literal(generated_values)},\n"
        ")"
    )


def generate_semantic_join_replay_code(
    left_ref: str,
    right_ref: str,
    left_columns: list[str],
    right_columns: list[str],
    result_table_id: str,
    matches: list[tuple[Any, ...]],
) -> str:
    """Replays a semantic join from the row matches it selected."""
    return (
        "replay_semantic_join(\n"
        f"    left={left_ref!r},\n"
        f"    right={right_ref!r},\n"
        f"    left_columns={left_columns!r},\n"
        f"    right_columns={right_columns!r},\n"
        f"    result={result_table_id!r},\n"
        f"    # (left column values as text, ..., right column values as text, ..., similarity_score)\n"
        f"    matches={_rows_literal(matches)},\n"
        ")"
    )


def generate_table_from_rows_code(
    table_id: str, columns: list[str], rows: list[tuple[Any, ...]]
) -> str:
    """Recreates a (small) model-produced table from its recorded rows."""
    return (
        "create_table_from_rows(\n"
        f"    table={table_id!r},\n"
        f"    columns={columns!r},\n"
        f"    rows={_rows_literal(rows)},\n"
        ")"
    )


def generate_snapshot_load_code(table_id: str) -> str:
    """Loads a step's result from `<table_id>.csv` (too large to record)."""
    return f"load_snapshot_table({table_id!r})"


def generate_unreplayable_step_code(reason: str) -> str:
    """Stops the script at a step whose effect could not be recorded."""
    return f"raise RuntimeError(\n    {reason!r}\n)"


def generate_export_code(target_table_ids: list[str]) -> str:
    """Writes the target tables to CSV files."""
    return f"export_tables({target_table_ids!r})"


SCRIPT_PREAMBLE = (
    '''import csv
import re
import sys
from pathlib import Path

import duckdb
import numpy as np  # noqa: F401  (available to Python steps, as in Pneuma-Seeker)
import pandas as pd

try:
    import scipy  # noqa: F401  (available to Python steps, as in Pneuma-Seeker)
except ImportError:
    scipy = None

# Folder with the input CSV files: the first CLI argument, else this script's folder.
DATA_DIR = (
    Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
)
OUTPUT_DIR = DATA_DIR / "output"

# In-memory stand-in for the chat's workspace database.
_con = duckdb.connect()


class _WorkspaceDB:
    """Mimics Pneuma-Seeker's `db_api` so Python steps run unchanged."""

    def execute_query(self, user_id, chat_id, sql, sql_params=()):
        return _con.execute(sql, sql_params).fetchdf()

    def register_temporary_df(self, user_id, chat_id, df, table_name):
        try:
            _con.unregister(table_name)
        except Exception:
            pass
        _con.register(table_name, df)


db_api = _WorkspaceDB()
user_id = chat_id = "local"


def _quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def run_sql(sql):
    """Executes a SQL statement against the workspace."""
    return _con.execute(sql).fetchdf()


'''
    + inspect.getsource(clean_column_table_name)
    + '''

def _dedupe(names):
    seen, result = {}, []
    for name in names:
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)
    return result


def _find_csv(table, dataset):
    folders = [DATA_DIR] + ([DATA_DIR / dataset] if dataset else [])
    for folder in folders:
        exact = folder / f"{table}.csv"
        if exact.is_file():
            return exact
        for candidate in sorted(folder.glob("*.csv")):
            if clean_column_table_name(candidate.stem) == table:
                return candidate
    raise FileNotFoundError(
        f"Missing input file for table {table!r}: put {table}.csv in {DATA_DIR}"
    )


def load_csv_table(table, dataset=None):
    """Loads `<table>.csv` the same way Pneuma-Seeker ingests datasets (cleaned
    column names). Dataset tables are stored under a catalog named after the
    dataset so `dataset."table"` references keep working."""
    path = _find_csv(table, dataset)
    with open(path, "r", encoding="utf-8-sig") as f:
        original_cols = next(csv.reader([f.readline().strip()]))
    cleaned_cols = _dedupe([clean_column_table_name(c) for c in original_cols])
    select = ", ".join(
        f"{_quote(orig)} AS {_quote(clean)}"
        for orig, clean in zip(original_cols, cleaned_cols)
    )
    if dataset:
        attached = _con.execute("SELECT database_name FROM duckdb_databases()").fetchall()
        if dataset not in {row[0] for row in attached}:
            _con.execute(f"ATTACH ':memory:' AS {_quote(dataset)}")
        target = f"{_quote(dataset)}.{_quote(table)}"
    else:
        target = _quote(table)
    escaped_path = path.as_posix().replace("'", "''")
    _con.execute(
        f"""CREATE OR REPLACE TABLE {target} AS
        SELECT {select}
        FROM read_csv_auto('{escaped_path}', HEADER=TRUE, IGNORE_ERRORS=TRUE,
            STRICT_MODE=FALSE, NULL_PADDING=TRUE, SAMPLE_SIZE=100_000, PARALLEL=FALSE)"""
    )
    print(f"Loaded {path.name} -> {target}")


def load_snapshot_table(table):
    """Loads a step's result that was too large to embed in this script from
    `<table>.csv` (part of the "Download views" ZIP in Pneuma-Seeker). The CSV
    holds the table's final state in the chat."""
    path = DATA_DIR / f"{table}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing snapshot {path.name}: download the views of this chat in "
            f"Pneuma-Seeker and put {path.name} in {DATA_DIR}"
        )
    escaped_path = path.as_posix().replace("'", "''")
    _con.execute(
        f"CREATE OR REPLACE TABLE {_quote(table)} AS "
        f"SELECT * FROM read_csv_auto('{escaped_path}', HEADER=TRUE)"
    )
    print(f"Loaded snapshot {path.name} -> {_quote(table)}")


def replay_generated_column(table, source_columns, new_column, generated_values):
    """Adds `new_column` to `table` using the values an LLM generated for each
    combination of source column values."""
    if not generated_values:
        run_sql(
            f"CREATE OR REPLACE TABLE {_quote(table)} AS "
            f"SELECT *, NULL::VARCHAR AS {_quote(new_column)} FROM {_quote(table)}"
        )
        return
    key_cols = [f"__src_{i}" for i in range(len(source_columns))]
    mapping = pd.DataFrame(generated_values, columns=key_cols + ["__value"], dtype=object)
    _con.register("__replay_mapping", mapping)
    on = " AND ".join(
        f'CAST(t.{_quote(col)} AS VARCHAR) IS NOT DISTINCT FROM m."{key}"'
        for col, key in zip(source_columns, key_cols)
    )
    run_sql(
        f"CREATE OR REPLACE TABLE {_quote(table)} AS "
        f'SELECT t.*, CAST(m."__value" AS VARCHAR) AS {_quote(new_column)} '
        f"FROM {_quote(table)} AS t LEFT JOIN __replay_mapping AS m ON {on} "
        f"ORDER BY t.rowid"
    )
    _con.unregister("__replay_mapping")


def replay_semantic_join(left, right, left_columns, right_columns, result, matches):
    """Rebuilds a semantic join result from the (left row, right row) matches it
    selected. Output columns are `left_<col>`, `right_<col>` and `similarity_score`."""
    left_all = list(run_sql(f"SELECT * FROM {left} LIMIT 0").columns)
    right_all = list(run_sql(f"SELECT * FROM {right} LIMIT 0").columns)
    select = ", ".join(
        [f"l.{_quote(c)} AS {_quote('left_' + c)}" for c in left_all]
        + [f"r.{_quote(c)} AS {_quote('right_' + c)}" for c in right_all]
    )
    left_keys = [f"__l_{i}" for i in range(len(left_columns))]
    right_keys = [f"__r_{i}" for i in range(len(right_columns))]
    pairs = pd.DataFrame(
        matches, columns=left_keys + right_keys + ["__score"], dtype=object
    )
    pairs["__score"] = pd.to_numeric(pairs["__score"])
    _con.register("__replay_matches", pairs)
    left_on = " AND ".join(
        f'CAST(l.{_quote(c)} AS VARCHAR) IS NOT DISTINCT FROM p."{k}"'
        for c, k in zip(left_columns, left_keys)
    )
    right_on = " AND ".join(
        f'CAST(r.{_quote(c)} AS VARCHAR) IS NOT DISTINCT FROM p."{k}"'
        for c, k in zip(right_columns, right_keys)
    )
    run_sql(
        f"CREATE OR REPLACE TABLE {_quote(result)} AS "
        f'SELECT {select}, CAST(p."__score" AS DOUBLE) AS similarity_score '
        f"FROM __replay_matches AS p "
        f"JOIN {left} AS l ON {left_on} "
        f"JOIN {right} AS r ON {right_on} "
        f"ORDER BY l.rowid, similarity_score DESC"
    )
    _con.unregister("__replay_matches")


def create_table_from_rows(table, columns, rows):
    """Recreates a table from recorded rows."""
    df = pd.DataFrame(rows, columns=columns, dtype=object)
    _con.register("__replay_rows", df)
    cols = ", ".join(f"CAST({_quote(c)} AS VARCHAR) AS {_quote(c)}" for c in columns)
    run_sql(f"CREATE OR REPLACE TABLE {_quote(table)} AS SELECT {cols} FROM __replay_rows")
    _con.unregister("__replay_rows")


def export_tables(table_ids):
    """Writes each table to OUTPUT_DIR/<table>.csv."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for table in table_ids:
        out = OUTPUT_DIR / f"{table}.csv"
        df = run_sql(f"SELECT * FROM {_quote(table)}")
        df.to_csv(out, index=False)
        print(f"Wrote {out} ({len(df)} rows)")'''
)
