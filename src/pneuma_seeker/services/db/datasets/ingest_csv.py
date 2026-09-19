import os
import re
from pathlib import Path

import duckdb
from pandas import read_csv
from tqdm import tqdm


def clean_column_table_name(name):
    """Cleans and normalizes column/table names."""
    name = name.strip()
    name = name.lower()
    # Semantic replacements
    name = name.replace("$", "usd")
    name = name.replace("%", "percentage")
    name = name.replace("#", "num")
    # Replace spaces and hyphens with underscores
    name = name.replace("-", "_").replace(" ", "_")
    # Replace "(" and ")" with underscores
    name = name.replace("(", "_").replace(")", "_")
    # Remove anything that's not a letter, digit, or underscore
    name = re.sub(r"[^0-9a-z_]", "_", name)
    # Collapse multiple underscores into one
    name = re.sub(r"_+", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    return name


def dedupe_columns(cols):
    seen = {}
    result = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            result.append(c)
        else:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")
    return result


def sniff_csv_dialect_options(dataset_con, file_path: str) -> str:
    """Returns explicit DELIM/QUOTE/ESCAPE options to pin a CSV's dialect.

    STRICT_MODE=FALSE is needed so a few malformed rows don't abort a whole
    file, but it also relaxes DuckDB's dialect sniffer, which can then settle
    on a quoting scheme that silently mis-parses a correctly-quoted file and
    drops large contiguous blocks of rows. Sniffing separately (sniff_csv runs
    strict) and pinning the result keeps the tolerant row handling without
    letting the dialect drift.
    """
    try:
        delim, quote, escape = dataset_con.execute(
            "SELECT Delimiter, Quote, Escape FROM sniff_csv(?)", [file_path]
        ).fetchone()
    except Exception:
        return ""
    options = []
    for name, value in (("DELIM", delim), ("QUOTE", quote), ("ESCAPE", escape)):
        # sniff_csv reports absent settings as a multi-char marker, e.g. "(empty)"
        if isinstance(value, str) and len(value) == 1:
            options.append(f"""{name}='{value.replace("'", "''")}'""")
    return ("," + ", ".join(options)) if options else ""


DATASET_NAME = "proc_spend"
DATASET_PATH = f"../../../../../data_src/{DATASET_NAME}/dataset"
OVERWRITE_DB = True

if os.path.exists(f"{DATASET_NAME}.db") and not OVERWRITE_DB:
    raise FileExistsError(
        f"{DATASET_NAME}.db already exists. Aborting to prevent overwrite."
    )
else:
    if os.path.exists(f"{DATASET_NAME}.db") and OVERWRITE_DB:
        os.remove(f"{DATASET_NAME}.db")
    print(f"Ingesting CSV files from {DATASET_PATH} into {DATASET_NAME}.db")

    dataset_con = duckdb.connect(f"{DATASET_NAME}.db")
    try:
        for table_file_name in tqdm(sorted(os.listdir(DATASET_PATH))):
            if not table_file_name.lower().endswith(".csv"):
                continue

            print(f"Ingesting {table_file_name}...")

            file_path = (Path(DATASET_PATH) / table_file_name).as_posix()
            table_stem = Path(table_file_name).stem
            cleaned_table_name = clean_column_table_name(table_stem)

            # Read header only to get original column names (fast)
            try:
                rel = read_csv(file_path, nrows=1)
                original_cols = list(rel.columns)
            except Exception:
                # Fallback: let DuckDB auto-detect and ingest (still fine)
                original_cols = None

            dialect_options = sniff_csv_dialect_options(dataset_con, file_path)

            if original_cols:
                cleaned_cols = dedupe_columns(
                    [clean_column_table_name(c) for c in original_cols]
                )

                # 1. Create a clean mapping dictionary for DuckDB to use
                #    e.g., {'School_ID': 'school_id', 'Legacy_Unit_ID': 'legacy_unit_id'}
                mapping_dict = dict(zip(original_cols, cleaned_cols))

                # 2. Use DuckDB's RENAME modifier to map original names to cleaned names
                rename_clause = ", ".join(
                    f'"{orig}" AS "{cleaned}"' for orig, cleaned in mapping_dict.items()
                )

                dataset_con.execute(
                    f"""
                    CREATE OR REPLACE TABLE "{cleaned_table_name}" AS
                    SELECT * RENAME ({rename_clause})
                    FROM read_csv_auto(
                        '{file_path}',
                        HEADER=TRUE,
                        IGNORE_ERRORS=TRUE,
                        STRICT_MODE=FALSE,
                        NULL_PADDING=TRUE,
                        SAMPLE_SIZE=100_000,
                        PARALLEL=FALSE
                        {dialect_options}
                    );
                    """
                )
            else:
                # If we couldn't get header with pandas, let DuckDB create the table
                dataset_con.execute(
                    f"""
                    CREATE OR REPLACE TABLE "{cleaned_table_name}" AS
                    SELECT * FROM read_csv_auto(
                        '{file_path}',
                        HEADER=TRUE,
                        IGNORE_ERRORS=TRUE,
                        STRICT_MODE=FALSE,
                        NULL_PADDING=TRUE,
                        SAMPLE_SIZE=100_000,
                        PARALLEL=FALSE
                        {dialect_options}
                    );
                    """
                )
    finally:
        dataset_con.close()
