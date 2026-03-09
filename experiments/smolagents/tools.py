from pathlib import Path
import pandas as pd
from smolagents import tool


@tool
def list_files(path: str) -> list[str]:
    """
    List all CSV files in a given directory.

    Args:
        path (str): Directory path containing dataset tables.

    Returns:
        list[str]: Sorted list of CSV file paths.
    """
    p = Path(path)

    if not p.exists():
        return [f"Error: path '{path}' does not exist."]
    if not p.is_dir():
        return [f"Error: path '{path}' is not a directory."]

    csv_files = sorted(str(f.resolve()) for f in p.glob("*.csv"))

    if not csv_files:
        return ["No CSV files found in directory."]

    return csv_files


@tool
def get_tables_representation(table_paths: list[str]) -> str:
    """
    Generate a structured textual summary of one or more CSV tables.

    Args:
        table_paths (list[str]): List of CSV file paths.

    Returns:
        str: A formatted string containing, for each table:
            - Table name
            - File path
            - Table shape (rows x columns)
            - Columns with data types
            - Up to 5 sampled rows
    """
    tables_representation = []

    for table_path in table_paths:
        try:
            table = pd.read_csv(table_path)
        except Exception as e:
            tables_representation.append(f"Error loading {table_path}: {str(e)}")
            continue

        table_name = Path(table_path).stem
        n_rows, n_cols = table.shape

        cols = " | ".join(
            f"{col} ({dtype})" for col, dtype in zip(table.columns, table.dtypes)
        )

        lines = [
            f"=== Table: {table_name} ===",
            f"Path: {table_path}",
            f"Shape: {n_rows} rows x {n_cols} columns",
            f"Columns: {cols}",
        ]

        if n_rows > 0:
            sample_rows = table.sample(min(5, n_rows), random_state=42).sort_index()

            lines.append("Sample rows:")

            for idx, (_, row) in enumerate(sample_rows.iterrows(), start=1):
                row_str = " | ".join(str(row[col]) for col in table.columns)
                lines.append(f"  Row {idx}: {row_str}")

        tables_representation.append("\n".join(lines))

    return "\n\n".join(tables_representation)
