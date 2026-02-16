import re
from typing import Any

from pneuma_seeker.shared.schemas.core.ir_system import AbstractDocument


def generate_read_external_tables_code(table_number: int, doc: AbstractDocument):
    doc_path = doc.path or "<no_path_provided>"
    read_document_code = f"""pd.read_csv(r"{doc_path}")"""
    if doc_path.endswith(".xlsx") or doc_path.endswith(".xls"):
        read_document_code = f"""pd.read_excel(r"{doc_path}")"""
    return f"""# User-uploaded table #{table_number}\ntables["{doc.doc_id}"] = {read_document_code}"""


def generate_pandas_read_csv_code(doc: AbstractDocument):
    """Generates Python code to read a CSV file into a pandas DataFrame."""
    doc_var_name = re.sub(r"\W|^(?=\d)", "_", doc.doc_id or "var")
    doc_path = doc.path or "<no_path_provided>"
    return f"""tables["{doc_var_name}"] = pd.read_csv(r"{doc_path}")"""


def generate_view_textual_document_code(doc: AbstractDocument):
    """Generates Python code to view a textual document."""
    doc_var_name = re.sub(r"\W|^(?=\d)", "_", doc.doc_id or "var")
    return f"""{doc_var_name} = "{doc.content}" """.strip()


def generate_pandas_read_multi_doc_code(docs: list[AbstractDocument]):
    """Generates Python code to read multiple CSV files into pandas DataFrames."""
    python_code_lines = ["import pandas as pd"]
    for doc in docs:
        doc_var_name = re.sub(r"\W|^(?=\d)", "_", doc.doc_id or "var")
        doc_path = doc.path or "<no_path_provided>"
        python_code_lines.append(
            f"""tables["{doc_var_name}"] = pd.read_csv(r"{doc_path}")"""
        )
    return "\n".join(python_code_lines)


def generate_table_select_code(
    target_var_name: str, source_id: str, relevant_cols: list[str]
):
    """Generates Python code to select specific columns from a table."""
    source_var_name = re.sub(r"\W|^(?=\d)", "_", source_id or "var")
    return f"""{target_var_name} = tables["{source_var_name}"][{repr(relevant_cols)}]"""


def generate_semantic_col_generator_code(
    conditioned_cols: list[str],
    doc: AbstractDocument,
    new_col_name: str,
    new_col_values: list[Any],
):
    """Generates Python code to semantically generate a new column for a table."""
    doc_var_name = re.sub(r"\W|^(?=\d)", "_", doc.doc_id or "var")
    return f"""# Semantically generate new column `{new_col_name}` for the table {doc_var_name}, conditioned on these columns: {conditioned_cols}
tables["{doc_var_name}"]["{new_col_name}"] = {new_col_values}
"""


def generate_semantic_join_generator_code(
    doc_1: AbstractDocument,
    doc_2: AbstractDocument,
    relevant_left_cols: list[str],
    relevant_right_cols: list[str],
    top_k: int,
):
    """Generates Python code to semantically join two tables."""
    left_doc_var_name = re.sub(r"\W|^(?=\d)", "_", doc_1.doc_id or "var")
    right_doc_var_name = re.sub(r"\W|^(?=\d)", "_", doc_2.doc_id or "var")

    return f"""# Semantically join two tables, A ({left_doc_var_name}) and B ({right_doc_var_name}), and keep the top-{top_k} join candidates for each row in the smaller table
# => Relevant columns in A: {relevant_left_cols}
# => Relevant columns in B: {relevant_right_cols}
"""


def append_comment_to_existing_code(code: str, comment: str):
    return f"# {comment}\n{code}"
