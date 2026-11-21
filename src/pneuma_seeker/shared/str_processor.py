# src/pneuma_seeker/shared/str_processor.py
import ast
import re

from pandas import NA, DataFrame, isna


# Bounds on what a model is shown of a result; stored results stay whole.
PREVIEW_MAX_CELL_CHARS = 300
PREVIEW_MAX_CHARS = 16_000


def _preview_cell(value, max_cell_chars: int) -> str:
    if value is None or (isinstance(value, float) and isna(value)) or value is NA:
        return "NULL"
    text = f"{value:.6g}" if isinstance(value, float) else " ".join(str(value).split())
    if len(text) <= max_cell_chars:
        return text
    return f"{text[:max_cell_chars]}… (+{len(text) - max_cell_chars:,} chars)"


def dataframe_to_preview_str(
    df: DataFrame, max_cell_chars: int = PREVIEW_MAX_CELL_CHARS, max_chars: int = PREVIEW_MAX_CHARS
) -> str:
    """
    Renders a DataFrame for a model to read: a header line, then one line per row,
    cells separated by " | " (not padded to the widest cell). Each cell is cut to
    `max_cell_chars` and the whole to `max_chars`, at a row boundary; each cut says
    what it left out, so a model can query a narrower result instead.

    Callers are expected to have already bounded the row count (e.g. via a SQL LIMIT).
    """
    lines = [" | ".join(str(c) for c in df.columns)]
    for row in df.itertuples(index=False, name=None):
        lines.append(" | ".join(_preview_cell(v, max_cell_chars) for v in row))
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    shown, size = [lines[0]], len(lines[0])
    for line in lines[1:]:
        if size + 1 + len(line) > max_chars:
            if len(shown) == 1:  # always show one row, cut if it alone is too long
                shown.append(line[: max(0, max_chars - size - 1)])
            break
        shown.append(line)
        size += 1 + len(line)
    return "\n".join(shown) + (
        f"\n… (showing {len(shown) - 1} of {len(df)} rows; the result is {len(text):,} characters. "
        "Aggregate, filter or select fewer columns to see the rest.)"
    )


def clean_column_table_name(name):
    """Cleans and normalizes column/table names."""
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


def extract_sql_strings(code: str) -> list[str]:
    tree = ast.parse(code)
    sqls = []

    class SQLVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute_query"
            ):
                if len(node.args) >= 3:
                    arg = node.args[2]

                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        sqls.append(arg.value)

                    elif isinstance(arg, ast.Str):
                        sqls.append(arg.s)

            self.generic_visit(node)

    SQLVisitor().visit(tree)
    return sqls


TABLE_REF_REGEX = re.compile(
    r"""
    (?:
        from|join
    )
    \s+
    (?:
        "(?P<schema_q>[a-zA-Z0-9_]+)"\."(?P<table_q>[a-zA-Z0-9_]+)" |
        "(?P<schema_q2>[a-zA-Z0-9_]+)"\.(?P<table_u2>[a-zA-Z0-9_]+) |
        (?P<schema_u3>[a-zA-Z0-9_]+)\."(?P<table_q3>[a-zA-Z0-9_]+)" |
        (?P<table_only>[a-zA-Z0-9_]+) |
        (?P<schema_u4>[a-zA-Z0-9_]+)\.(?P<table_u4>[a-zA-Z0-9_]+)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_tables_from_sql_regex(sql: str) -> set[str]:
    tables = set()

    for m in TABLE_REF_REGEX.finditer(sql):
        if m.group("schema_q") and m.group("table_q"):
            tables.add(f'{m.group("schema_q")}.{m.group("table_q")}')
        elif m.group("schema_q2") and m.group("table_u2"):
            tables.add(f'{m.group("schema_q2")}.{m.group("table_u2")}')
        elif m.group("schema_u3") and m.group("table_q3"):
            tables.add(f'{m.group("schema_u3")}.{m.group("table_q3")}')
        elif m.group("schema_u4") and m.group("table_u4"):
            tables.add(f'{m.group("schema_u4")}.{m.group("table_u4")}')
        elif m.group("table_only"):
            tables.add(m.group("table_only"))

    return tables


def extract_table_ids_from_code(code: str) -> set[str]:
    sqls = extract_sql_strings(code)
    tables: set[str] = set()

    for sql in sqls:
        tables |= extract_tables_from_sql_regex(sql)

    return tables
