# src/pneuma_seeker/shared/str_processor.py
import ast
import re


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
