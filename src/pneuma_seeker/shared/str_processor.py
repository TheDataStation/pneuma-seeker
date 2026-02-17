# src/pneuma_seeker/shared/str_processor.py
import ast
import re

from sql_metadata import Parser


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
                if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
                    if isinstance(node.args[2].value, str):
                        sqls.append(node.args[2].value)
            self.generic_visit(node)

    SQLVisitor().visit(tree)
    return sqls


def extract_tables_from_sql(sql: str) -> set[str]:
    parser = Parser(sql)
    return set(parser.tables)


def extract_table_ids_from_code(code: str) -> set[str]:
    try:
        sqls = extract_sql_strings(code)
        tables: set[str] = set()

        for sql in sqls:
            tables |= extract_tables_from_sql(sql)

        return tables
    except Exception as e:
        return set()
