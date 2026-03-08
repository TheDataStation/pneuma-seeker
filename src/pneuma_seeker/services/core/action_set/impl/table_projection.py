import re
from typing import Any
from pandas import DataFrame

from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.services.core.action_set.interfaces import Applicable


class TableProjection(Action, Applicable):
    def get_name(self) -> str:
        return ActionNames.TABLE_PROJECTION.value

    def get_description(self) -> str:
        return f"""**{ActionNames.TABLE_PROJECTION.value}**
    - Projects a table (internal, external, or intermediate) to a subset of its columns, optionally renaming columns at the same time.
    - The output table is materialized into the workspace database.
    - Args:
    {{
        "<target_table_id>": {{
            "id": "<source_table_id>",
            "columns": {{"<output_col_1>": "<source_col_1>", "<output_col_2>": "<source_col_2>"}}
        }}
    }}
    - Notes:
        - If the source table is an internal table that was retrieved (i.e., the table has an ID like "Table x (dataset: y)"), reference it as a dataset-qualified name like `y."x"`.
        - If the source table is an intermediate/external table created in the workspace, reference it by its workspace name directly (e.g., `my_intermediate_table`).
        - The `columns` mapping is **output_column_name -> source_column_name** (use this to rename columns during projection).
        - The output table (`target_table_id`) is always created/overwritten in the workspace.\n"""

    def get_input_schema(self) -> dict[str, str]:
        return {}

    def get_notes(self) -> str:
        return ""

    def apply(self, input: dict[str, Any]) -> DataFrame:
        src_table_id = input.get("src_table_id")
        target_table_id = input.get("target_table_id")
        column_mapping = input.get("column_mapping")

        if not isinstance(src_table_id, str):
            raise ValueError("Input 'src_table_id' must be a string.")
        if not isinstance(target_table_id, str):
            raise ValueError("Input 'target_table_id' must be a string.")

        if not isinstance(column_mapping, dict):
            raise ValueError(
                "Input 'column_mapping' must be a dict[str, str] mapping output column names to source column names."
            )
        if len(column_mapping) == 0:
            raise ValueError("column_mapping must contain at least one mapping.")
        if not all(
            isinstance(alias, str) and isinstance(source, str)
            for alias, source in column_mapping.items()
        ):
            raise ValueError("Input 'column_mapping' must map strings to strings.")

        column_mapping = dict(column_mapping)

        src_table_ref = self.__validate_table_ref(src_table_id)
        self.__link_datasets_for_table_ref(src_table_ref)

        cols_sql = ", ".join(
            f"{self.__quote_ident(source)} AS {self.__quote_ident(alias)}"
            for alias, source in column_mapping.items()
        )
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'CREATE OR REPLACE TABLE {self.__quote_ident(target_table_id)} AS SELECT {cols_sql} FROM {src_table_ref};',
        )
        return self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f"SELECT * FROM {self.__quote_ident(target_table_id)} LIMIT 5;",
        )

    def __quote_ident(self, identifier: str) -> str:
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    _IDENT_RE = r"[A-Za-z_][A-Za-z0-9_]*"
    _QUOTED_IDENT_RE = r'"(?:[^"]|"")*"'
    _TABLE_REF_RE = re.compile(
        rf"^(?:{_IDENT_RE}|{_QUOTED_IDENT_RE})(?:\.(?:{_IDENT_RE}|{_QUOTED_IDENT_RE}))?$"
    )

    def __validate_table_ref(self, table_ref: str) -> str:
        """Validate that a user-provided table reference is safe to embed in SQL.

        Per the materializer prompt contract, callers must provide either:
        - Workspace/intermediate table name (e.g. my_table)
        - Dataset-qualified table ref (e.g. dataset.table or dataset.\"table\")
        """

        if not isinstance(table_ref, str):
            raise ValueError("Table reference must be a string.")
        stripped = table_ref.strip()
        if not self._TABLE_REF_RE.fullmatch(stripped):
            raise ValueError(
                "Invalid table reference format. Use a bare table name or dataset-qualified form like dataset.table or dataset.\"table\"."
            )
        return stripped

    def __link_datasets_for_table_ref(self, table_ref: str) -> None:
        """If table_ref is dataset-qualified, ensure the dataset is attached."""

        if "." not in table_ref:
            return

        dataset_part = table_ref.split(".", 1)[0].strip()
        if dataset_part.startswith('"') and dataset_part.endswith('"'):
            dataset_part = dataset_part[1:-1].replace('""', '"')

        if self.config.DATA_SOURCES and dataset_part in self.config.DATA_SOURCES:
            self.db_api.link_dataset_tables(self.user_id, self.chat_id, dataset_part)
