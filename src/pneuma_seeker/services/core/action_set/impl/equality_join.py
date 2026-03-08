import re
from typing import Any

from pandas import DataFrame

from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.services.core.action_set.interfaces import Applicable


class EqualityJoin(Action, Applicable):
    def get_name(self) -> str:
        return ActionNames.EQUALITY_JOIN.value

    def get_description(self) -> str:
        return f"""**{ActionNames.EQUALITY_JOIN.value}**
    - Joins two tables (internal, external, or intermediate) by exact equality on specified key columns.
    - The output table is materialized into the workspace database.
    - Args:
    {{
        "left_table_id": "<left table reference>",
        "right_table_id": "<right table reference>",
        "left_table_column_keys": ["<left join key col 1>", "<left join key col 2>", ...],
        "right_table_column_keys": ["<right join key col 1>", "<right join key col 2>", ...],
        "result_table_id": "<intermediate/target table name for the join result>"
    }}
    - Notes:
        - If a table is an internal table that was retrieved (i.e., the table has an ID like "Table x (dataset: y)"), reference it as a dataset-qualified name like `y."x"`.
        - If a table is an intermediate/external table created during processing, reference it by its workspace name directly (e.g., `my_intermediate_table`).
        - The key lists must be non-empty and the same length; keys are matched positionally."""

    def get_input_schema(self) -> dict[str, str]:
        return {}

    def get_notes(self) -> str:
        return ""

    def apply(self, input: dict[str, Any]) -> DataFrame:
        left_table_id = input.get("left_table_id")
        right_table_id = input.get("right_table_id")
        left_table_column_keys = input.get("left_table_column_keys")
        right_table_column_keys = input.get("right_table_column_keys")
        result_table_id = input.get("result_table_id")

        if not isinstance(left_table_id, str):
            raise ValueError("Input 'left_table_id' must be a string.")
        if not isinstance(right_table_id, str):
            raise ValueError("Input 'right_table_id' must be a string.")
        if not isinstance(result_table_id, str):
            raise ValueError("Input 'result_table_id' must be a string.")

        if not isinstance(left_table_column_keys, list) or not all(
            isinstance(c, str) for c in left_table_column_keys
        ):
            raise ValueError(
                "Input 'left_table_column_keys' must be a list of strings."
            )
        if not isinstance(right_table_column_keys, list) or not all(
            isinstance(c, str) for c in right_table_column_keys
        ):
            raise ValueError(
                "Input 'right_table_column_keys' must be a list of strings."
            )
        if len(left_table_column_keys) == 0 or len(right_table_column_keys) == 0:
            raise ValueError(
                "left_table_column_keys and right_table_column_keys must contain at least one column."
            )
        if len(left_table_column_keys) != len(right_table_column_keys):
            raise ValueError(
                "left_table_column_keys and right_table_column_keys must have the same length."
            )

        left_table_ref = self.__validate_table_ref(left_table_id)
        right_table_ref = self.__validate_table_ref(right_table_id)

        self.__link_datasets_for_table_ref(left_table_ref)
        self.__link_datasets_for_table_ref(right_table_ref)

        left_cols = self.__get_table_columns(left_table_ref)
        right_cols = self.__get_table_columns(right_table_ref)

        if not set(left_table_column_keys) <= set(left_cols):
            raise ValueError(
                "left_table_column_keys is not a subset of left_table columns."
            )
        if not set(right_table_column_keys) <= set(right_cols):
            raise ValueError(
                "right_table_column_keys is not a subset of right_table columns."
            )

        join_predicates = " AND ".join(
            f'l."{lk}" = r."{rk}"'
            for lk, rk in zip(left_table_column_keys, right_table_column_keys)
        )

        left_select = ", ".join(
            f'l."{col}" AS "left_{col}"' for col in left_cols
        )
        right_select = ", ".join(
            f'r."{col}" AS "right_{col}"' for col in right_cols
        )

        # Materialize join into a workspace table.
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f"""
            CREATE OR REPLACE TABLE {self.__quote_ident(result_table_id)} AS
            SELECT {left_select}, {right_select}
            FROM {left_table_ref} AS l
            JOIN {right_table_ref} AS r
            ON {join_predicates};
            """,
        )

        return self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f"SELECT * FROM {self.__quote_ident(result_table_id)} LIMIT 5;",
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
        if not self._TABLE_REF_RE.fullmatch(table_ref.strip()):
            raise ValueError(
                "Invalid table reference format. Use a bare table name or dataset-qualified form like dataset.table or dataset.\"table\"."
            )
        return table_ref.strip()

    def __link_datasets_for_table_ref(self, table_ref: str) -> None:
        """If table_ref is dataset-qualified, ensure the dataset is attached."""

        if "." not in table_ref:
            return

        dataset_part = table_ref.split(".", 1)[0].strip()
        # Remove quotes if present to match config.DATA_SOURCES values.
        if dataset_part.startswith('"') and dataset_part.endswith('"'):
            dataset_part = dataset_part[1:-1].replace('""', '"')

        if self.config.DATA_SOURCES and dataset_part in self.config.DATA_SOURCES:
            self.db_api.link_dataset_tables(self.user_id, self.chat_id, dataset_part)

    def __get_table_columns(self, table_ref: str) -> list[str]:
        df = self.db_api.execute_query(
            self.user_id, self.chat_id, f"SELECT * FROM {table_ref} LIMIT 0;"
        )
        return list(df.columns)
