from typing import Any
from pandas import DataFrame

from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.actions.interfaces.abstract_action import Action
from pneuma_seeker.services.core.actions.interfaces.applicable import Applicable


class TableProjection(Action, Applicable):
    def get_name(self) -> str:
        return ActionNames.TABLE_PROJECTION.value

    def get_description(self) -> str:
        return "Projects a table to a subset of its columns. Expects a 'table' (DataFrame) and 'relevant_columns' (list of column names) as input, and returns a new DataFrame containing only the specified columns with sample rows."

    def get_input_schema(self) -> dict[str, str]:
        return {}

    def get_notes(self) -> str:
        return ""

    def apply(self, input: dict[str, Any]) -> DataFrame:
        src_table_id = input.get("src_table_id")
        target_table_id = input.get("target_table_id")
        src_table_columns = input.get("src_table_columns")

        if not isinstance(src_table_id, str):
            raise ValueError("Input 'src_table_id' must be a string.")
        if not isinstance(target_table_id, str):
            raise ValueError("Input 'target_table_id' must be a string.")
        if not isinstance(src_table_columns, list) or not all(
            isinstance(col, str) for col in src_table_columns
        ):
            raise ValueError("Input 'src_table_columns' must be a list of strings.")
        if len(src_table_columns) == 0:
            raise ValueError("src_table_columns must contain at least one column.")

        self.db_api.link_dataset_tables(
            self.user_id, self.chat_id, self.config.DATA_SOURCES[0]
        )
        cols_sql = ", ".join(f'"{c}"' for c in src_table_columns)
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'CREATE OR REPLACE TABLE "{target_table_id}" AS SELECT {cols_sql} FROM {src_table_id};',
        )
        return self.db_api.execute_query(
            self.user_id, self.chat_id, f'SELECT * FROM "{target_table_id}" LIMIT 5;'
        )
