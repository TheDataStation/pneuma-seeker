from typing import Any
from pandas import DataFrame

from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.services.core.action_set.interfaces import Applicable
from pneuma_seeker.shared.parser import augmented_literal_eval
from pneuma_seeker.shared.schemas.language_model.message import LLMMessage
from pneuma_seeker.shared.schemas.language_model.role import Role


class SemanticColumnGeneration(Action, Applicable):
    def get_name(self) -> str:
        return ActionNames.SEMANTIC_COLUMN_GENERATION.value

    def get_description(self) -> str:
        return f"""**{ActionNames.SEMANTIC_COLUMN_GENERATION.value}**
    - Adds a new column to an *intermediate* table using an LLM.
    - The column is derived from specified `relevant_columns` only — no other columns are used.
    - External and internal tables should first be transformed into intermediate tables if new columns are needed, because retrieved internal tables can be replaced.
    - Args: {{
        "table_id": "<intermediate_table_id>",
        "new_column_name": "<column to add>",
        "relevant_columns": ["<list of source columns for generation>"],
        "instruction": "<instruction describing how to generate the new column values>"
    }}
    - Example: {{
        "table_id": "products_2024",
        "new_column_name": "category",
        "relevant_columns": ["product_name", "description"],
        "instruction": "Classify each product into 'Electronics', 'Furniture', or 'Clothing'."
    }}"""

    def get_input_schema(self) -> dict[str, str]:
        return {
            "table": "The table to which the new column will be added.",
            "column_name": "Name of the new column to be generated.",
            "description": "Description of the content and purpose of the new column.",
        }

    def get_notes(self) -> str:
        return (
            "This action uses semantic analysis to generate a new column in the specified table. "
            "Ensure that the table exists and that the description accurately reflects the intended content of the new column."
        )

    def apply(self, input: dict[str, Any]) -> DataFrame:
        src_table_id = input.get("src_table_id")
        src_table_columns = input.get("src_table_columns")
        new_column_name = input.get("new_column_name")
        instruction = input.get("instruction")

        if not isinstance(src_table_id, str):
            raise ValueError("Invalid input type. 'src_table_id' must be a string.")
        if not isinstance(new_column_name, str) or not isinstance(instruction, str):
            raise ValueError(
                "Invalid input types. 'new_column_name' and 'instruction' must be strings."
            )
        if not isinstance(src_table_columns, list) or not all(
            isinstance(c, str) for c in src_table_columns
        ):
            raise ValueError("'src_table_columns' must be a list[str]")
        if len(src_table_columns) == 0:
            raise ValueError("'src_table_columns' must contain at least one column")

        batch_size = self.config.SEMANTIC_COL_GEN_VALUE_GENERATION_BATCH_SIZE
        temp_table = f"{src_table_id}__tmp_semantic_col"
        cols_sql = ", ".join(f'"{c}"' for c in src_table_columns)

        # 1. Create empty temp table with new column
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f"""
            CREATE OR REPLACE TABLE "{temp_table}" AS
            SELECT *, NULL::VARCHAR AS "{new_column_name}"
            FROM "{src_table_id}"
            WHERE FALSE;
            """,
        )

        offset = 0
        while True:
            # 2. Load batch
            df = self.db_api.execute_query(
                self.user_id,
                self.chat_id,
                f"""
                SELECT rowid, {cols_sql}
                FROM "{src_table_id}"
                ORDER BY rowid
                LIMIT {batch_size}
                OFFSET {offset};
                """,
            )

            if df.empty:
                break

            formatted_values = self.__format_values(df[src_table_columns])
            unique_values = list(dict.fromkeys(formatted_values))
            cached_values: dict[str, Any] = {}

            # 3. LLM transform
            for i in range(0, len(unique_values), batch_size):
                batch = unique_values[i : i + batch_size]

                encoded_prompt = [
                    LLMMessage(
                        role=Role.SYSTEM.value,
                        content=(
                            "You are given a list of values from a table, and your task is to generate a new column. "
                            "Output the values directly as a Python list of strings/integers/floats WITHOUT any extra formatting or explanation."
                        ),
                    ),
                    LLMMessage(
                        role=Role.USER.value,
                        content=f"User-defined instruction to form the new column named {new_column_name}: {instruction}",
                    ),
                    LLMMessage(
                        role=Role.USER.value,
                        content=f"Values to transform: {batch}",
                    ),
                ]

                raw_output = "".join(
                    self.language_model_api.chat(encoded_prompt)
                ).strip()
                start = raw_output.find("[")
                end = raw_output.rfind("]")

                if start != -1 and end != -1 and start < end:
                    try:
                        transformed_values = augmented_literal_eval(
                            raw_output[start : end + 1]
                        )
                    except Exception:
                        transformed_values = []
                else:
                    transformed_values = []

                for idx, val in enumerate(transformed_values):
                    cached_values[batch[idx]] = val

            df[new_column_name] = [cached_values.get(v) for v in formatted_values]

            # 4. Materialize batch back into DuckDB using VALUES()
            values_sql = ", ".join(
                f"({row.rowid}, {repr(row[new_column_name])})"
                for _, row in df.iterrows()
            )

            self.db_api.execute_query(
                self.user_id,
                self.chat_id,
                f"""
                INSERT INTO "{temp_table}"
                SELECT src.*, v.val AS "{new_column_name}"
                FROM "{src_table_id}" AS src
                JOIN (VALUES {values_sql}) AS v(rowid, val)
                USING (rowid)
                ORDER BY rowid;
                """,
            )

            offset += batch_size

        # 5. Swap tables
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'ALTER TABLE "{src_table_id}" RENAME TO "{src_table_id}__old";',
        )
        self.db_api.execute_query(
            self.user_id,
            self.chat_id,
            f'ALTER TABLE "{temp_table}" RENAME TO "{src_table_id}";',
        )
        self.db_api.execute_query(
            self.user_id, self.chat_id, f'DROP TABLE "{src_table_id}__old";'
        )

        return self.db_api.execute_query(
            self.user_id, self.chat_id, f'SELECT * FROM "{src_table_id}" LIMIT 5;'
        )

    def __format_values(self, table: DataFrame):
        formatted_values: list[str] = []
        for _, row in table.iterrows():
            row_values: list[str] = []
            for col_name in table.columns:
                row_values.append(f"{col_name}: {row[col_name]}")
            formatted_values.append("; ".join(row_values))
        return formatted_values
