from typing import Any

import numpy as np
import pandas as pd
import scipy

from pneuma_seeker.services.core.actions.interfaces.abstract_action import Action
from pneuma_seeker.services.core.actions.interfaces.executable import Executable
from pneuma_seeker.shared.schemas.core.action import ActionNames


class PythonExecutor(Action, Executable):
    """
    Executes Python code snippets within a controlled environment,
    tracking used tables and integrating with the provenance graph.
    """

    def get_name(self) -> str:
        """Returns the name of the tool."""
        return ActionNames.PYTHON_EXECUTOR.value

    def get_description(self) -> str:
        """Returns the description of the tool."""
        return """Executes Python code snippets with access to pandas, numpy, duckdb, and scipy, returning results (DataFrame) and tracking used tables."""

    def get_input_schema(self) -> dict[str, str]:
        """Returns the input schema of the tool."""
        return {
            "tables": "A dictionary mapping table IDs to pandas DataFrames.",
            "code": "A string containing the Python code to execute. The code should use the 'tables' dictionary to access DataFrames and must set a variable 'result' as the output DataFrame.",
        }

    def get_notes(self) -> str:
        """Returns additional notes about the tool."""
        return "The executed code must define a variable 'result' containing the output DataFrame."

    def execute(
        self,
        input: dict[str, Any],
    ) -> pd.DataFrame:
        """Executes the tool with the given input and returns the output."""
        code = input.get("code")
        result_table_id = input.get("result_table_id")
        if not isinstance(code, str):
            raise ValueError("Input 'code' must be a string.")
        if not isinstance(result_table_id, str):
            raise ValueError("Input 'result_table_id' must be a string.")

        env = {
            "pd": pd,
            "np": np,
            "scipy": scipy,
            "db_api": self.db_api,
        }
        exec(code, env)

        return self.db_api.execute_query(
            self.user_id, self.chat_id, f"SELECT * FROM {result_table_id} LIMIT 5;"
        )
