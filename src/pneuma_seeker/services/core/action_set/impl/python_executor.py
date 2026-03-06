from typing import Any

import numpy as np
import pandas as pd
import scipy

from pneuma_seeker.services.core.action_set.interfaces import Action
from pneuma_seeker.services.core.action_set.interfaces import Executable
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

        # Basic sanitization/fixes for model-produced code to reduce SyntaxError
        def _strip_code_fence(s: str) -> str:
            s = s.strip()
            # extract content from triple-backtick blocks
            import re

            m = re.search(r"```(?:python)?\n(.+?)```", s, flags=re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
            # sometimes models return JSON like {"code": "..."}
            try:
                import json

                parsed = json.loads(s)
                if isinstance(parsed, dict) and "code" in parsed and isinstance(parsed["code"], str):
                    return parsed["code"].strip()
            except Exception:
                pass
            return s

        def _fix_common_unicode_quotes(s: str) -> str:
            return s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

        code = _strip_code_fence(code)
        code = _fix_common_unicode_quotes(code)

        # Try compiling first to give a clearer error and attempt minimal fixes
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            # As a fallback, try removing leading/trailing lines that often include
            # assistant messages like 'S:' or 'RESULT:'
            lines = code.splitlines()
            # drop leading non-indented short prefixes
            while lines and (lines[0].strip().endswith(":" ) or len(lines[0].strip()) <= 3 and not lines[0].lstrip().startswith(("def ", "import ", "from ", "result", "pd", "np"))):
                lines.pop(0)
            new_code = "\n".join(lines)
            new_code = _fix_common_unicode_quotes(new_code)
            try:
                compile(new_code, "<string>", "exec")
                code = new_code
            except SyntaxError:
                # surface a richer error message including the sanitized code
                raise SyntaxError(f"Code compilation failed after sanitization: {e}; sanitized code:\n{new_code}")

        env = {
            "pd": pd,
            "np": np,
            "scipy": scipy,
            "db_api": self.db_api,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
        }
        exec(code, env)

        return self.db_api.execute_query(
            self.user_id, self.chat_id, f"SELECT * FROM {result_table_id} LIMIT 10;"
        )
