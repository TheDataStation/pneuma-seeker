import os
import sys
import unittest
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
)

import pandas as pd

from pneuma_seeker.services.core.actions.executors.python_executor import PythonExecutor
from pneuma_seeker.shared.config import Config


class PythonExecutorTests(unittest.TestCase):
    """Unit tests for PythonExecutor."""

    def setUp(self) -> None:
        self.config = Config()
        self.logger = MagicMock()
        self.db_api = MagicMock()
        self.lm_api = MagicMock()
        self.python_executor = PythonExecutor(
            "user_id", "chat_id", self.config, self.logger, self.db_api, self.lm_api
        )

    def test_execute_code_happy_path_and_table_ids(self):
        tables = {"tbl_1": pd.DataFrame({"a": [1, 2], "b": [3, 4]})}
        code = "result = pd.DataFrame({'sum': [tables['tbl_1']['a'].sum()]})"

        out = self.python_executor.execute({"tables": tables, "code": code})

        self.assertTrue(isinstance(out, pd.DataFrame))
        self.assertEqual(out.iloc[0, 0], 3)

        used_table_ids = self.python_executor.extract_table_ids(code)
        self.assertEqual(used_table_ids, ["tbl_1"])

    def test_execute_code_exception_returns_exception_and_no_table_ids(self):
        tables = {}
        code = 'raise ValueError("boom")'

        with pytest.raises(ValueError, match="boom"):
            self.python_executor.execute({"tables": tables, "code": code})
        used_table_ids = self.python_executor.extract_table_ids(code)
        self.assertEqual(used_table_ids, [])

    def test_execute_code_no_result_variable_raises(self):
        tables = {}
        code = "x = 42  # No result variable defined"

        with self.assertRaises(ValueError) as context:
            self.python_executor.execute({"tables": tables, "code": code})

        self.assertIn(
            "Executed code did not set a 'result' variable.", str(context.exception)
        )

    def test_execute_code_result_not_dataframe_raises(self):
        tables = {}
        code = "result = 42  # result is not a DataFrame"

        with self.assertRaises(ValueError) as context:
            self.python_executor.execute({"tables": tables, "code": code})

        self.assertIn(
            "The 'result' variable must be a pandas DataFrame.", str(context.exception)
        )
