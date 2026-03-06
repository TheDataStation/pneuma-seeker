import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../src"))
)

import pandas as pd

from pneuma_seeker.services.core.api.db import DBAPI
from pneuma_seeker.services.core.action_set.impl.python_executor import PythonExecutor
from pneuma_seeker.shared.config import Config


class PythonExecutorTests(unittest.TestCase):
    """Unit tests for PythonExecutor."""

    def setUp(self) -> None:
        self.user_id = "user_rollback"
        self.chat_id = "chat_rollback"
        self.config = Config()
        self.logger = MagicMock()
        self.tmpdir = tempfile.mkdtemp()
        self.db_api = DBAPI(
            self.config,
            self.logger,
            str(Path(self.tmpdir) / "datasets"),
            str(Path(self.tmpdir) / "workspaces"),
        )

        os.makedirs(self.db_api.pneuma_db.dataset_db_path, exist_ok=True)
        os.makedirs(self.db_api.pneuma_db.workspace_db_path, exist_ok=True)

        self.lm_api = MagicMock()
        self.python_executor = PythonExecutor(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.lm_api,
        )

    def test_execute_code_creates_table_and_returns_df(self):
        result_table_id = "result_tbl"
        code = (
            "value = int(np.sum([1, 2]))\n"
            f"db_api.execute_query('{self.user_id}', '{self.chat_id}', "
            f'f"CREATE TABLE {result_table_id} AS SELECT {{value}} AS total")'
        )

        out = self.python_executor.execute(
            {"code": code, "result_table_id": result_table_id}
        )

        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(list(out.columns), ["total"])
        self.assertEqual(out.iloc[0, 0], 3)

    def test_execute_code_raises_on_exec_error(self):
        code = 'raise ValueError("boom")'

        with pytest.raises(ValueError, match="boom"):
            self.python_executor.execute(
                {"code": code, "result_table_id": "result_tbl"}
            )

    def test_execute_code_requires_code_string(self):
        with self.assertRaises(ValueError) as context:
            self.python_executor.execute({"code": 123, "result_table_id": "result"})

        self.assertIn("Input 'code' must be a string.", str(context.exception))

    def test_execute_code_requires_result_table_id_string(self):
        with self.assertRaises(ValueError) as context:
            self.python_executor.execute({"code": "pass", "result_table_id": 123})

        self.assertIn(
            "Input 'result_table_id' must be a string.", str(context.exception)
        )

    def test_execute_code_raises_when_result_table_missing(self):
        with self.assertRaises(Exception) as context:
            self.python_executor.execute(
                {"code": "pass", "result_table_id": "missing_tbl"}
            )

        self.assertTrue(
            "missing_tbl" in str(context.exception).lower()
            or "not found" in str(context.exception).lower()
        )
