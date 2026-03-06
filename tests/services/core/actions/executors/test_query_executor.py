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
from pneuma_seeker.services.core.action_set.impl.query_executor import QueryExecutor
from pneuma_seeker.shared.config import Config


class QueryExecutorTests(unittest.TestCase):
    """Unit tests for QueryExecutor."""

    def setUp(self) -> None:
        self.user_id = "user_query"
        self.chat_id = "chat_query"
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
        self.query_executor = QueryExecutor(
            self.user_id,
            self.chat_id,
            self.config,
            self.logger,
            self.db_api,
            self.lm_api,
        )

    def test_execute_query_creates_table_and_returns_df(self):
        result_table_id = "result_tbl"
        query = "SELECT 1 + 2 AS total"

        out = self.query_executor.execute(
            {"query": query, "result_table_id": result_table_id}
        )

        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(list(out.columns), ["total"])
        self.assertEqual(out.iloc[0, 0], 3)

        # Table should be persisted in the workspace DB.
        persisted = self.db_api.execute_query(
            self.user_id, self.chat_id, f"SELECT * FROM {result_table_id} LIMIT 1;"
        )
        self.assertEqual(persisted.iloc[0, 0], 3)

    def test_execute_query_allows_trailing_semicolon(self):
        result_table_id = "result_tbl2"
        query = "SELECT 5 AS x;"

        out = self.query_executor.execute(
            {"query": query, "result_table_id": result_table_id}
        )

        self.assertEqual(list(out.columns), ["x"])
        self.assertEqual(out.iloc[0, 0], 5)

    def test_execute_query_raises_on_sql_error(self):
        with pytest.raises(Exception):
            self.query_executor.execute(
                {
                    "query": "SELECT does_not_exist FROM definitely_missing_table",
                    "result_table_id": "result_tbl",
                }
            )

    def test_execute_query_requires_query_string(self):
        with self.assertRaises(ValueError) as context:
            self.query_executor.execute({"query": 123, "result_table_id": "result"})

        self.assertIn("Input 'query' must be a string.", str(context.exception))

    def test_execute_query_requires_result_table_id_string(self):
        with self.assertRaises(ValueError) as context:
            self.query_executor.execute({"query": "SELECT 1", "result_table_id": 123})

        self.assertIn(
            "Input 'result_table_id' must be a string.", str(context.exception)
        )

    def test_execute_query_rejects_multiple_statements(self):
        with self.assertRaises(ValueError) as context:
            self.query_executor.execute(
                {
                    "query": "SELECT 1 AS a; SELECT 2 AS b;",
                    "result_table_id": "result_tbl",
                }
            )

        self.assertIn("single sql statement", str(context.exception).lower())
