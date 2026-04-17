import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../src")
    ),
)

from pneuma_seeker.services.indexing.connectors.postgresql_connector import (
    PostgreSQLConnector,
)


class TestPostgreSQLConnector(unittest.TestCase):
    def setUp(self):
        self.config = {
            "type": "postgres",
            "host": "localhost",
            "port": 5432,
            "user": "user",
            "password": "password",
            "dbname": "mydb",
            "schema": "public",
            "row_limit": 100,
        }

    @patch("pneuma_seeker.services.indexing.connectors.postgresql_connector.duckdb.connect")
    def test_check_connection_success(self, mock_connect):
        mock_con = MagicMock()
        mock_con.execute.return_value = mock_con
        mock_connect.return_value = mock_con

        connector = PostgreSQLConnector(self.config)
        self.assertTrue(connector.check_connection())

        executed_sql = [call.args[0] for call in mock_con.execute.call_args_list]
        self.assertTrue(any("ATTACH" in sql for sql in executed_sql))
        self.assertTrue(any("DETACH source_db" in sql for sql in executed_sql))

    @patch("pneuma_seeker.services.indexing.connectors.postgresql_connector.duckdb.connect")
    def test_discover_returns_tables(self, mock_connect):
        mock_con = MagicMock()
        mock_con.execute.return_value = mock_con
        mock_con.fetchall.return_value = [
            ("public", "users"),
            ("public", "orders"),
        ]
        mock_connect.return_value = mock_con

        connector = PostgreSQLConnector(self.config)
        streams = connector.discover()

        self.assertEqual(len(streams), 2)
        self.assertEqual(streams[0]["stream"], "public.users")
        self.assertEqual(streams[0]["table_name"], "public_users")

    @patch("pneuma_seeker.services.indexing.connectors.postgresql_connector.duckdb.connect")
    def test_read_returns_records_and_applies_limit(self, mock_connect):
        mock_con = MagicMock()
        mock_con.execute.return_value = mock_con
        mock_con.fetchdf.return_value = pd.DataFrame(
            [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        )
        mock_connect.return_value = mock_con

        connector = PostgreSQLConnector(self.config)
        rows = list(connector.read("public.users"))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], 1)

        executed_sql = [call.args[0] for call in mock_con.execute.call_args_list]
        select_sql = [sql for sql in executed_sql if sql.strip().startswith("SELECT")][0]
        self.assertIn("LIMIT 100", select_sql)

    def test_init_missing_required_config_raises(self):
        with self.assertRaises(ValueError):
            PostgreSQLConnector({"type": "postgres", "host": "localhost"})


if __name__ == "__main__":
    unittest.main()