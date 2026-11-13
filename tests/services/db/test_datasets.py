# tests/services/db/test_datasets.py
import csv
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from pneuma_seeker.models import PermissionKey
from pneuma_seeker.services.db.datasets.connectors import DatasetUnavailable
from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.shared.config import Config


def offline_config() -> Config:
    """Returns a config whose Postgres is unreachable, so tests never touch a real database."""
    config = Config()
    config.POSTGRES_HOST, config.POSTGRES_PORT = "127.0.0.1", 1
    return config


class DatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(
            offline_config(),
            logging.getLogger("test"),
            dataset_db_path=str(Path(self.tmpdir) / "datasets"),
            workspace_db_path=str(Path(self.tmpdir) / "workspaces"),
        )
        self.dataset_dir = Path(self.tmpdir) / "test_data"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_test_csv(self, filename: str, data: list[dict], folder: Path | None = None):
        path = (folder or self.dataset_dir) / filename
        pd.DataFrame(data).to_csv(path, index=False)
        return path

    def _tables(self, dataset_name: str) -> list[str]:
        con = self.db.get_dataset_connection(dataset_name)
        try:
            return [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        finally:
            con.close()


class TestDatasetIngestion(DatasetTestCase):
    def test_ingest_stores_each_csv_as_a_parquet_table(self):
        self._create_test_csv("users.csv", [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        self.db.ingest_dataset("test_dataset", self.dataset_dir.as_posix())
        self.assertTrue((self.db.dataset_db_path / "test_dataset" / "users" / "users.parquet").is_file())
        self.assertEqual(self._tables("test_dataset"), ["users"])

    def test_ingest_multiple_csvs(self):
        self._create_test_csv("users.csv", [{"id": 1, "name": "Alice"}])
        self._create_test_csv("products.csv", [{"product_id": 1, "title": "Widget"}])
        self.db.ingest_dataset("multi_dataset", self.dataset_dir.as_posix())
        self.assertCountEqual(self._tables("multi_dataset"), ["users", "products"])

    def test_ingest_cleans_table_and_column_names(self):
        self._create_test_csv("Bad Columns.csv", [{"First Name": 1, "Amount ($)": 2, "Email@Domain": "x"}])
        self.db.ingest_dataset("special_cols", self.dataset_dir.as_posix())
        con = self.db.get_dataset_connection("special_cols")
        try:
            columns = list(con.execute("SELECT * FROM bad_columns LIMIT 1").fetchdf().columns)
        finally:
            con.close()
        self.assertEqual(columns, ["first_name", "amount_usd", "email_domain"])

    def test_ingest_ignores_non_csv_files(self):
        self._create_test_csv("data.csv", [{"id": 1, "value": "test"}])
        (self.dataset_dir / "readme.txt").write_text("Some readme content")
        self.db.ingest_dataset("ignore_test", self.dataset_dir.as_posix())
        self.assertEqual(self._tables("ignore_test"), ["data"])

    def test_ingest_deduplicates_column_names(self):
        with open(self.dataset_dir / "dupes.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "value", "id"])
            writer.writerow([1, "test", 2])
        self.db.ingest_dataset("dedup_test", self.dataset_dir.as_posix())
        con = self.db.get_dataset_connection("dedup_test")
        try:
            self.assertEqual(len(con.execute("SELECT * FROM dupes").fetchdf().columns), 3)
        finally:
            con.close()

    def test_reingest_replaces_the_dataset_unless_told_not_to(self):
        self._create_test_csv("old.csv", [{"a": 1}])
        self.db.ingest_dataset("re_ds", self.dataset_dir.as_posix())
        (self.dataset_dir / "old.csv").unlink()
        self._create_test_csv("new.csv", [{"a": 1}])
        self.db.ingest_dataset("re_ds", self.dataset_dir.as_posix(), overwrite=False)
        self.assertEqual(self._tables("re_ds"), ["old"])
        self.db.ingest_dataset("re_ds", self.dataset_dir.as_posix())
        self.assertEqual(self._tables("re_ds"), ["new"])


class TestDatasetConnections(DatasetTestCase):
    def test_connection_reads_only_its_dataset_and_files(self):
        self._create_test_csv("t.csv", [{"a": 1}])
        self.db.ingest_dataset("conn_ds", self.dataset_dir.as_posix())
        outside = self._create_test_csv("outside.csv", [{"secret": 1}], folder=Path(self.tmpdir))
        con = self.db.get_dataset_connection("conn_ds")
        try:
            self.assertEqual(con.execute("SELECT a FROM t").fetchall(), [(1,)])
            with self.assertRaises(Exception):
                con.execute(f"SELECT * FROM read_csv('{outside}')").fetchall()
            with self.assertRaises(Exception):
                con.execute("SET enable_external_access=true")
        finally:
            con.close()

    def test_unknown_dataset_raises(self):
        with self.assertRaises(DatasetUnavailable):
            self.db.get_dataset_connection("no_such_ds")


class TestTableDescription(DatasetTestCase):
    def test_returns_empty_without_the_memory_layer(self):
        self.assertIsNone(self.db.memory_manager)
        self.assertEqual(self.db.get_table_description("ds", "users"), "")

    def test_reads_the_memory_layer(self):
        memory = MagicMock()
        memory.get_by_key.side_effect = lambda kind, ds, key: (
            MagicMock(content="User table") if (kind, ds, key) == ("table_metadata", "ds", "users") else None
        )
        self.db.memory_manager = memory
        self.assertEqual(self.db.get_table_description("ds", "users"), "User table")
        self.assertEqual(self.db.get_table_description("ds", "orders"), "")

    def test_ingest_stores_metadata_descriptions_in_the_memory_layer(self):
        self._create_test_csv("Users.csv", [{"id": 1}])
        metadata = self._create_test_csv(
            "metadata.csv",
            [{"table_name": "Users", "description": "People"}, {"table_name": "x", "description": None}],
            folder=Path(self.tmpdir),
        )
        memory = MagicMock()
        self.db.memory_manager = memory
        self.db.ingest_dataset("desc_ds", self.dataset_dir.as_posix(), metadata.as_posix())
        memory.replace_dataset_rows.assert_called_once_with(
            "desc_ds", "table_metadata", [{"key_text": "users", "content": "People"}]
        )
        self.assertFalse((self.db.dataset_db_path / "desc_ds" / "metadata.csv").exists())


class TestDatasetLinking(DatasetTestCase):
    def setUp(self):
        super().setUp()
        self._create_test_csv("users.csv", [{"id": 1}])
        self.db.ingest_dataset("link_ds", self.dataset_dir.as_posix())

    def _query(self, sql: str):
        return self.db.execute_query("u1", "c1", sql)

    def test_link_attaches_tables_as_dataset_qualified_views(self):
        self.db.link_dataset_tables("u1", "c1", "link_ds")
        self.assertEqual(self._query('SELECT id FROM link_ds."users"')["id"].tolist(), [1])

    def test_link_is_idempotent(self):
        self.db.link_dataset_tables("u1", "c1", "link_ds")
        self.db.link_dataset_tables("u1", "c1", "link_ds")
        self.assertEqual(len(self._query('SELECT * FROM link_ds."users"')), 1)

    def test_link_reattaches_after_the_data_changes(self):
        self.db.link_dataset_tables("u1", "c1", "link_ds")
        extra = Path(self.tmpdir) / "extra"
        extra.mkdir()
        self._create_test_csv("orders.csv", [{"order_id": 7}], folder=extra)
        self.db.add_file_to_dataset("link_ds", extra / "orders.csv", "orders.csv", create=False, replace=False)
        self.db.link_dataset_tables("u1", "c1", "link_ds")
        self.assertEqual(self._query('SELECT order_id FROM link_ds."orders"')["order_id"].tolist(), [7])

    def test_link_and_queries_share_one_lock_even_before_the_connection_exists(self):
        workspace = self.db.workspace_manager
        lock = workspace.connection_lock("u2", "c2")  # taken before the connection is created
        workspace.get_ws_db_connection("u2", "c2")
        self.assertIs(workspace.connection_lock("u2", "c2"), lock)

    def test_parallel_links_and_queries_on_one_connection(self):
        import threading

        extra = Path(self.tmpdir) / "extra"
        extra.mkdir()
        for i in range(20):
            self._create_test_csv(f"t{i}.csv", [{"a": i}], folder=extra)
        self.db.ingest_dataset("other_ds", extra.as_posix())
        errors = []

        def work(dataset: str) -> None:
            try:
                for _ in range(5):
                    self.db.link_dataset_tables("u1", "c1", dataset)
                    self.db.execute_query("u1", "c1", f'SELECT * FROM {dataset}."t0" LIMIT 1' if dataset == "other_ds" else 'SELECT * FROM link_ds."users"')
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(d,)) for d in ("link_ds", "other_ds") * 3]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])

    def test_link_unknown_dataset_raises(self):
        with self.assertRaises(DatasetUnavailable):
            self.db.link_dataset_tables("u1", "c1", "no_such_ds")


class TestGetAccessibleDatasets(DatasetTestCase):
    def setUp(self):
        super().setUp()
        self._create_test_csv("t.csv", [{"a": 1}])
        for name in ("ds_alpha", "ds_beta"):
            self.db.ingest_dataset(name, self.dataset_dir.as_posix())
        (self.db.dataset_db_path / "empty_dir").mkdir()

    def test_admin_sees_every_dataset(self):
        self.assertEqual(self.db.get_accessible_datasets(True, {}), ["ds_alpha", "ds_beta"])

    def test_non_admin_sees_granted_datasets(self):
        perms = {f"{PermissionKey.DATASET_ACCESS_PREFIX.value}:ds_alpha": "read", "other": "write"}
        self.assertEqual(self.db.get_accessible_datasets(False, perms), ["ds_alpha"])
        self.assertTrue(self.db.is_dataset_accessible("ds_alpha", False, perms))
        self.assertFalse(self.db.is_dataset_accessible("ds_beta", False, perms))

    def test_permission_for_a_missing_dataset_grants_nothing(self):
        perms = {f"{PermissionKey.DATASET_ACCESS_PREFIX.value}:ds_ghost": "read"}
        self.assertEqual(self.db.get_accessible_datasets(False, perms), [])
        self.assertFalse(self.db.is_dataset_accessible("ds_ghost", True, perms))


class TestListAndQueryDatasetTable(DatasetTestCase):
    """Tests for browsing a dataset's tables directly, without a chat session."""

    def setUp(self):
        super().setUp()
        pd.DataFrame(
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
                {"id": 3, "name": "Charlie"},
            ]
        ).to_csv(self.dataset_dir / "users.csv", index=False)
        pd.DataFrame([{"product_id": 1, "title": "Widget"}]).to_csv(
            self.dataset_dir / "products.csv", index=False
        )
        self.db.ingest_dataset("browse_ds", self.dataset_dir.as_posix())

    def test_list_dataset_tables_returns_all_table_names(self):
        tables = self.db.list_dataset_tables("browse_ds")
        self.assertCountEqual(tables, ["users", "products"])

    def test_list_dataset_tables_missing_dataset_returns_empty(self):
        self.assertEqual(self.db.list_dataset_tables("no_such_ds"), [])

    def test_query_dataset_table_returns_all_columns_and_rows(self):
        rows_df, total_count, columns = self.db.query_dataset_table(
            "browse_ds",
            "users",
            limit=50,
            offset=0,
            order_by=None,
            order_dir="asc",
            search=None,
        )
        self.assertEqual(total_count, 3)
        self.assertEqual(columns, ["id", "name"])
        self.assertEqual(len(rows_df), 3)

    def test_query_dataset_table_pagination(self):
        rows_df, total_count, _ = self.db.query_dataset_table(
            "browse_ds",
            "users",
            limit=2,
            offset=2,
            order_by="id",
            order_dir="asc",
            search=None,
        )
        self.assertEqual(total_count, 3)
        self.assertEqual(len(rows_df), 1)
        self.assertEqual(rows_df.iloc[0]["id"], 3)

    def test_query_dataset_table_search_filters_rows(self):
        rows_df, total_count, _ = self.db.query_dataset_table(
            "browse_ds",
            "users",
            limit=50,
            offset=0,
            order_by=None,
            order_dir="asc",
            search="ali",
        )
        self.assertEqual(total_count, 1)
        self.assertEqual(rows_df.iloc[0]["name"], "Alice")

    def test_query_dataset_table_order_by_desc(self):
        rows_df, _, _ = self.db.query_dataset_table(
            "browse_ds",
            "users",
            limit=50,
            offset=0,
            order_by="id",
            order_dir="desc",
            search=None,
        )
        self.assertEqual(rows_df["id"].tolist(), [3, 2, 1])

    def test_query_dataset_table_nonexistent_table_raises(self):
        with self.assertRaises(Exception):
            self.db.query_dataset_table(
                "browse_ds",
                "no_such_table",
                limit=50,
                offset=0,
                order_by=None,
                order_dir="asc",
                search=None,
            )


class TestQueryDatasetSQL(DatasetTestCase):
    """Tests for the general read-only SQL query path (query_dataset_sql)."""

    def setUp(self):
        super().setUp()
        pd.DataFrame(
            [
                {"id": 1, "name": "Alice", "amount": 10},
                {"id": 2, "name": "Bob", "amount": 20},
                {"id": 3, "name": "Charlie", "amount": 30},
            ]
        ).to_csv(self.dataset_dir / "orders.csv", index=False)
        self.db.ingest_dataset("sql_ds", self.dataset_dir.as_posix())

    def test_query_dataset_sql_runs_select_with_aggregation(self):
        rows_df, total_count, columns = self.db.query_dataset_sql(
            "sql_ds", "SELECT SUM(amount) AS total FROM orders"
        )
        self.assertEqual(total_count, 1)
        self.assertEqual(columns, ["total"])
        self.assertEqual(rows_df.iloc[0]["total"], 60)

    def test_query_dataset_sql_supports_information_schema_for_column_types(self):
        rows_df, _, _ = self.db.query_dataset_sql(
            "sql_ds",
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'orders' ORDER BY ordinal_position",
        )
        self.assertEqual(rows_df["column_name"].tolist(), ["id", "name", "amount"])

    def test_query_dataset_sql_with_params(self):
        rows_df, total_count, _ = self.db.query_dataset_sql(
            "sql_ds", "SELECT * FROM orders WHERE amount > ?", params=(15,)
        )
        self.assertEqual(total_count, 2)
        self.assertCountEqual(rows_df["name"].tolist(), ["Bob", "Charlie"])

    def test_query_dataset_sql_respects_limit(self):
        rows_df, total_count, _ = self.db.query_dataset_sql("sql_ds", "SELECT * FROM orders", limit=1)
        self.assertEqual(total_count, 3)  # total_count reflects the full result, not the capped page
        self.assertEqual(len(rows_df), 1)

    def test_query_dataset_sql_rejects_non_select(self):
        with self.assertRaises(ValueError):
            self.db.query_dataset_sql("sql_ds", "DROP TABLE orders")

    def test_query_dataset_sql_rejects_multiple_statements(self):
        with self.assertRaises(ValueError):
            self.db.query_dataset_sql("sql_ds", "SELECT * FROM orders; DROP TABLE orders")

    def test_query_dataset_sql_rejects_empty_string(self):
        with self.assertRaises(ValueError):
            self.db.query_dataset_sql("sql_ds", "   ")

    def test_query_dataset_sql_cannot_write_files(self):
        """The connection can't touch files outside the dataset, so a SELECT-shaped
        statement smuggling a write is rejected by DuckDB, not just the text check."""
        target = Path(self.tmpdir) / "x.csv"
        with self.assertRaises(Exception):
            self.db.query_dataset_sql(
                "sql_ds", f"WITH x AS (SELECT * FROM orders) SELECT * FROM x, (COPY orders TO '{target}') AS y"
            )
        self.assertFalse(target.exists())

    def test_query_dataset_sql_cannot_read_other_files(self):
        outside = self._create_test_csv("outside.csv", [{"secret": 1}], folder=Path(self.tmpdir))
        with self.assertRaises(Exception):
            self.db.query_dataset_sql("sql_ds", f"SELECT * FROM read_csv('{outside}')")

    def test_query_dataset_sql_cannot_reach_other_datasets(self):
        other_dir = Path(self.tmpdir) / "other_data"
        other_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"secret": 1}]).to_csv(other_dir / "secrets.csv", index=False)
        self.db.ingest_dataset("other_ds", other_dir.as_posix())

        with self.assertRaises(Exception):
            self.db.query_dataset_sql("sql_ds", "SELECT * FROM other_ds.secrets")


if __name__ == "__main__":
    unittest.main()
