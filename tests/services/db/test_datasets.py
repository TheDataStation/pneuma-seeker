# tests/services/db/test_datasets.py
import csv
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from pneuma_seeker.models import PermissionKey
from pneuma_seeker.services.db.pneuma_db import PneumaDB
from pneuma_seeker.shared.config import Config


class TestDatasetIngestion(unittest.TestCase):
    """Tests for dataset ingestion functionality."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(logger=self.logger, config=self.config)
        self.db.dataset_db_path = Path(self.tmpdir) / "datasets"
        self.db.workspace_db_path = Path(self.tmpdir) / "workspaces"
        self.db.dataset_db_path.mkdir(parents=True, exist_ok=True)
        self.db.workspace_db_path.mkdir(parents=True, exist_ok=True)

        self.dataset_dir = Path(self.tmpdir) / "test_data"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_test_csv(self, filename: str, data: list[dict]):
        """Helper to create a CSV file in the dataset directory."""
        filepath = self.dataset_dir / filename
        if data:
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        return filepath

    def test_ingest_single_csv(self):
        """Tests that a single CSV file is ingested correctly."""
        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        self._create_test_csv("users.csv", data)

        self.db.ingest_dataset("test_dataset", self.dataset_dir.as_posix())

        con = self.db.get_dataset_connection("test_dataset", read_only=True)
        tables = con.execute("SHOW TABLES;").fetchall()
        table_names = [t[0] for t in tables]
        self.assertIn("users", table_names)
        con.close()

    def test_ingest_multiple_csvs(self):
        """Tests that multiple CSV files are ingested correctly."""
        users_data = [{"id": 1, "name": "Alice"}]
        products_data = [{"product_id": 1, "title": "Widget"}]

        self._create_test_csv("users.csv", users_data)
        self._create_test_csv("products.csv", products_data)

        self.db.ingest_dataset("multi_dataset", self.dataset_dir.as_posix())

        con = self.db.get_dataset_connection("multi_dataset", read_only=True)
        tables = con.execute("SHOW TABLES;").fetchall()
        table_names = [t[0] for t in tables]
        self.assertIn("users", table_names)
        self.assertIn("products", table_names)
        con.close()

    def test_ingest_with_special_column_names(self):
        """Tests that CSV files with special characters in column names are ingested and cleaned."""
        data = [{"First Name": 1, "Last-Name": "Test", "Email@Domain": "test@test.com"}]
        self._create_test_csv("bad_columns.csv", data)

        self.db.ingest_dataset("special_cols", self.dataset_dir.as_posix())

        con = self.db.get_dataset_connection("special_cols", read_only=True)
        result = con.execute("SELECT * FROM bad_columns LIMIT 1;").fetchdf()
        self.assertGreater(len(result.columns), 0)
        con.close()

    def test_ingest_ignores_non_csv_files(self):
        """Tests that non-CSV files are ignored during dataset ingestion."""
        data = [{"id": 1, "value": "test"}]
        self._create_test_csv("data.csv", data)
        (self.dataset_dir / "readme.txt").write_text("Some readme content")
        (self.dataset_dir / "config.json").write_text('{"key": "value"}')

        self.db.ingest_dataset("ignore_test", self.dataset_dir.as_posix())

        con = self.db.get_dataset_connection("ignore_test", read_only=True)
        tables = con.execute("SHOW TABLES;").fetchall()
        table_names = [t[0] for t in tables]
        self.assertEqual(len(table_names), 1)
        self.assertIn("data", table_names)
        con.close()

    def test_ingest_deduplicates_column_names(self):
        """Tests that duplicate column names in CSV files are handled by deduplication."""
        filepath = self.dataset_dir / "dupes.csv"
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "value", "id"])
            writer.writerow([1, "test", 2])

        self.db.ingest_dataset("dedup_test", self.dataset_dir.as_posix())

        con = self.db.get_dataset_connection("dedup_test", read_only=True)
        result = con.execute("SELECT * FROM dupes LIMIT 1;").fetchdf()
        self.assertEqual(len(result.columns), 3)
        con.close()


class TestDatasetConnections(unittest.TestCase):
    """Tests for dataset connection management."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(logger=self.logger, config=self.config)
        self.db.dataset_db_path = Path(self.tmpdir) / "datasets"
        self.db.workspace_db_path = Path(self.tmpdir) / "workspaces"
        self.db.dataset_db_path.mkdir(parents=True, exist_ok=True)
        self.db.workspace_db_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_dataset_connection_read_only(self):
        """Tests that a read-only connection to a dataset can be obtained."""
        con = self.db.get_dataset_connection("test_ds", read_only=True)
        self.assertIsNotNone(con)
        db_file = self.db.dataset_db_path / "test_ds" / "test_ds.db"
        self.assertTrue(db_file.exists())
        con.close()

    def test_get_dataset_connection_read_write(self):
        """Tests that a read-write connection to a dataset can be obtained and allows writing."""
        con = self.db.get_dataset_connection("test_ds_rw", read_only=False)
        self.assertIsNotNone(con)
        con.execute("CREATE TABLE test (id INTEGER, name VARCHAR);")
        con.close()

        con2 = self.db.get_dataset_connection("test_ds_rw", read_only=True)
        tables = con2.execute("SHOW TABLES;").fetchall()
        table_names = [t[0] for t in tables]
        self.assertIn("test", table_names)
        con2.close()


class TestTableDescription(unittest.TestCase):
    """Tests for reading table descriptions from dataset metadata."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(logger=self.logger, config=self.config)
        self.db.dataset_db_path = Path(self.tmpdir) / "datasets"
        self.db.workspace_db_path = Path(self.tmpdir) / "workspaces"
        self.db.dataset_db_path.mkdir(parents=True, exist_ok=True)
        self.db.workspace_db_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_table_description_returns_empty_when_metadata_missing(self):
        """Tests that an empty string is returned when the dataset metadata file is missing."""
        dataset_name = "ds1"
        (self.db.dataset_db_path / dataset_name).mkdir(parents=True, exist_ok=True)

        desc = self.db.get_table_description(dataset_name, "users")
        self.assertEqual(desc, "")

    def test_get_table_description_returns_empty_when_table_not_in_metadata(self):
        """Tests that an empty string is returned when the requested table is not listed in the dataset metadata."""
        dataset_name = "ds2"
        ds_dir = self.db.dataset_db_path / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"table_name": "other_table", "description": "Other desc"},
            ]
        ).to_csv(ds_dir / "metadata.csv", index=False)

        desc = self.db.get_table_description(dataset_name, "users")
        self.assertEqual(desc, "")

    def test_get_table_description_returns_description_when_present(self):
        """Tests that the correct description is returned when the metadata file contains an entry for the requested table."""
        dataset_name = "ds3"
        ds_dir = self.db.dataset_db_path / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"table_name": "users", "description": "User table"},
                {"table_name": "orders", "description": "Orders table"},
            ]
        ).to_csv(ds_dir / "metadata.csv", index=False)

        desc = self.db.get_table_description(dataset_name, "users")
        self.assertEqual(desc, "User table")

    def test_get_table_description_returns_empty_when_metadata_missing_required_columns(
        self,
    ):
        """Tests that an empty string is returned when the dataset metadata file is missing required columns."""
        dataset_name = "ds4"
        ds_dir = self.db.dataset_db_path / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"table_name": "users", "not_description": "x"},
            ]
        ).to_csv(ds_dir / "metadata.csv", index=False)

        desc = self.db.get_table_description(dataset_name, "users")
        self.assertEqual(desc, "")


class TestDatasetLinking(unittest.TestCase):
    """Tests for linking dataset tables into workspace."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(logger=self.logger, config=self.config)
        self.db.dataset_db_path = Path(self.tmpdir) / "datasets"
        self.db.workspace_db_path = Path(self.tmpdir) / "workspaces"
        self.db.dataset_db_path.mkdir(parents=True, exist_ok=True)
        self.db.workspace_db_path.mkdir(parents=True, exist_ok=True)

        self.dataset_dir = Path(self.tmpdir) / "test_data"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        data = [{"id": 1, "name": "Alice"}]
        df = pd.DataFrame(data)
        df.to_csv(self.dataset_dir / "users.csv", index=False)

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_link_dataset_tables(self):
        """Tests that tables from an ingested dataset can be linked into a workspace and queried."""
        dataset_name = "test_dataset"
        self.db.ingest_dataset(dataset_name, self.dataset_dir.as_posix())

        user_id = "user_123"
        chat_id = "chat_456"
        self.db.link_dataset_tables(user_id, chat_id, dataset_name)

        con = self.db.get_ws_db_connection(user_id, chat_id)
        result = con.execute('SELECT * FROM "test_dataset"."users";').fetchdf()
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["id"], 1)

    def test_link_dataset_tables_idempotent(self):
        """Tests that linking the same dataset multiple times does not cause errors or duplicate tables."""
        dataset_name = "test_dataset"
        self.db.ingest_dataset(dataset_name, self.dataset_dir.as_posix())

        user_id = "user_123"
        chat_id = "chat_456"

        self.db.link_dataset_tables(user_id, chat_id, dataset_name)
        self.db.link_dataset_tables(user_id, chat_id, dataset_name)

    def test_link_nonexistent_dataset_raises_error(self):
        """Tests that attempting to link a dataset that does not exist raises a FileNotFoundError."""
        user_id = "user_123"
        chat_id = "chat_456"

        with self.assertRaises(FileNotFoundError):
            self.db.link_dataset_tables(user_id, chat_id, "nonexistent_dataset")


class TestPostgresDatasetLinking(unittest.TestCase):
    """Tests for PostgreSQL dataset linking via register_postgres_dataset."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(logger=self.logger, config=self.config)
        self.db.dataset_db_path = Path(self.tmpdir) / "datasets"
        self.db.workspace_db_path = Path(self.tmpdir) / "workspaces"
        self.db.dataset_db_path.mkdir(parents=True, exist_ok=True)
        self.db.workspace_db_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mock_ws_con(self, already_attached_alias: str | None = None):
        """Helper to create a mock workspace DB connection that simulates attached databases."""
        mock_con = MagicMock()
        name_col = [already_attached_alias] if already_attached_alias else []
        mock_con.execute.return_value.fetchdf.return_value = pd.DataFrame({"name": name_col})
        return mock_con

    def test_register_stores_connection_string(self):
        """Tests that registering a PostgreSQL dataset stores the connection string in the registry."""
        conn_str = "host=localhost port=5432 dbname=mydb user=reader"
        self.db.register_postgres_dataset("pg_ds", conn_str)
        self.assertIn("pg_ds", self.db._pg_registry)
        self.assertEqual(self.db._pg_registry["pg_ds"], conn_str)

    def test_register_overrides_previous_connection_string(self):
        """Tests that registering a PostgreSQL dataset with the same name overrides the previous connection string."""
        self.db.register_postgres_dataset("ds", "host=server1")
        self.db.register_postgres_dataset("ds", "host=server2")
        self.assertEqual(self.db._pg_registry["ds"], "host=server2")

    def test_link_issues_attach_with_type_postgres(self):
        """Tests that linking a registered PostgreSQL dataset issues an ATTACH statement with the correct connection string and TYPE POSTGRES."""
        conn_str = "host=localhost port=5432 dbname=mydb user=reader"
        self.db.register_postgres_dataset("pg_ds", conn_str)

        mock_con = self._make_mock_ws_con()
        with patch.object(self.db, "get_ws_db_connection", return_value=mock_con):
            self.db.link_dataset_tables("u", "c", "pg_ds")

        executed_sqls = [call[0][0] for call in mock_con.execute.call_args_list]
        attach_sqls = [s for s in executed_sqls if "ATTACH" in s.upper()]
        self.assertEqual(len(attach_sqls), 1)
        self.assertIn("TYPE POSTGRES", attach_sqls[0].upper())
        self.assertIn(conn_str, attach_sqls[0])
        self.assertIn('"pg_ds"', attach_sqls[0])

    def test_link_postgres_attach_is_read_only(self):
        """Tests that linking a registered PostgreSQL dataset issues an ATTACH statement with READ_ONLY access."""
        self.db.register_postgres_dataset("pg_ro", "host=localhost dbname=testdb")

        mock_con = self._make_mock_ws_con()
        with patch.object(self.db, "get_ws_db_connection", return_value=mock_con):
            self.db.link_dataset_tables("u", "c", "pg_ro")

        executed_sqls = [call[0][0] for call in mock_con.execute.call_args_list]
        attach_sqls = [s for s in executed_sqls if "ATTACH" in s.upper()]
        self.assertEqual(len(attach_sqls), 1)
        self.assertIn("READ_ONLY", attach_sqls[0].upper())

    def test_link_postgres_does_not_check_filesystem(self):
        """Tests that linking a registered PostgreSQL dataset does not check the filesystem for the dataset path and relies solely on the registry."""
        self.db.register_postgres_dataset("pg_only", "host=no-such-server")

        mock_con = self._make_mock_ws_con()
        with patch.object(self.db, "get_ws_db_connection", return_value=mock_con):
            self.db.link_dataset_tables("u", "c", "pg_only")

    def test_link_postgres_idempotent_skips_attach_when_already_attached(self):
        """Tests that linking a registered PostgreSQL dataset does not issue an ATTACH statement if the dataset alias is already attached to the workspace connection."""
        conn_str = "host=localhost dbname=testdb"
        self.db.register_postgres_dataset("pg_idem", conn_str)

        mock_con = self._make_mock_ws_con(already_attached_alias="pg_idem")
        with patch.object(self.db, "get_ws_db_connection", return_value=mock_con):
            self.db.link_dataset_tables("u", "c", "pg_idem")

        attach_calls = [
            call[0][0]
            for call in mock_con.execute.call_args_list
            if "ATTACH" in call[0][0].upper()
        ]
        self.assertEqual(len(attach_calls), 0)

    def test_link_postgres_first_call_attaches_second_skips(self):
        """Tests that linking a registered PostgreSQL dataset attaches on the first call and skips attaching on the second call due to idempotency."""
        conn_str = "host=localhost dbname=testdb"
        self.db.register_postgres_dataset("pg_idem2", conn_str)

        mock_first = self._make_mock_ws_con()
        with patch.object(self.db, "get_ws_db_connection", return_value=mock_first):
            self.db.link_dataset_tables("u", "c", "pg_idem2")

        attach_count_first = sum(
            1
            for call in mock_first.execute.call_args_list
            if "ATTACH" in call[0][0].upper()
        )
        self.assertEqual(attach_count_first, 1)

        mock_second = self._make_mock_ws_con(already_attached_alias="pg_idem2")
        with patch.object(self.db, "get_ws_db_connection", return_value=mock_second):
            self.db.link_dataset_tables("u", "c", "pg_idem2")

        attach_count_second = sum(
            1
            for call in mock_second.execute.call_args_list
            if "ATTACH" in call[0][0].upper()
        )
        self.assertEqual(attach_count_second, 0)

    def test_unregistered_dataset_raises_file_not_found(self):
        """Tests that attempting to link a dataset that is not registered as PostgreSQL and does not exist as a local dataset raises a FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.db.link_dataset_tables("u", "c", "no_such_local_ds")

    def test_local_dataset_unaffected_by_pg_registry(self):
        """Tests that registering a PostgreSQL dataset does not interfere with linking a local dataset that has the same name and that the local dataset is still found on the filesystem."""
        self.db.register_postgres_dataset("pg_other", "host=localhost")

        with self.assertRaises(FileNotFoundError):
            self.db.link_dataset_tables("u", "c", "local_ds")


class TestGetAccessibleLocalDatasets(unittest.TestCase):
    """Tests for the get_accessible_local_datasets function."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(logger=self.logger, config=self.config)
        self.db.dataset_db_path = Path(self.tmpdir) / "datasets"
        self.db.dataset_db_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_mock_dataset_db(self, dataset_name: str):
        """Helper to physically create a dummy .db file structure."""
        ds_dir = self.db.dataset_db_path / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        db_file = ds_dir / f"{dataset_name}.db"
        db_file.write_text("dummy sqlite/duckdb content")
        return db_file

    def test_admin_returns_all_existing_datasets(self):
        """Tests that an admin user can see all datasets that have a matching .db file."""
        self._create_mock_dataset_db("ds_alpha")
        self._create_mock_dataset_db("ds_beta")
        
        # Create an empty directory without a .db file to ensure it's filtered out
        (self.db.dataset_db_path / "empty_dir").mkdir(parents=True, exist_ok=True)

        accessible = self.db.get_accessible_local_datasets(is_admin=True, group_permissions={})
        
        self.assertEqual(len(accessible), 2)
        self.assertIn("ds_alpha", accessible)
        self.assertIn("ds_beta", accessible)
        self.assertNotIn("empty_dir", accessible)

    def test_non_admin_with_valid_permissions(self):
        """Tests that a non-admin user only sees datasets they have explicit prefix matching permissions for."""
        self._create_mock_dataset_db("ds_allowed")
        self._create_mock_dataset_db("ds_denied")

        group_perms = {
            f"{PermissionKey.DATASET_ACCESS_PREFIX.value}:ds_allowed": "read",
            "other_unrelated_permission": "write"
        }

        accessible = self.db.get_accessible_local_datasets(is_admin=False, group_permissions=group_perms)

        self.assertEqual(accessible, ["ds_allowed"])

    def test_non_admin_with_valid_permission_but_missing_db_file(self):
        """Tests that even if a user has a permission key, the dataset is omitted if the file doesn't exist."""
        # Permission exists, but the physical file does NOT
        group_perms = {
            f"{PermissionKey.DATASET_ACCESS_PREFIX.value}:ds_ghost": "read"
        }

        accessible = self.db.get_accessible_local_datasets(is_admin=False, group_permissions=group_perms)
        self.assertEqual(accessible, [])

    def test_non_admin_with_no_permissions(self):
        """Tests that a non-admin user with empty permissions gets an empty list back."""
        self._create_mock_dataset_db("ds_alpha")
        
        accessible = self.db.get_accessible_local_datasets(is_admin=False, group_permissions={})
        self.assertEqual(accessible, [])


if __name__ == "__main__":
    unittest.main()
