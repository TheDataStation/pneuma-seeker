import csv
import logging
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from pneuma_seeker.provenance.graph import ProvenanceGraph, ProvenanceNode
from pneuma_seeker.services.core.conductor.state import ConductorState
from pneuma_seeker.services.db.main import PneumaDB
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Table,
    Text,
)
from pneuma_seeker.shared.schemas.db.document_type import DocumentType
from pneuma_seeker.shared.schemas.language_model.role import Role


class TestPneumaDBInit(unittest.TestCase):
    """Tests for PneumaDB initialization."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_init_with_default_paths(self):
        """Test initialization with default paths."""
        db = PneumaDB(logger=self.logger, config=self.config)
        self.assertIsNotNone(db.dataset_db_path)
        self.assertIsNotNone(db.workspace_db_path)
        db.close_all_connections()

    def test_init_with_custom_paths(self):
        """Test initialization with custom paths."""
        dataset_path = "custom_datasets"
        workspace_path = "custom_workspaces"
        db = PneumaDB(
            logger=self.logger,
            config=self.config,
            dataset_db_path=dataset_path,
            workspace_db_path=workspace_path,
        )
        self.assertTrue(str(db.dataset_db_path).endswith(dataset_path))
        self.assertTrue(str(db.workspace_db_path).endswith(workspace_path))
        db.close_all_connections()

    def test_init_creates_directories(self):
        """Test that initialization creates required directories."""
        dataset_path = os.path.join(self.tmpdir, "datasets")
        workspace_path = os.path.join(self.tmpdir, "workspaces")
        db = PneumaDB(
            logger=self.logger,
            config=self.config,
            dataset_db_path=dataset_path,
            workspace_db_path=workspace_path,
        )
        db.close_all_connections()


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

        # Create a test dataset directory
        self.dataset_dir = Path(self.tmpdir) / "test_data"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_test_csv(self, filename: str, data: list[dict]):
        """Helper to create test CSV files."""
        filepath = self.dataset_dir / filename
        if data:
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        return filepath

    def test_ingest_single_csv(self):
        """Test ingesting a single CSV file."""
        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        self._create_test_csv("users.csv", data)

        self.db.ingest_dataset("test_dataset", self.dataset_dir.as_posix())

        # Verify table was created
        con = self.db.get_dataset_connection("test_dataset", read_only=True)
        tables = con.execute("SHOW TABLES;").fetchall()
        table_names = [t[0] for t in tables]
        self.assertIn("users", table_names)
        con.close()

    def test_ingest_multiple_csvs(self):
        """Test ingesting multiple CSV files."""
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
        """Test that column names are cleaned during ingestion."""
        data = [{"First Name": 1, "Last-Name": "Test", "Email@Domain": "test@test.com"}]
        self._create_test_csv("bad_columns.csv", data)

        self.db.ingest_dataset("special_cols", self.dataset_dir.as_posix())

        con = self.db.get_dataset_connection("special_cols", read_only=True)
        result = con.execute("SELECT * FROM bad_columns LIMIT 1;").fetchdf()
        # Columns should be cleaned
        self.assertGreater(len(result.columns), 0)
        con.close()

    def test_ingest_ignores_non_csv_files(self):
        """Test that non-CSV files are ignored during ingestion."""
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
        """Test that duplicate column names are deduplicated."""
        # Create a CSV with duplicate column names manually
        filepath = self.dataset_dir / "dupes.csv"
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "value", "id"])  # Duplicate 'id'
            writer.writerow([1, "test", 2])

        self.db.ingest_dataset("dedup_test", self.dataset_dir.as_posix())

        con = self.db.get_dataset_connection("dedup_test", read_only=True)
        result = con.execute("SELECT * FROM dupes LIMIT 1;").fetchdf()
        # Should have 3 columns with 'id' deduplicated
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
        """Test getting a read-only connection to a dataset."""
        con = self.db.get_dataset_connection("test_ds", read_only=True)
        self.assertIsNotNone(con)
        # Verify it's read-only by checking the database file exists
        db_file = self.db.dataset_db_path / "test_ds" / "test_ds.db"
        self.assertTrue(db_file.exists())
        con.close()

    def test_get_dataset_connection_read_write(self):
        """Test getting a read-write connection to a dataset."""
        con = self.db.get_dataset_connection("test_ds_rw", read_only=False)
        self.assertIsNotNone(con)
        # Create a test table
        con.execute("CREATE TABLE test (id INTEGER, name VARCHAR);")
        con.close()

        # Verify table persists
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
        dataset_name = "ds1"
        (self.db.dataset_db_path / dataset_name).mkdir(parents=True, exist_ok=True)

        desc = self.db.get_table_description(dataset_name, "users")
        self.assertEqual(desc, "")

    def test_get_table_description_returns_empty_when_table_not_in_metadata(self):
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
        dataset_name = "ds4"
        ds_dir = self.db.dataset_db_path / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Missing 'description' column
        pd.DataFrame(
            [
                {"table_name": "users", "not_description": "x"},
            ]
        ).to_csv(ds_dir / "metadata.csv", index=False)

        desc = self.db.get_table_description(dataset_name, "users")
        self.assertEqual(desc, "")


class TestWorkspaceConnections(unittest.TestCase):
    """Tests for workspace connection management."""

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

    def test_get_ws_db_connection_creates_new(self):
        """Test that getting a new workspace connection initializes tables."""
        user_id = "user_123"
        chat_id = "chat_456"

        con = self.db.get_ws_db_connection(user_id, chat_id)
        self.assertIsNotNone(con)

        # Verify tables were created
        tables = con.execute("SHOW TABLES;").fetchdf()
        table_names = tables["name"].tolist()

        expected_tables = [
            "chat_history",
            "conductor_state",
            "documents",
            "document_metadata",
            "state_document_roles",
            "provenance_nodes",
            "provenance_edges",
        ]
        for expected in expected_tables:
            self.assertIn(expected, table_names)

    def test_get_ws_db_connection_caching(self):
        """Test that workspace connections are cached."""
        user_id = "user_123"
        chat_id = "chat_456"

        con1 = self.db.get_ws_db_connection(user_id, chat_id)
        con2 = self.db.get_ws_db_connection(user_id, chat_id)

        # Should be the same connection object
        self.assertIs(con1, con2)

    def test_get_ws_db_connection_multiple_users(self):
        """Test that different users get different connections."""
        con1 = self.db.get_ws_db_connection("user_1", "chat_1")
        con2 = self.db.get_ws_db_connection("user_2", "chat_1")

        # Should be different connections
        self.assertIsNot(con1, con2)

    def test_close_workspace_connection(self):
        """Test closing a specific workspace connection."""
        user_id = "user_123"
        chat_id = "chat_456"

        con = self.db.get_ws_db_connection(user_id, chat_id)
        self.assertIn((user_id, chat_id), self.db._conn_cache)

        self.db.close_workspace_connection(user_id, chat_id)
        self.assertNotIn((user_id, chat_id), self.db._conn_cache)

    def test_close_all_connections(self):
        """Test closing all workspace connections."""
        self.db.get_ws_db_connection("user_1", "chat_1")
        self.db.get_ws_db_connection("user_2", "chat_2")
        self.db.get_ws_db_connection("user_3", "chat_3")

        self.assertEqual(len(self.db._conn_cache), 3)

        self.db.close_all_connections()
        self.assertEqual(len(self.db._conn_cache), 0)


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

        # Create a test dataset
        self.dataset_dir = Path(self.tmpdir) / "test_data"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        data = [{"id": 1, "name": "Alice"}]
        df = pd.DataFrame(data)
        df.to_csv(self.dataset_dir / "users.csv", index=False)

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_link_dataset_tables(self):
        """Test linking a dataset into a workspace."""
        dataset_name = "test_dataset"
        self.db.ingest_dataset(dataset_name, self.dataset_dir.as_posix())

        user_id = "user_123"
        chat_id = "chat_456"
        self.db.link_dataset_tables(user_id, chat_id, dataset_name)

        # Verify we can query the linked tables
        con = self.db.get_ws_db_connection(user_id, chat_id)
        result = con.execute('SELECT * FROM "test_dataset"."users";').fetchdf()
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["id"], 1)

    def test_link_dataset_tables_idempotent(self):
        """Test that linking the same dataset twice is safe."""
        dataset_name = "test_dataset"
        self.db.ingest_dataset(dataset_name, self.dataset_dir.as_posix())

        user_id = "user_123"
        chat_id = "chat_456"

        # Link twice - should not raise an error
        self.db.link_dataset_tables(user_id, chat_id, dataset_name)
        self.db.link_dataset_tables(user_id, chat_id, dataset_name)

    def test_link_nonexistent_dataset_raises_error(self):
        """Test that linking a non-existent dataset raises FileNotFoundError."""
        user_id = "user_123"
        chat_id = "chat_456"

        with self.assertRaises(FileNotFoundError):
            self.db.link_dataset_tables(user_id, chat_id, "nonexistent_dataset")


class TestQueryExecution(unittest.TestCase):
    """Tests for SQL query execution."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(logger=self.logger, config=self.config)
        self.db.dataset_db_path = Path(self.tmpdir) / "datasets"
        self.db.workspace_db_path = Path(self.tmpdir) / "workspaces"
        self.db.dataset_db_path.mkdir(parents=True, exist_ok=True)
        self.db.workspace_db_path.mkdir(parents=True, exist_ok=True)

        self.user_id = "user_test"
        self.chat_id = "chat_test"

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_execute_simple_query(self):
        """Test executing a simple SQL query."""
        result = self.db.execute_query(self.user_id, self.chat_id, "SELECT 1 as value;")
        self.assertEqual(result.iloc[0]["value"], 1)

    def test_execute_query_with_params(self):
        """Test executing a query with parameters."""
        con = self.db.get_ws_db_connection(self.user_id, self.chat_id)
        con.execute("CREATE TABLE test_table (id INTEGER, name VARCHAR);")
        con.execute("INSERT INTO test_table VALUES (1, 'Alice');")
        result = self.db.execute_query(
            self.user_id,
            self.chat_id,
            "SELECT * FROM test_table WHERE id = ?;",
            (1,),
        )
        con.close()
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Alice")

    def test_execute_query_returns_dataframe(self):
        """Test that execute_query returns a DataFrame."""
        result = self.db.execute_query(
            self.user_id, self.chat_id, "SELECT 42 as answer;"
        )
        self.assertIsInstance(result, pd.DataFrame)


class TestPersistDf(unittest.TestCase):
    """Tests for persist_df helper."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(logger=self.logger, config=self.config)
        self.db.dataset_db_path = Path(self.tmpdir) / "datasets"
        self.db.workspace_db_path = Path(self.tmpdir) / "workspaces"
        self.db.dataset_db_path.mkdir(parents=True, exist_ok=True)
        self.db.workspace_db_path.mkdir(parents=True, exist_ok=True)

        self.user_id = "user_doc"
        self.chat_id = "chat_doc"

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_persist_df_creates_table(self):
        """DataFrames should be persisted as workspace DB tables."""
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})

        self.db.persist_df(
            self.user_id,
            self.chat_id,
            df,
            "external_table",
            overwrite_content=False,
        )

        con = self.db.get_ws_db_connection(self.user_id, self.chat_id)
        tables = con.execute("SHOW TABLES;").fetchdf()
        table_names = tables["name"].tolist()
        self.assertIn("external_table", table_names)

        result = con.execute('SELECT * FROM "external_table" ORDER BY id;').fetchdf()
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["name"], "A")

    def test_persist_df_does_not_overwrite_when_false(self):
        """Tables should not be overwritten when overwrite_content=False."""
        df_v1 = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})

        self.db.persist_df(
            self.user_id,
            self.chat_id,
            df_v1,
            "external_no_overwrite",
            overwrite_content=False,
        )

        df_v2 = pd.DataFrame({"id": [1, 2], "name": ["Z", "Y"]})

        self.db.persist_df(
            self.user_id,
            self.chat_id,
            df_v2,
            "external_no_overwrite",
            overwrite_content=False,
        )

        con = self.db.get_ws_db_connection(self.user_id, self.chat_id)
        result = con.execute(
            'SELECT * FROM "external_no_overwrite" ORDER BY id;'
        ).fetchdf()
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["name"], "A")

    def test_persist_df_overwrites_when_true(self):
        """Tables should be overwritten when overwrite_content=True."""
        df_v1 = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})

        self.db.persist_df(
            self.user_id,
            self.chat_id,
            df_v1,
            "external_overwrite",
            overwrite_content=True,
        )

        df_v2 = pd.DataFrame({"id": [1, 2], "name": ["Z", "Y"]})

        self.db.persist_df(
            self.user_id,
            self.chat_id,
            df_v2,
            "external_overwrite",
            overwrite_content=True,
        )

        con = self.db.get_ws_db_connection(self.user_id, self.chat_id)
        result = con.execute(
            'SELECT * FROM "external_overwrite" ORDER BY id;'
        ).fetchdf()
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["name"], "Z")


class TestSessionPersistence(unittest.TestCase):
    """Tests for session persistence and loading."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(logger=self.logger, config=self.config)
        self.db.dataset_db_path = Path(self.tmpdir) / "datasets"
        self.db.workspace_db_path = Path(self.tmpdir) / "workspaces"
        self.db.dataset_db_path.mkdir(parents=True, exist_ok=True)
        self.db.workspace_db_path.mkdir(parents=True, exist_ok=True)

        self.user_id = "user_test"
        self.chat_id = "chat_test"

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_persist_and_load_basic_session(self):
        """Test persisting and loading a basic chat session."""
        user_input = "What is the population?"
        system_response = "The population is 8 billion."

        conductor_state = ConductorState()
        conductor_state.is_T_materialized = True
        conductor_state.S = "result = df.sum()"
        conductor_state.is_S_executed = True

        provenance_graph = ProvenanceGraph(self.logger)

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            user_input,
            system_response,
            conductor_state,
            provenance_graph,
            [],
            [],
        )

        # Load and verify
        (
            chat_history,
            loaded_state,
            loaded_graph,
            retrieved,
            enumerated,
            web_search,
            web_crawl,
            join_paths,
        ) = self.db.load_session(self.user_id, self.chat_id)

        self.assertEqual(len(chat_history), 2)
        self.assertEqual(chat_history[0]["role"], Role.USER.value)
        self.assertEqual(chat_history[0]["content"], user_input)
        self.assertEqual(chat_history[1]["role"], Role.ASSISTANT.value)
        self.assertEqual(chat_history[1]["content"], system_response)

        self.assertEqual(loaded_state.is_T_materialized, True)
        self.assertEqual(loaded_state.S, "result = df.sum()")
        self.assertEqual(loaded_state.is_S_executed, True)

    def test_persist_session_with_documents(self):
        """Test persisting a session with documents."""
        conductor_state = ConductorState()
        provenance_graph = ProvenanceGraph(self.logger)

        table_1 = pd.DataFrame({"col1": [1, 2, 3]})
        table_2 = pd.DataFrame({"colA": ["a", "b"]})

        os.makedirs(os.path.join(self.tmpdir, "tables"), exist_ok=True)

        table_1.to_csv(os.path.join(self.tmpdir, "tables", "table_1.csv"), index=False)
        table_2.to_csv(os.path.join(self.tmpdir, "tables", "table_2.csv"), index=False)

        doc1 = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_1,
            metadata={"dataset_name": "test_ds"},
        )

        doc2 = Table(
            doc_id="table_2",
            retriever_type=RetrieverType.ENUMERATOR,
            content=table_2,
            metadata={"dataset_name": "test_ds"},
        )

        self.db.ingest_dataset("test_ds", os.path.join(self.tmpdir, "tables"))

        retrieved_tables: list[AbstractDocument] = [doc1]
        enumerated_tables: list[AbstractDocument] = [doc2]

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "Query",
            "Response",
            conductor_state,
            provenance_graph,
            retrieved_tables,
            enumerated_tables,
        )

        (
            chat_history,
            loaded_state,
            loaded_graph,
            loaded_retrieved,
            loaded_enumerated,
            web_search,
            web_crawl,
            join_paths,
        ) = self.db.load_session(self.user_id, self.chat_id)

        self.assertEqual(len(loaded_retrieved), 1)
        self.assertEqual(len(loaded_enumerated), 1)
        self.assertEqual(loaded_retrieved[0].doc_id, "table_1")
        self.assertEqual(loaded_enumerated[0].doc_id, "table_2")

        # Type integrity: retriever_type should be restored as the RetrieverType enum
        self.assertIsInstance(loaded_retrieved[0].retriever_type, RetrieverType)
        self.assertEqual(
            loaded_retrieved[0].retriever_type, RetrieverType.PNEUMA_RETRIEVER
        )
        self.assertIsInstance(loaded_enumerated[0].retriever_type, RetrieverType)
        self.assertEqual(loaded_enumerated[0].retriever_type, RetrieverType.ENUMERATOR)

    def test_persist_session_with_provenance_graph(self):
        """Test persisting a session with a provenance graph."""
        conductor_state = ConductorState()
        provenance_graph = ProvenanceGraph(self.logger)

        # Create some nodes
        node1 = ProvenanceNode(
            RetrieverType.PNEUMA_RETRIEVER,
            "df = load_data()",
            "Load data from source",
        )
        node2 = ProvenanceNode(
            RetrieverType.MATERIALIZER,
            "df = df.filter(...)",
            "Filter rows",
        )
        node3 = ProvenanceNode(
            RetrieverType.CONDUCTOR,
            "result = df.sum()",
            "Aggregate result",
        )

        provenance_graph.add_node(node1)
        provenance_graph.add_node(node2)
        provenance_graph.add_node(node3)
        provenance_graph.connect(node1, node2)
        provenance_graph.connect(node2, node3)

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "Query",
            "Response",
            conductor_state,
            provenance_graph,
            [],
            [],
        )

        (
            chat_history,
            loaded_state,
            loaded_graph,
            retrieved,
            enumerated,
            web_search,
            web_crawl,
            join_paths,
        ) = self.db.load_session(self.user_id, self.chat_id)

        # Should have 4 nodes: 1 default root + 3 added
        self.assertEqual(len(loaded_graph.nodes), 4)

        # Verify connections
        node_list = list(loaded_graph.nodes.values())
        # Find nodes by code
        nodes_by_code = {n.python_code: n for n in node_list}
        self.assertIn("df = load_data()", nodes_by_code)
        self.assertIn("df = df.filter(...)", nodes_by_code)
        self.assertIn("result = df.sum()", nodes_by_code)

        # Verify edge structure (loaded graph should preserve parent->child)
        self.assertIn(
            "df = df.filter(...)",
            [c.python_code for c in nodes_by_code["df = load_data()"].children],
        )
        self.assertIn(
            "result = df.sum()",
            [c.python_code for c in nodes_by_code["df = df.filter(...)"].children],
        )

    def test_persist_overwrites_previous_state_when_not_fine_grained(self):
        """When fine-grained tracking is disabled, state tables should be overwritten each persist."""
        conductor_state_1 = ConductorState()
        conductor_state_1.S = "state1"
        conductor_state_2 = ConductorState()
        conductor_state_2.S = "state2"

        # Prepare dataset and two different tables
        os.makedirs(os.path.join(self.tmpdir, "tables"), exist_ok=True)
        pd.DataFrame({"a": [1]}).to_csv(
            os.path.join(self.tmpdir, "tables", "table_1.csv"), index=False
        )
        pd.DataFrame({"b": [2]}).to_csv(
            os.path.join(self.tmpdir, "tables", "table_2.csv"), index=False
        )
        self.db.ingest_dataset("test_ds", os.path.join(self.tmpdir, "tables"))

        doc1 = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=pd.DataFrame({"a": [1]}),
            metadata={"dataset_name": "test_ds"},
        )
        doc2 = Table(
            doc_id="table_2",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=pd.DataFrame({"b": [2]}),
            metadata={"dataset_name": "test_ds"},
        )

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "U1",
            "A1",
            conductor_state_1,
            ProvenanceGraph(self.logger),
            [doc1],
            [],
        )
        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "U2",
            "A2",
            conductor_state_2,
            ProvenanceGraph(self.logger),
            [doc2],
            [],
        )

        con = self.db.get_ws_db_connection(self.user_id, self.chat_id)
        state_count = con.execute("SELECT COUNT(*) FROM conductor_state;").fetchone()
        docs_count = con.execute("SELECT COUNT(*) FROM documents;").fetchone()

        assert state_count is not None
        assert docs_count is not None
        self.assertEqual(state_count[0], 1)
        self.assertEqual(docs_count[0], 1)

        (
            chat_history,
            loaded_state,
            loaded_graph,
            loaded_retrieved,
            loaded_enumerated,
            loaded_web_search,
            loaded_web_crawl,
            loaded_join_paths,
        ) = self.db.load_session(self.user_id, self.chat_id)

        # chat_history accumulates even when state is overwritten
        self.assertEqual(len(chat_history), 4)
        self.assertEqual(loaded_state.S, "state2")
        self.assertEqual(len(loaded_retrieved), 1)
        self.assertEqual(loaded_retrieved[0].doc_id, "table_2")
        con.close()


class TestSessionPersistenceFineGrainedTracking(unittest.TestCase):
    """Tests persistence when fine-grained state tracking is enabled."""

    def setUp(self):
        self.config = Config()
        self.config.ENABLE_FINE_GRAINED_STATE_CHANGE_TRACKING = True
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(logger=self.logger, config=self.config)
        self.db.dataset_db_path = Path(self.tmpdir) / "datasets"
        self.db.workspace_db_path = Path(self.tmpdir) / "workspaces"
        self.db.dataset_db_path.mkdir(parents=True, exist_ok=True)
        self.db.workspace_db_path.mkdir(parents=True, exist_ok=True)

        self.user_id = "user_fg"
        self.chat_id = "chat_fg"

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fine_grained_keeps_multiple_states_and_loads_latest(self):
        state1 = ConductorState()
        state1.is_T_materialized = True
        state1.S = "print('state1')"
        state1.is_S_executed = True

        state2 = ConductorState()
        state2.is_T_materialized = False
        state2.S = "print('state2')"
        state2.is_S_executed = False

        graph1 = ProvenanceGraph(self.logger)
        graph2 = ProvenanceGraph(self.logger)

        table_1 = pd.DataFrame({"x": [10, 20]})
        table_2 = pd.DataFrame({"y": [30, 40]})

        doc1 = Table(
            doc_id="table_1",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_1,
            metadata={"dataset_name": "test_ds"},
        )
        doc2 = Table(
            doc_id="table_2",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table_2,
            metadata={"dataset_name": "test_ds"},
        )

        os.makedirs(os.path.join(self.tmpdir, "tables"), exist_ok=True)

        table_1.to_csv(os.path.join(self.tmpdir, "tables", "table_1.csv"), index=False)
        table_2.to_csv(os.path.join(self.tmpdir, "tables", "table_2.csv"), index=False)
        self.db.ingest_dataset("test_ds", os.path.join(self.tmpdir, "tables"))

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "U1",
            "A1",
            state1,
            graph1,
            [doc1],
            [],
        )
        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "U2",
            "A2",
            state2,
            graph2,
            [doc2],
            [],
        )

        con = self.db.get_ws_db_connection(self.user_id, self.chat_id)
        state_count = con.execute("SELECT COUNT(*) FROM conductor_state;").fetchone()
        assert state_count is not None
        self.assertEqual(state_count[0], 2)

        (
            chat_history,
            loaded_state,
            loaded_graph,
            loaded_retrieved,
            loaded_enumerated,
            loaded_web_search,
            loaded_web_crawl,
            loaded_join_paths,
        ) = self.db.load_session(self.user_id, self.chat_id)

        # chat_history still accumulates
        self.assertEqual(len(chat_history), 4)

        # load_session should return the latest state
        self.assertEqual(loaded_state.S, state2.S)
        self.assertEqual(loaded_state.is_S_executed, state2.is_S_executed)

        # retrieved docs should correspond to latest state only
        self.assertEqual(len(loaded_retrieved), 1)
        self.assertEqual(loaded_retrieved[0].doc_id, "table_2")
        self.assertIsInstance(loaded_retrieved[0].retriever_type, RetrieverType)

        con.close()

    def test_fine_grained_loads_latest_provenance_graph_only(self):
        """Ensure provenance_nodes/edges are loaded only for latest state_id."""
        # Two graphs with different non-root nodes
        graph1 = ProvenanceGraph(self.logger)
        graph2 = ProvenanceGraph(self.logger)

        root1 = graph1.get_node({"python_code": graph1.ROOT_NODE_CODE})
        root2 = graph2.get_node({"python_code": graph2.ROOT_NODE_CODE})
        assert root1 is not None
        assert root2 is not None

        n1 = ProvenanceNode(RetrieverType.USER, "x = 1", "state1 node")
        n2 = ProvenanceNode(RetrieverType.USER, "y = 2", "state2 node")
        graph1.add_node(n1)
        graph2.add_node(n2)
        graph1.connect(root1, n1)
        graph2.connect(root2, n2)

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "U1",
            "A1",
            ConductorState(),
            graph1,
            [],
            [],
        )

        # Avoid flakiness if timestamps have coarse resolution.
        time.sleep(0.01)

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "U2",
            "A2",
            ConductorState(),
            graph2,
            [],
            [],
        )

        (
            chat_history,
            loaded_state,
            loaded_graph,
            loaded_retrieved,
            loaded_enumerated,
            loaded_web_search,
            loaded_web_crawl,
            loaded_join_paths,
        ) = self.db.load_session(self.user_id, self.chat_id)

        loaded_codes = {n.python_code for n in loaded_graph.nodes.values()}
        self.assertIn("y = 2", loaded_codes)
        self.assertNotIn("x = 1", loaded_codes)

        # Edge should be present for latest graph
        loaded_node_by_code = {n.python_code: n for n in loaded_graph.nodes.values()}
        self.assertIn(
            "y = 2",
            [c.python_code for c in loaded_node_by_code[root2.python_code].children],
        )

    def test_web_result_metadata_roundtrip(self):
        """Ensure metadata keys/values persist and reload for Text documents."""
        conductor_state = ConductorState()
        provenance_graph = ProvenanceGraph(self.logger)

        web_search_doc = Text(
            doc_id="web_search_meta",
            retriever_type=RetrieverType.WEB_SEARCH,
            content="Search result content",
            metadata={"url": "https://example.com", "title": "Example"},
        )

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "Query",
            "Response",
            conductor_state,
            provenance_graph,
            [],
            [],
            web_search_result=web_search_doc,
        )

        (*_, loaded_web_search, loaded_web_crawl, __) = self.db.load_session(
            self.user_id, self.chat_id
        )
        assert loaded_web_search is not None
        self.assertEqual(loaded_web_search.metadata.get("url"), "https://example.com")
        self.assertEqual(loaded_web_search.metadata.get("title"), "Example")

    def test_last_node_id_roundtrip(self):
        """Ensure documents with last_node_id persist and reload correctly."""
        graph = ProvenanceGraph(self.logger)
        root = graph.get_node({"python_code": graph.ROOT_NODE_CODE})
        assert root is not None
        node = ProvenanceNode(RetrieverType.USER, "z = 3", "node for last_node_id")
        graph.add_node(node)
        graph.connect(root, node)

        os.makedirs(os.path.join(self.tmpdir, "tables"), exist_ok=True)
        pd.DataFrame({"z": [3]}).to_csv(
            os.path.join(self.tmpdir, "tables", "table_z.csv"), index=False
        )
        self.db.ingest_dataset("test_ds", os.path.join(self.tmpdir, "tables"))

        doc = Table(
            doc_id="table_z",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=pd.DataFrame({"z": [3]}),
            metadata={"dataset_name": "test_ds"},
            last_node_id=node.id,
        )

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "U",
            "A",
            ConductorState(),
            graph,
            [doc],
            [],
        )

        (
            _chat_history,
            _loaded_state,
            _loaded_graph,
            loaded_retrieved,
            _loaded_enumerated,
            _loaded_web_search,
            _loaded_web_crawl,
            _loaded_join_paths,
        ) = self.db.load_session(self.user_id, self.chat_id)
        self.assertEqual(len(loaded_retrieved), 1)
        self.assertEqual(loaded_retrieved[0].last_node_id, node.id)

    def test_load_raises_if_dataset_db_missing_for_table_doc(self):
        """If a Table doc references a missing dataset DB, load_session should fail loudly."""
        # Persist a Table doc pointing at a dataset that was never ingested
        doc = Table(
            doc_id="some_table",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=pd.DataFrame({"a": [1]}),
            metadata={"dataset_name": "missing_ds"},
        )

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "U",
            "A",
            ConductorState(),
            ProvenanceGraph(self.logger),
            [doc],
            [],
        )

        with self.assertRaises(FileNotFoundError):
            self.db.load_session(self.user_id, self.chat_id)

    def test_fine_grained_allows_reusing_doc_id(self):
        """Fine-grained mode should not fail when the same doc_id appears in multiple states."""
        doc_id = "web_search_same_id"

        doc_v1 = Text(
            doc_id=doc_id,
            retriever_type=RetrieverType.WEB_SEARCH,
            content="v1",
            metadata={"url": "https://example.com/v1"},
        )
        doc_v2 = Text(
            doc_id=doc_id,
            retriever_type=RetrieverType.WEB_SEARCH,
            content="v2",
            metadata={"url": "https://example.com/v2"},
        )

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "U1",
            "A1",
            ConductorState(),
            ProvenanceGraph(self.logger),
            [],
            [],
            web_search_result=doc_v1,
        )

        time.sleep(0.01)

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "U2",
            "A2",
            ConductorState(),
            ProvenanceGraph(self.logger),
            [],
            [],
            web_search_result=doc_v2,
        )

        con = self.db.get_ws_db_connection(self.user_id, self.chat_id)

        conductor_state_count = con.execute(
            "SELECT COUNT(*) FROM conductor_state;"
        ).fetchone()
        documents_count = con.execute("SELECT COUNT(*) FROM documents;").fetchone()

        assert conductor_state_count is not None
        assert documents_count is not None
        self.assertEqual(conductor_state_count[0], 2)
        self.assertEqual(documents_count[0], 1)

        (*_, loaded_web_search, __, ___) = self.db.load_session(
            self.user_id, self.chat_id
        )
        assert loaded_web_search is not None
        self.assertEqual(loaded_web_search.doc_id, doc_id)
        self.assertEqual(loaded_web_search.content, "v2")
        self.assertEqual(
            loaded_web_search.metadata.get("url"), "https://example.com/v2"
        )

    def test_persist_session_with_join_paths(self):
        """Test persisting a session with join paths."""
        conductor_state = ConductorState()
        provenance_graph = ProvenanceGraph(self.logger)
        join_paths_str = "table1.id = table2.fk_id"

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "Query",
            "Response",
            conductor_state,
            provenance_graph,
            [],
            [],
            join_paths=join_paths_str,
        )

        (
            chat_history,
            loaded_state,
            loaded_graph,
            retrieved,
            enumerated,
            web_search,
            web_crawl,
            loaded_join_paths,
        ) = self.db.load_session(self.user_id, self.chat_id)

        self.assertEqual(loaded_join_paths, join_paths_str)

    def test_persist_session_with_web_results(self):
        """Test persisting a session with web search and crawl results."""
        conductor_state = ConductorState()
        provenance_graph = ProvenanceGraph(self.logger)

        web_search_doc = Text(
            doc_id="web_search_1",
            retriever_type=RetrieverType.WEB_SEARCH,
            content="Search result content",
            metadata={"url": "https://example.com"},
        )

        web_crawl_doc = Text(
            doc_id="web_crawl_1",
            retriever_type=RetrieverType.WEB_CRAWL,
            content="Crawled page content",
            metadata={"url": "https://example.com/page"},
        )

        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "Query",
            "Response",
            conductor_state,
            provenance_graph,
            [],
            [],
            web_search_result=web_search_doc,
            web_crawl_result=web_crawl_doc,
        )

        (
            chat_history,
            loaded_state,
            loaded_graph,
            retrieved,
            enumerated,
            loaded_web_search,
            loaded_web_crawl,
            join_paths,
        ) = self.db.load_session(self.user_id, self.chat_id)

        self.assertIsNotNone(loaded_web_search)
        if loaded_web_search:
            self.assertEqual(loaded_web_search.doc_id, "web_search_1")
        self.assertIsNotNone(loaded_web_crawl)
        if loaded_web_crawl:
            self.assertEqual(loaded_web_crawl.doc_id, "web_crawl_1")

    def test_load_session_empty_workspace(self):
        """Test loading a session from an empty workspace."""
        (
            chat_history,
            conductor_state,
            provenance_graph,
            retrieved,
            enumerated,
            web_search,
            web_crawl,
            join_paths,
        ) = self.db.load_session(self.user_id, self.chat_id)

        self.assertEqual(chat_history, [])
        self.assertEqual(conductor_state.T, {})
        self.assertIsNotNone(provenance_graph)
        self.assertEqual(retrieved, [])
        self.assertEqual(enumerated, [])
        self.assertIsNone(web_search)
        self.assertIsNone(web_crawl)
        self.assertIsNone(join_paths)

    def test_persist_session_accumulates_previous(self):
        """Test that persisting a new session accumulates the previous one."""
        # First session
        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "First query",
            "First response",
            ConductorState(),
            ProvenanceGraph(self.logger),
            [],
            [],
        )

        # Second session
        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "Second query",
            "Second response",
            ConductorState(),
            ProvenanceGraph(self.logger),
            [],
            [],
        )

        (
            chat_history,
            loaded_state,
            loaded_graph,
            retrieved,
            enumerated,
            web_search,
            web_crawl,
            join_paths,
        ) = self.db.load_session(self.user_id, self.chat_id)

        # Should have both sessions' messages
        self.assertEqual(len(chat_history), 4)
        self.assertEqual(chat_history[0]["content"], "First query")
        self.assertEqual(chat_history[1]["content"], "First response")
        self.assertEqual(chat_history[2]["content"], "Second query")
        self.assertEqual(chat_history[3]["content"], "Second response")


class TestIntegration(unittest.TestCase):
    """Integration tests for the full workflow."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(logger=self.logger, config=self.config)
        self.db.dataset_db_path = Path(self.tmpdir) / "datasets"
        self.db.workspace_db_path = Path(self.tmpdir) / "workspaces"
        self.db.dataset_db_path.mkdir(parents=True, exist_ok=True)
        self.db.workspace_db_path.mkdir(parents=True, exist_ok=True)

        # Create test datasets
        self.dataset_dir = Path(self.tmpdir) / "test_data"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create sample data
        users_df = pd.DataFrame(
            {"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
        )
        users_df.to_csv(self.dataset_dir / "users.csv", index=False)

        orders_df = pd.DataFrame(
            {"order_id": [1, 2, 3], "user_id": [1, 1, 2], "amount": [100, 200, 150]}
        )
        orders_df.to_csv(self.dataset_dir / "orders.csv", index=False)

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_workflow(self):
        """Test a complete workflow: ingest, link, query, and persist."""
        dataset_name = "ecommerce"
        user_id = "user_123"
        chat_id = "chat_456"

        # 1. Ingest dataset
        self.db.ingest_dataset(dataset_name, self.dataset_dir.as_posix())

        # 2. Link into workspace
        self.db.link_dataset_tables(user_id, chat_id, dataset_name)

        # 3. Execute a query
        result = self.db.execute_query(
            user_id,
            chat_id,
            f'SELECT * FROM "{dataset_name}"."users" WHERE user_id = 1;',
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["name"], "Alice")

        # 4. Persist session
        conductor_state = ConductorState()
        conductor_state.is_T_materialized = True
        conductor_state.S = "SELECT * FROM users WHERE user_id = 1;"
        conductor_state.is_S_executed = True

        provenance_graph = ProvenanceGraph(self.logger)
        node = ProvenanceNode(
            RetrieverType.PNEUMA_RETRIEVER,
            'query = "SELECT * FROM users WHERE user_id = 1;"',
            "Retrieve user data",
        )
        provenance_graph.add_node(node)

        retrieved_doc = Table(
            doc_id="users",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=result,
            metadata={"dataset_name": dataset_name},
        )

        self.db.persist_session(
            user_id,
            chat_id,
            "Show me Alice's info",
            "Here is Alice's information",
            conductor_state,
            provenance_graph,
            [retrieved_doc],
            [],
        )

        # 5. Load session and verify
        (
            chat_history,
            loaded_state,
            loaded_graph,
            retrieved_tables,
            enumerated_tables,
            web_search,
            web_crawl,
            join_paths,
        ) = self.db.load_session(user_id, chat_id)

        self.assertEqual(len(chat_history), 2)
        self.assertEqual(chat_history[0]["content"], "Show me Alice's info")
        self.assertEqual(loaded_state.S, conductor_state.S)
        self.assertEqual(len(retrieved_tables), 1)
        self.assertEqual(retrieved_tables[0].doc_id, "users")

    def test_multi_user_isolation(self):
        """Test that different users' data is isolated."""
        dataset_name = "ecommerce"

        # Ingest dataset once
        self.db.ingest_dataset(dataset_name, self.dataset_dir.as_posix())

        # Two different users
        user1_id = "user_1"
        chat1_id = "chat_1"
        user2_id = "user_2"
        chat2_id = "chat_2"

        # Both link the same dataset
        self.db.link_dataset_tables(user1_id, chat1_id, dataset_name)
        self.db.link_dataset_tables(user2_id, chat2_id, dataset_name)

        # User 1 persists a session
        self.db.persist_session(
            user1_id,
            chat1_id,
            "User 1 query",
            "User 1 response",
            ConductorState(),
            ProvenanceGraph(self.logger),
            [],
            [],
        )

        # User 2 persists a different session
        self.db.persist_session(
            user2_id,
            chat2_id,
            "User 2 query",
            "User 2 response",
            ConductorState(),
            ProvenanceGraph(self.logger),
            [],
            [],
        )

        # Load and verify isolation
        (chat1_history, *_) = self.db.load_session(user1_id, chat1_id)
        (chat2_history, *_) = self.db.load_session(user2_id, chat2_id)

        self.assertEqual(chat1_history[0]["content"], "User 1 query")
        self.assertEqual(chat2_history[0]["content"], "User 2 query")


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
        """Returns a MagicMock workspace connection pre-configured for link_dataset_tables tests."""
        mock_con = MagicMock()
        name_col = [already_attached_alias] if already_attached_alias else []
        mock_con.execute.return_value.fetchdf.return_value = pd.DataFrame({"name": name_col})
        return mock_con

    def test_register_stores_connection_string(self):
        """register_postgres_dataset should store the connection string in the registry."""
        conn_str = "host=localhost port=5432 dbname=mydb user=reader"
        self.db.register_postgres_dataset("pg_ds", conn_str)
        self.assertIn("pg_ds", self.db._pg_registry)
        self.assertEqual(self.db._pg_registry["pg_ds"], conn_str)

    def test_register_overrides_previous_connection_string(self):
        """Re-registering a dataset name updates the stored connection string."""
        self.db.register_postgres_dataset("ds", "host=server1")
        self.db.register_postgres_dataset("ds", "host=server2")
        self.assertEqual(self.db._pg_registry["ds"], "host=server2")

    def test_link_issues_attach_with_type_postgres(self):
        """link_dataset_tables should ATTACH with TYPE postgres for PG-registered datasets."""
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
        """The postgres ATTACH should always include READ_ONLY."""
        self.db.register_postgres_dataset("pg_ro", "host=localhost dbname=testdb")

        mock_con = self._make_mock_ws_con()
        with patch.object(self.db, "get_ws_db_connection", return_value=mock_con):
            self.db.link_dataset_tables("u", "c", "pg_ro")

        executed_sqls = [call[0][0] for call in mock_con.execute.call_args_list]
        attach_sqls = [s for s in executed_sqls if "ATTACH" in s.upper()]
        self.assertEqual(len(attach_sqls), 1)
        self.assertIn("READ_ONLY", attach_sqls[0].upper())

    def test_link_postgres_does_not_check_filesystem(self):
        """Linking a PG-registered dataset must not raise FileNotFoundError."""
        self.db.register_postgres_dataset("pg_only", "host=no-such-server")

        mock_con = self._make_mock_ws_con()
        with patch.object(self.db, "get_ws_db_connection", return_value=mock_con):
            # Should not raise even though no local .db file exists
            self.db.link_dataset_tables("u", "c", "pg_only")

    def test_link_postgres_idempotent_skips_attach_when_already_attached(self):
        """If the alias is already in PRAGMA database_list, ATTACH is not called again."""
        conn_str = "host=localhost dbname=testdb"
        self.db.register_postgres_dataset("pg_idem", conn_str)

        # Simulate already attached (alias == clean_column_table_name("pg_idem") == "pg_idem")
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
        """First link_dataset_tables call ATTACHes; second call (alias already present) skips."""
        conn_str = "host=localhost dbname=testdb"
        self.db.register_postgres_dataset("pg_idem2", conn_str)

        # First call: not yet attached
        mock_first = self._make_mock_ws_con()
        with patch.object(self.db, "get_ws_db_connection", return_value=mock_first):
            self.db.link_dataset_tables("u", "c", "pg_idem2")

        attach_count_first = sum(
            1
            for call in mock_first.execute.call_args_list
            if "ATTACH" in call[0][0].upper()
        )
        self.assertEqual(attach_count_first, 1)

        # Second call: simulate already attached
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
        """Non-PG datasets still raise FileNotFoundError when the .db file is missing."""
        with self.assertRaises(FileNotFoundError):
            self.db.link_dataset_tables("u", "c", "no_such_local_ds")

    def test_local_dataset_unaffected_by_pg_registry(self):
        """Registering a PG dataset does not affect unrelated local dataset behaviour."""
        self.db.register_postgres_dataset("pg_other", "host=localhost")

        # "local_ds" is not in the registry, so it should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.db.link_dataset_tables("u", "c", "local_ds")


class TestTransactionRollback(unittest.TestCase):
    """Tests for transaction rollback and recovery in persist_session."""

    def setUp(self):
        self.config = Config()
        self.logger = logging.getLogger("test")
        self.tmpdir = tempfile.mkdtemp()
        self.db = PneumaDB(logger=self.logger, config=self.config)
        self.db.dataset_db_path = Path(self.tmpdir) / "datasets"
        self.db.workspace_db_path = Path(self.tmpdir) / "workspaces"
        self.db.dataset_db_path.mkdir(parents=True, exist_ok=True)
        self.db.workspace_db_path.mkdir(parents=True, exist_ok=True)
        self.user_id = "user_rollback"
        self.chat_id = "chat_rollback"

    def tearDown(self):
        self.db.close_all_connections()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rollback_on_error(self):
        """Test that persist_session rolls back and DB remains consistent after error."""
        conductor_state = ConductorState()
        provenance_graph = ProvenanceGraph(self.logger)

        table = pd.DataFrame({"a": [1, 2]})
        os.makedirs(os.path.join(self.tmpdir, "tables"), exist_ok=True)
        table.to_csv(os.path.join(self.tmpdir, "tables", "table.csv"), index=False)
        self.db.ingest_dataset("test_ds", os.path.join(self.tmpdir, "tables"))

        # Provide a document so persist_session will call __insert_document.
        # We'll force that call to raise to simulate a mid-transaction failure.
        failing_doc = Table(
            doc_id="doc_fail",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table,
            metadata={"dataset_name": "test_ds"},
        )

        # Patch __insert_document to raise an exception to simulate failure
        with patch.object(
            self.db,
            "_PneumaDB__insert_document",
            side_effect=Exception("Simulated failure"),
        ):
            # Should not raise, but should log and rollback
            self.db.persist_session(
                self.user_id,
                self.chat_id,
                "User input",
                "System response",
                conductor_state,
                provenance_graph,
                [failing_doc],
                [],
            )

        # After failure, DB should not have any documents, provenance_nodes, or conductor_state rows
        con = self.db.get_ws_db_connection(self.user_id, self.chat_id)
        chat_count = con.execute("SELECT COUNT(*) FROM chat_history;").fetchone()
        doc_count = con.execute("SELECT COUNT(*) FROM documents;").fetchone()
        node_count = con.execute("SELECT COUNT(*) FROM provenance_nodes;").fetchone()
        state_count = con.execute("SELECT COUNT(*) FROM conductor_state;").fetchone()

        assert isinstance(chat_count, tuple)
        assert isinstance(doc_count, tuple)
        assert isinstance(node_count, tuple)
        assert isinstance(state_count, tuple)

        # Chat history is intentionally persisted in its own transaction.
        self.assertEqual(chat_count[0], 2)
        self.assertEqual(doc_count[0], 0)
        self.assertEqual(node_count[0], 0)
        self.assertEqual(state_count[0], 0)
        con.close()

    def test_recovery_after_failure(self):
        """Test that after a failed persist_session, subsequent calls succeed and DB is consistent."""
        conductor_state = ConductorState()
        provenance_graph = ProvenanceGraph(self.logger)

        table = pd.DataFrame({"a": [1, 2]})
        os.makedirs(os.path.join(self.tmpdir, "tables"), exist_ok=True)
        table.to_csv(os.path.join(self.tmpdir, "tables", "table.csv"), index=False)
        self.db.ingest_dataset("test_ds", os.path.join(self.tmpdir, "tables"))

        failing_doc = Table(
            doc_id="doc_fail",
            retriever_type=RetrieverType.PNEUMA_RETRIEVER,
            content=table,
            metadata={"dataset_name": "test_ds"},
        )

        # Simulate failure
        with patch.object(
            self.db,
            "_PneumaDB__insert_document",
            side_effect=Exception("Simulated failure"),
        ):
            self.db.persist_session(
                self.user_id,
                self.chat_id,
                "User input",
                "System response",
                conductor_state,
                provenance_graph,
                [failing_doc],
                [],
            )

        # Now call persist_session normally
        self.db.persist_session(
            self.user_id,
            self.chat_id,
            "User input",
            "System response",
            conductor_state,
            provenance_graph,
            [],
            [],
        )

        # DB should now have chat_history and conductor_state rows
        con = self.db.get_ws_db_connection(self.user_id, self.chat_id)
        chat_count = con.execute("SELECT COUNT(*) FROM chat_history;").fetchone()
        state_count = con.execute("SELECT COUNT(*) FROM conductor_state;").fetchone()

        assert isinstance(chat_count, tuple)
        assert isinstance(state_count, tuple)

        # 2 messages from failed call + 2 from successful call
        self.assertEqual(chat_count[0], 4)
        self.assertEqual(state_count[0], 1)
        con.close()


if __name__ == "__main__":
    unittest.main()
